"""StrategyStore — 策略持久化存储器。

复用 MemoryStore 的 YAML Frontmatter + Markdown 文件格式，
实现策略的 CRUD、置信度衰减、应用追踪和关键词检索。

存储路径：.codeagent/strategies/
目录结构：
  {category}/{strategy_id}.md        # 活跃策略
  archived/{category}/{strategy_id}.md  # 已归档策略
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import frontmatter

from codeagent.context_engine.evolution.strategy_extractor import Strategy

logger = logging.getLogger(__name__)

# 置信度边界
_MIN_CONFIDENCE = 0.0
_MAX_CONFIDENCE = 1.0
_ARCHIVE_THRESHOLD = 0.2

# 衰减配置
_DECAY_DAYS = 30          # 超过 30 天未应用衰减 0.1
_ARCHIVE_DAYS = 90        # 超过 90 天直接归档

# 搜索字段权重
_FIELD_WEIGHTS: dict[str, float] = {
    "condition": 2.0,
    "action": 1.5,
    "rationale": 1.0,
}

# 有效类别目录
_VALID_CATEGORIES = {"workflow", "coding_style", "error_avoidance", "tool_usage"}


class StrategyStore:
    """策略存储器——复用 MemoryStore 的文件系统存储机制。

    存储路径：.codeagent/strategies/
    保存格式：YAML Frontmatter + Markdown（与 MemoryStore 一致）

    Args:
        base_path: 策略存储根目录（默认 .codeagent/strategies）
    """

    def __init__(self, base_path: str = ".codeagent/strategies") -> None:
        self._base_path = Path(base_path)

    # ── 公共接口 ──────────────────────────────────────────────

    def save(self, strategy: Strategy) -> str:
        """保存策略到文件系统。

        文件名格式：{category}/{strategy_id}.md
        写入完后更新索引。
        自动创建父目录。

        Args:
            strategy: Strategy 对象

        Returns:
            strategy_id
        """
        category = self._ensure_valid_category(strategy.category)
        file_path = self._base_path / category / f"{strategy.strategy_id}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._strategy_to_frontmatter(strategy)
        file_path.write_text(content, encoding="utf-8")

        logger.debug("Strategy saved: %s (%s)", strategy.strategy_id, category)
        return strategy.strategy_id

    def get(self, strategy_id: str) -> Optional[Strategy]:
        """按 ID 加载策略。

        先查活跃目录，再查 archived 目录。

        Args:
            strategy_id: 策略 ID

        Returns:
            Strategy 对象，不存在时返回 None
        """
        # 先查活跃目录
        for category in _VALID_CATEGORIES:
            file_path = self._base_path / category / f"{strategy_id}.md"
            if file_path.exists():
                return self._load_from_file(file_path)

        # 再查 archived 目录
        archived_base = self._base_path / "archived"
        if archived_base.exists():
            for category in _VALID_CATEGORIES:
                file_path = archived_base / category / f"{strategy_id}.md"
                if file_path.exists():
                    return self._load_from_file(file_path)

        return None

    def list_active(self) -> list[Strategy]:
        """列出置信度 >= 0.2 的所有活跃策略。

        Returns:
            活跃策略列表
        """
        results: list[Strategy] = []
        for category in _VALID_CATEGORIES:
            category_dir = self._base_path / category
            if not category_dir.is_dir():
                continue
            for fpath in sorted(category_dir.iterdir()):
                if fpath.suffix != ".md":
                    continue
                try:
                    strategy = self._load_from_file(fpath)
                    if strategy is not None and strategy.confidence >= _ARCHIVE_THRESHOLD:
                        results.append(strategy)
                except Exception as exc:
                    logger.warning("Failed to load active strategy %s: %s", fpath, exc)
        return results

    def list_archived(self) -> list[Strategy]:
        """列出已归档的策略（置信度 < 0.2 或手动归档）。

        Returns:
            已归档策略列表
        """
        results: list[Strategy] = []
        archived_base = self._base_path / "archived"
        if not archived_base.is_dir():
            return results
        for category in _VALID_CATEGORIES:
            category_dir = archived_base / category
            if not category_dir.is_dir():
                continue
            for fpath in sorted(category_dir.iterdir()):
                if fpath.suffix != ".md":
                    continue
                try:
                    strategy = self._load_from_file(fpath)
                    if strategy is not None:
                        results.append(strategy)
                except Exception as exc:
                    logger.warning("Failed to load archived strategy %s: %s", fpath, exc)
        return results

    def update_confidence(self, strategy_id: str, success: bool) -> None:
        """更新策略置信度。

        - success=True: confidence += 0.1（上限 1.0）
        - success=False: confidence -= 0.15（下限 0.0，低于 0.2 自动归档）

        Args:
            strategy_id: 策略 ID
            success: 是否成功
        """
        strategy = self.get(strategy_id)
        if strategy is None:
            logger.warning("Strategy %s not found, cannot update confidence", strategy_id)
            return

        if success:
            strategy.confidence = min(round(strategy.confidence + 0.1, 2), _MAX_CONFIDENCE)
        else:
            strategy.confidence = max(round(strategy.confidence - 0.15, 2), _MIN_CONFIDENCE)

        # 重新保存（先删除旧文件，再写入）
        self._delete_file(strategy_id)
        self.save(strategy)

        # 低于阈值时归档
        if strategy.confidence < _ARCHIVE_THRESHOLD:
            self._archive(strategy_id)

        logger.debug(
            "Strategy %s confidence updated to %.2f (success=%s)",
            strategy_id, strategy.confidence, success,
        )

    def increment_applied(self, strategy_id: str, success: bool) -> None:
        """记录应用结果。

        - applied_count += 1
        - success=True 时 success_count += 1

        Args:
            strategy_id: 策略 ID
            success: 应用是否成功
        """
        strategy = self.get(strategy_id)
        if strategy is None:
            logger.warning("Strategy %s not found, cannot increment applied", strategy_id)
            return

        strategy.applied_count += 1
        if success:
            strategy.success_count += 1
        strategy.last_applied = datetime.now()

        # 重新保存（先删除旧文件，再写入）
        self._delete_file(strategy_id)
        self.save(strategy)

        logger.debug(
            "Strategy %s applied: count=%d, success=%d",
            strategy_id, strategy.applied_count, strategy.success_count,
        )

    def apply_decay(self) -> int:
        """衰减所有活跃策略的置信度。

        - 超过 30 天未应用的策略：confidence -= 0.1
        - confidence < 0.2：移至 archived/
        - 超过 90 天未应用的策略：直接归档（无论 confidence）

        Returns:
            已衰减/归档的策略数量
        """
        now = datetime.now()
        decay_count = 0

        for strategy in self.list_active():
            last = strategy.last_applied or strategy.created_at
            days_since = (now - last).days

            should_archive = False

            # 超过 90 天直接归档
            if days_since >= _ARCHIVE_DAYS:
                should_archive = True
                decay_count += 1
                logger.debug("Strategy %s archived: %d days since last applied", strategy.strategy_id, days_since)
            # 超过 30 天衰减置信度
            elif days_since >= _DECAY_DAYS:
                old_confidence = strategy.confidence
                strategy.confidence = max(round(strategy.confidence - 0.1, 2), _MIN_CONFIDENCE)
                decay_count += 1

                # 更新文件
                self._delete_file(strategy.strategy_id)
                self.save(strategy)

                logger.debug(
                    "Strategy %s decayed: %.2f -> %.2f (%d days)",
                    strategy.strategy_id, old_confidence, strategy.confidence, days_since,
                )

                # 衰减后低于阈值则归档
                if strategy.confidence < _ARCHIVE_THRESHOLD:
                    should_archive = True

            if should_archive:
                self._archive(strategy.strategy_id)

        return decay_count

    def search(self, query: str, top_k: int = 5) -> list[Strategy]:
        """检索策略。

        关键词匹配 condition + action + rationale 字段。
        返回按 confidence × relevance_score 综合排序的结果。

        Args:
            query: 查询关键词
            top_k: 最多返回条数

        Returns:
            按综合分降序排列的策略列表
        """
        if not query.strip():
            return []

        query_words = self._tokenize(query)
        if not query_words:
            return []

        scored: list[tuple[Strategy, float]] = []
        for strategy in self.list_active():
            score = self._compute_search_score(query_words, strategy)
            if score > 0:
                scored.append((strategy, score))

        # 按综合分降序
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    def get_stats(self) -> dict:
        """返回策略统计信息。

        Returns:
            {"total": int, "active": int, "archived": int, "avg_confidence": float}
        """
        active = self.list_active()
        archived = self.list_archived()

        avg_conf = 0.0
        if active:
            avg_conf = sum(s.confidence for s in active) / len(active)

        return {
            "total": len(active) + len(archived),
            "active": len(active),
            "archived": len(archived),
            "avg_confidence": round(avg_conf, 4),
        }

    # ── 序列化 / 反序列化 ──────────────────────────────────────

    def _strategy_to_frontmatter(self, strategy: Strategy) -> str:
        """将 Strategy 序列化为 YAML Frontmatter + Markdown 格式。

        Args:
            strategy: Strategy 对象

        Returns:
            Frontmatter 格式的字符串
        """
        # 构造 Frontmatter 元数据（展平结构，不使用嵌套 metadata）
        meta = {
            "name": strategy.strategy_id,
            "type": "strategy",
            "title": strategy.title,
            "category": strategy.category,
            "confidence": strategy.confidence,
            "condition": strategy.condition,
            "action": strategy.action,
            "rationale": strategy.rationale,
            "applied_count": strategy.applied_count,
            "success_count": strategy.success_count,
            "created_at": strategy.created_at.isoformat(),
            "last_applied": strategy.last_applied.isoformat() if strategy.last_applied else None,
            "source_task_ids": strategy.source_task_ids,
        }

        body = (
            f"## 条件\n\n{strategy.condition}\n\n"
            f"## 建议动作\n\n{strategy.action}\n\n"
            f"## 理由\n\n{strategy.rationale}\n"
        )

        post = frontmatter.Post(body, **meta)
        return frontmatter.dumps(post)

    def _load_from_file(self, file_path: Path) -> Optional[Strategy]:
        """从 Frontmatter 文件加载 Strategy。

        Args:
            file_path: .md 文件路径

        Returns:
            Strategy 对象，解析失败时返回 None
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            return self._frontmatter_to_strategy(content)
        except Exception as exc:
            logger.warning("Failed to load strategy from %s: %s", file_path, exc)
            return None

    @staticmethod
    def _frontmatter_to_strategy(content: str) -> Optional[Strategy]:
        """将 Frontmatter 字符串解析为 Strategy 对象。

        Args:
            content: Frontmatter 格式的字符串

        Returns:
            Strategy 对象，解析失败时返回 None
        """
        try:
            post = frontmatter.loads(content)
            md = post.metadata

            # 从 name 字段获取 strategy_id
            strategy_id = md.get("name", "")
            if not strategy_id:
                return None

            # 解析 last_applied
            last_applied_raw = md.get("last_applied")
            last_applied: Optional[datetime] = None
            if last_applied_raw and isinstance(last_applied_raw, str):
                try:
                    last_applied = datetime.fromisoformat(last_applied_raw)
                except (ValueError, TypeError):
                    pass

            # 解析 created_at
            created_at = datetime.now()
            created_raw = md.get("created_at")
            if created_raw and isinstance(created_raw, str):
                try:
                    created_at = datetime.fromisoformat(created_raw)
                except (ValueError, TypeError):
                    pass

            # 解析 source_task_ids
            raw_ids = md.get("source_task_ids", [])
            if isinstance(raw_ids, str):
                source_task_ids = [raw_ids]
            elif isinstance(raw_ids, list):
                source_task_ids = [str(x) for x in raw_ids]
            else:
                source_task_ids = []

            return Strategy(
                strategy_id=strategy_id,
                title=str(md.get("title", "")),
                condition=str(md.get("condition", "")),
                action=str(md.get("action", "")),
                rationale=str(md.get("rationale", "")),
                category=str(md.get("category", "workflow")),  # type: ignore[arg-type]
                source_task_ids=source_task_ids,
                confidence=float(md.get("confidence", 0.5)),
                created_at=created_at,
                applied_count=int(md.get("applied_count", 0)),
                success_count=int(md.get("success_count", 0)),
                last_applied=last_applied,
            )
        except Exception as exc:
            logger.warning("Failed to parse strategy frontmatter: %s", exc)
            return None

    # ── 归档 ──────────────────────────────────────────────────

    def _archive(self, strategy_id: str) -> None:
        """将策略从活跃目录移至 archived/ 子目录。

        Args:
            strategy_id: 策略 ID
        """
        for category in _VALID_CATEGORIES:
            src = self._base_path / category / f"{strategy_id}.md"
            if src.exists():
                dst = self._base_path / "archived" / category / f"{strategy_id}.md"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                logger.debug("Strategy %s archived to %s", strategy_id, dst)
                return

    def _delete_file(self, strategy_id: str) -> None:
        """删除策略文件（在活跃和 archived 目录中查找）。

        Args:
            strategy_id: 策略 ID
        """
        # 查活跃目录
        for category in _VALID_CATEGORIES:
            file_path = self._base_path / category / f"{strategy_id}.md"
            if file_path.exists():
                file_path.unlink()
                return

        # 查 archived 目录
        archived_base = self._base_path / "archived"
        if archived_base.exists():
            for category in _VALID_CATEGORIES:
                file_path = archived_base / category / f"{strategy_id}.md"
                if file_path.exists():
                    file_path.unlink()
                    return

    # ── 关键词检索 ─────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """分词：小写 + 按非字母数字字符分割。

        Args:
            text: 输入文本

        Returns:
            分词后的集合
        """
        return set(re.findall(r"[a-zA-Z0-9一-鿿]+", text.lower()))

    @staticmethod
    def _compute_field_score(query_words: set[str], field_text: str) -> float:
        """计算单个字段的关键词匹配分数。

        Args:
            query_words: 查询分词集合
            field_text: 字段文本

        Returns:
            匹配分数 0.0~1.0
        """
        if not query_words or not field_text:
            return 0.0
        field_lower = field_text.lower()
        hits = sum(1 for qw in query_words if qw in field_lower)
        return hits / len(query_words)

    def _compute_search_score(self, query_words: set[str], strategy: Strategy) -> float:
        """计算策略与查询的综合匹配分数。

        综合分 = (condition_score × 2.0 + action_score × 1.5 + rationale_score × 1.0) / 4.5 × confidence

        Args:
            query_words: 查询分词集合
            strategy: 策略对象

        Returns:
            综合分数
        """
        condition_score = self._compute_field_score(query_words, strategy.condition) * _FIELD_WEIGHTS["condition"]
        action_score = self._compute_field_score(query_words, strategy.action) * _FIELD_WEIGHTS["action"]
        rationale_score = self._compute_field_score(query_words, strategy.rationale) * _FIELD_WEIGHTS["rationale"]

        total_weight = sum(_FIELD_WEIGHTS.values())
        raw_score = (condition_score + action_score + rationale_score) / total_weight

        return raw_score * strategy.confidence

    # ── 辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _ensure_valid_category(category: str) -> str:
        """确保类别有效，无效时回退为 'workflow'。

        Args:
            category: 类别值

        Returns:
            有效的类别值
        """
        return category if category in _VALID_CATEGORIES else "workflow"
