"""StrategyExtractor — LLM 驱动的策略提炼器。

分析任务执行轨迹，使用 LLM 自动提炼可复用的执行策略。
存储为目标策略格式供后续 StrategyStore/StrategyApplier 使用。

触发时机：
- 成功任务：每 5 次触发一次批量分析
- 失败任务：每次经过修复闭环的失败任务立即触发
- 手动触发：用户可以通过 /learn 命令触发
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Literal

logger = logging.getLogger(__name__)

# 每次最多提炼的策略数
_MAX_STRATEGIES_PER_EXTRACTION = 5

# 策略摘要中每条轨迹的最大行数
_MAX_SUMMARY_LINES_PER_TRAJECTORY = 5


@dataclass
class Strategy:
    """一条可复用的执行策略。

    Attributes:
        strategy_id: 唯一 ID（UUID）
        title: 简短标题（≤50 字符）
        condition: 触发条件（什么情况下适用）
        action: 建议动作
        rationale: 理由（从轨迹中归纳）
        category: 策略类别
        source_task_ids: 来源轨迹 ID 列表
        confidence: 初始置信度（0.0~1.0）
        created_at: 创建时间
        applied_count: 被应用的次数
        success_count: 应用成功的次数
        last_applied: 最后应用时间
    """

    strategy_id: str
    title: str
    condition: str
    action: str
    rationale: str
    category: Literal["workflow", "coding_style", "error_avoidance", "tool_usage"]
    source_task_ids: list[str]
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    applied_count: int = 0
    success_count: int = 0
    last_applied: Optional[datetime] = None


# ── LLM 提取提示词模板 ──────────────────────────────────────

_EXTRACTION_PROMPT_TEMPLATE = """\
你是一个代码 Agent 的策略分析师。分析以下 {n} 条任务执行轨迹，提炼可复用的执行策略。

关注以下模式：
1. 成功任务中反复出现的操作序列
2. 失败后修复成功的关键步骤
3. 特定类型任务（如修改 API、添加测试）的最佳实践

轨迹数据：
{trajectories_summary}

输出 JSON 格式（最多 5 条策略）：
{{"strategies": [
  {{
    "title": "添加 FastAPI 路由时同步更新测试",
    "condition": "当需要在 FastAPI 项目中添加新路由时",
    "action": "应同时在 tests/ 下添加对应的测试文件，覆盖正常和异常路径",
    "rationale": "在 3 次任务中，未同步写测试导致 CI 失败 2 次，同步写测试的任务全部通过",
    "category": "workflow"
  }}
]}}

规则：
1. 只提取有明确证据支持的模式，不要凭空推断
2. 每次最多输出 5 条策略
3. 没有有价值的策略时返回 {{"strategies": []}}
4. category 必须是以下之一：workflow, coding_style, error_avoidance, tool_usage
5. title 不超过 50 字，condition/action/rationale 各不超过 200 字
"""


class StrategyExtractor:
    """策略提炼器——分析任务轨迹，使用 LLM 自动提炼可复用的执行策略。

    触发时机：
    - 成功任务：每 5 次触发一次批量分析
    - 失败任务：每次经过修复闭环的失败任务立即触发
    - 手动触发：用户可以通过 /learn 命令触发

    Args:
        llm_client: LLM 调用函数，签名 await llm_client(model=..., messages=...)
        recorder: TrajectoryRecorder 实例，用于加载轨迹
        store: StrategyStore 实例，用于去重检查
        min_confidence: 最低置信度阈值（默认 0.5）
    """

    def __init__(
        self,
        llm_client: Callable[..., Any],
        recorder: Any,
        store: Any,
        min_confidence: float = 0.5,
        use_semantic_dedup: bool = True,
    ) -> None:
        self._llm_client = llm_client
        self._recorder = recorder
        self._store = store
        self._min_confidence = min_confidence
        self._use_semantic_dedup = use_semantic_dedup
        self._vector_model: Any = None

    # ── 公开接口 ──────────────────────────────────────────────

    async def should_extract(
        self,
        task_count: int,
        last_had_repair: bool,
    ) -> bool:
        """判断是否应触发策略提炼。

        - 每 5 次成功任务 → True
        - 任意 1 次经过修复的任务 → True
        - 否则 → False

        Args:
            task_count: 累计成功任务数
            last_had_repair: 当前/最近任务是否经过了修复

        Returns:
            是否应触发提炼
        """
        if last_had_repair:
            return True
        if task_count > 0 and task_count % 5 == 0:
            return True
        return False

    async def extract(
        self,
        trajectories: list[Any],
    ) -> list[Strategy]:
        """分析轨迹列表，提炼可复用的策略。

        1. 构建 LLM 提示词，传入轨迹摘要
        2. 调用 LLM 获取 JSON 格式的策略列表
        3. 对每条策略去重检查
        4. 返回非重复的策略列表

        Args:
            trajectories: 轨迹列表（Trajectory 对象列表）

        Returns:
            提炼出的策略列表（最多 5 条）
        """
        if not trajectories:
            return []

        prompt = self._build_extraction_prompt(trajectories)
        raw_json = await self._call_llm(prompt)
        if raw_json is None:
            return []

        parsed = self._parse_json_response(raw_json)
        if not parsed:
            return []

        strategies = self._process_strategies(parsed, trajectories)
        return strategies[: _MAX_STRATEGIES_PER_EXTRACTION]

    async def extract_from_recent(self) -> list[Strategy]:
        """从最近轨迹中自动提取策略。

        调用 recorder.load_recent() 获取最近轨迹，然后调用 extract()。

        Returns:
            提炼出的策略列表
        """
        try:
            trajectories = self._recorder.load_recent(n=20)
        except Exception as exc:
            logger.warning("Failed to load recent trajectories: %s", exc)
            return []
        return await self.extract(trajectories)

    # ── 去重检测（语义 + 关键词） ─────────────────────────

    def _check_duplicate(
        self,
        strategy: Strategy,
        existing: list[Strategy],
        threshold: float = 0.85,
    ) -> tuple[bool, Optional[str]]:
        """检查策略是否与现有策略重复。

        use_semantic_dedup=True 时优先使用向量余弦相似度；
        False 或向量模型不可用时回退到关键词匹配。
        """
        if not existing:
            return False, None

        # 标题完全一致始终视为重复（优先于语义/关键词匹配）
        for ex in existing:
            if ex.title.strip().lower() == strategy.title.strip().lower():
                return True, ex.strategy_id

        query_text = f"{strategy.condition} {strategy.action}"

        if self._use_semantic_dedup:
            model = self._get_vector_model()
            if model is not None:
                return self._semantic_dedup(query_text, existing, model, threshold)

        return self._keyword_dedup(query_text, existing, strategy, threshold)

    def _semantic_dedup(
        self,
        query_text: str,
        existing: list[Strategy],
        model: Any,
        threshold: float,
    ) -> tuple[bool, Optional[str]]:
        """使用向量余弦相似度进行语义去重。"""
        try:
            import numpy as np
            query_emb = model.encode([query_text], normalize_embeddings=True)
            for ex in existing:
                ex_text = f"{ex.condition} {ex.action}"
                ex_emb = model.encode([ex_text], normalize_embeddings=True)
                score = float(np.dot(query_emb[0], ex_emb[0]))
                if score >= threshold:
                    return True, ex.strategy_id
        except Exception as exc:
            logger.warning("Semantic dedup failed, falling back to keyword: %s", exc)
            return self._keyword_dedup(query_text, existing, None, threshold)
        return False, None

    def _get_vector_model(self):
        """懒加载向量模型，失败时返回 None。"""
        if self._vector_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._vector_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("Failed to load vector model for dedup: %s", exc)
                self._vector_model = False
        return self._vector_model if self._vector_model is not False else None

    def _keyword_dedup(
        self,
        query_text: str,
        existing: list[Strategy],
        strategy: Strategy | None,
        threshold: float,
    ) -> tuple[bool, Optional[str]]:
        """使用关键词匹配进行去重（降级方案）。"""
        query_words = self._tokenize(query_text)
        if not query_words:
            return False, None

        for existing_strategy in existing:
            # 规则 1：title 完全一致视为重复
            if strategy and existing_strategy.title.strip().lower() == strategy.title.strip().lower():
                return True, existing_strategy.strategy_id

            # 规则 2：condition + action 内容高度相似
            existing_text = f"{existing_strategy.condition} {existing_strategy.action}"
            score = self._keyword_score(query_words, existing_text)
            if score >= threshold:
                return True, existing_strategy.strategy_id

            # 规则 3：同 category + 高 action 相似度
            if strategy and existing_strategy.category == strategy.category:
                action_score = self._keyword_score(
                    self._tokenize(strategy.action),
                    existing_strategy.action,
                )
                if action_score >= threshold:
                    return True, existing_strategy.strategy_id

        return False, None

    # ── 内部辅助 ──────────────────────────────────────────────

    def _build_extraction_prompt(
        self,
        trajectories: list[Any],
    ) -> str:
        """构建 LLM 提取提示词。

        Args:
            trajectories: 轨迹列表

        Returns:
            格式化后的提示词
        """
        summary = self._trajectories_to_summary(trajectories)
        return _EXTRACTION_PROMPT_TEMPLATE.format(
            n=len(trajectories),
            trajectories_summary=summary,
        )

    def _trajectories_to_summary(self, trajectories: list[Any]) -> str:
        """将轨迹列表格式化为摘要文本。

        每条轨迹 3-5 行摘要，包含 task_id、user_request、步骤数、成功/失败状态。

        Args:
            trajectories: 轨迹列表

        Returns:
            格式化后的摘要文本
        """
        lines: list[str] = []
        for i, traj in enumerate(trajectories, 1):
            try:
                task_id = getattr(traj, "task_id", "unknown")[:8]
                user_req = getattr(traj, "user_request", "unknown")
                steps_count = len(getattr(traj, "steps", []))
                final_status = getattr(traj, "final_status", "unknown")
                repair_rounds = getattr(traj, "repair_rounds", 0)
                validation_passed = getattr(traj, "validation_passed", False)

                # 提取工具调用数
                tool_call_count = 0
                if hasattr(traj, "steps"):
                    for step in traj.steps:
                        tool_calls = getattr(step, "tool_calls", None) or []
                        tool_call_count += len(tool_calls)

                lines.append(f"--- Trajectory {i} ---")
                lines.append(f"Task: {task_id} | Request: {user_req}")
                lines.append(f"Steps: {steps_count} | Tool calls: {tool_call_count}")
                lines.append(f"Status: {final_status} | Repairs: {repair_rounds} | Validated: {validation_passed}")
                lines.append("")
            except Exception:
                lines.append(f"--- Trajectory {i} --- (failed to summarize)")
                lines.append("")

        return "\n".join(lines).strip()

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM 获取响应文本。

        Args:
            prompt: 提示词

        Returns:
            LLM 响应文本，失败时返回 None
        """
        try:
            routing = (
                {"model_role": "summary"}
                if hasattr(self._llm_client, "registry")
                and hasattr(self._llm_client, "select")
                else {}
            )
            response = await self._llm_client(
                model="deepseek/deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                **routing,
            )
            content = response.choices[0].message.content
            return content
        except Exception as exc:
            logger.warning("LLM call failed in StrategyExtractor: %s", exc)
            return None

    def _parse_json_response(self, raw: str) -> list[dict]:
        """解析 LLM 返回的 JSON 响应。

        Args:
            raw: LLM 原始响应文本

        Returns:
            解析后的策略字典列表（JSON 解析失败返回空列表）
        """
        try:
            cleaned = self._clean_json(raw)
            data = json.loads(cleaned)

            # 支持两种格式：{"strategies": [...]} 或直接数组
            if isinstance(data, dict):
                strategies_data = data.get("strategies", [])
            elif isinstance(data, list):
                strategies_data = data
            else:
                logger.warning("LLM response is neither object nor array")
                return []

            if not isinstance(strategies_data, list):
                logger.warning("LLM response 'strategies' field is not a list")
                return []

            return strategies_data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM JSON response: %s", exc)
            return []

    def _process_strategies(
        self,
        parsed: list[dict],
        trajectories: list[Any],
    ) -> list[Strategy]:
        """处理解析后的策略：去重检测 + 置信度过滤 + 构建 Strategy 对象。

        Args:
            parsed: 解析后的策略字典列表
            trajectories: 原始轨迹列表（用于提取 source_task_ids）

        Returns:
            处理后的 Strategy 列表
        """
        source_ids = [
            getattr(t, "task_id", "unknown")[:8] for t in trajectories
            if hasattr(t, "task_id")
        ]

        # 加载现有策略用于去重
        existing: list[Strategy] = []
        try:
            existing = self._store.list_active()
        except Exception as exc:
            logger.warning("Failed to load existing strategies for dedup: %s", exc)

        strategies: list[Strategy] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            try:
                strategy = self._dict_to_strategy(item, source_ids)

                # 置信度过滤
                if strategy.confidence < self._min_confidence:
                    continue

                # 去重检测
                is_dup, _ = self._check_duplicate(strategy, existing)
                if is_dup:
                    continue

                strategies.append(strategy)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping invalid strategy item: %s", exc)

        return strategies

    def _dict_to_strategy(
        self,
        item: dict,
        source_task_ids: list[str],
    ) -> Strategy:
        """将字典转换为 Strategy 对象。

        Args:
            item: 字典数据
            source_task_ids: 来源任务 ID 列表

        Returns:
            Strategy 对象
        """
        return Strategy(
            strategy_id=str(uuid.uuid4()),
            title=str(item.get("title", ""))[:50],
            condition=str(item.get("condition", ""))[:200],
            action=str(item.get("action", ""))[:200],
            rationale=str(item.get("rationale", ""))[:200],
            category=self._validate_category(item.get("category", "workflow")),
            source_task_ids=source_task_ids,
            confidence=float(item.get("confidence", 0.5)),
        )

    @staticmethod
    def _validate_category(category: str) -> Literal["workflow", "coding_style", "error_avoidance", "tool_usage"]:
        """验证并规范化策略类别。

        Args:
            category: 原始类别值

        Returns:
            有效的类别值（无效时默认 'workflow'）
        """
        valid = {"workflow", "coding_style", "error_avoidance", "tool_usage"}
        return category if category in valid else "workflow"  # type: ignore[return-value]

    # ── 关键词匹配 ────────────────────────────────────────────

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
    def _keyword_score(query_words: set[str], target_text: str) -> float:
        """计算关键词在目标文本中的匹配分数（0.0~1.0）。

        用 query 中的词在 target_text 中的覆盖率作为分数。

        Args:
            query_words: 查询分词集合
            target_text: 目标文本

        Returns:
            匹配分数 0.0~1.0
        """
        if not query_words or not target_text:
            return 0.0

        target_lower = target_text.lower()
        hits = sum(1 for qw in query_words if qw in target_lower)
        return hits / len(query_words)

    # ── JSON 清理 ─────────────────────────────────────────────

    @staticmethod
    def _clean_json(content: str) -> str:
        """清理 LLM 输出，提取 JSON。

        移除 markdown 代码块标记和无关文本。

        Args:
            content: 原始 LLM 输出

        Returns:
            清理后的 JSON 字符串
        """
        content = content.strip()

        # 尝试提取 ```json ... ``` 块
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    return part

        # 尝试直接解析
        if content.startswith("{") or content.startswith("["):
            return content

        return content
