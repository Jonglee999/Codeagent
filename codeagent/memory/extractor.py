"""MemoryExtractor — LLM 驱动的记忆提取。

从对话历史中提取记忆条目，支持去重检测、置信度过滤，
在用户纠正或任务完成时触发提取。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from codeagent.memory.retriever import MemoryRetriever
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType

logger = logging.getLogger(__name__)

# 最大对话轮次
_MAX_CONVERSATION_ROUNDS = 20

# "记住"类关键词语种覆盖
_REMEMBER_KEYWORDS = ["记住", "remember", "记得", "请记住", "please remember"]


@dataclass
class ExtractionCandidate:
    """一条提取候选记忆。"""

    entry: MemoryEntry
    confidence: float  # LLM 对该条记忆的置信度 0.0~1.0
    trigger: str  # "explicit" / "correction" / "task_complete"
    is_duplicate: bool = False  # 是否与现有记忆重复
    duplicate_name: Optional[str] = None  # 重复的现有记忆 name


# ── LLM 提取提示词模板 ──────────────────────────────────────

_EXTRACTION_PROMPT_TEMPLATE = """\
你是一个记忆提取助手。请分析以下对话历史，提取值得长期记住的信息。

对话历史：
{conversation_text}

触发原因：{trigger}

请提取以下类型的记忆（JSON 数组格式）：
- user：用户的技术背景、偏好、工作方式
- feedback：用户对 Agent 行为的纠正规则（"不要做X"、"应该用Y"）
- project：项目架构决策、技术约束
- code_pattern：项目特有的代码操作模式（如"该项目用 X 方式处理 Y"）
- session：本次任务的临时决策（仅当明确有价值时提取）

输出格式（JSON 数组）：
[
  {{
    "type": "feedback",
    "name": "no-mock-database",
    "description": "集成测试必须使用真实数据库",
    "body": "集成测试必须连接真实数据库...\\n\\n**Why:** ...\\n**How to apply:** ...",
    "confidence": 0.95
  }}
]

规则：
1. 只提取有明确证据支持的信息，不要推断
2. feedback 类型必须包含 Why 和 How to apply 说明
3. 没有值得提取的信息时返回空数组 []
4. name 使用 kebab-case，简洁描述内容
"""

_EXPLICIT_PROMPT_TEMPLATE = """\
你是一个记忆提取助手。用户想要记住以下信息。

用户消息：{user_message}

{context_text}
请将其提取为一条记忆条目（JSON 数组格式，如果不需要提取请返回空数组）：

[
  {{
    "type": "user",
    "name": "user-preference",
    "description": "一行摘要",
    "body": "详细内容",
    "confidence": 1.0
  }}
]

规则：
1. 如果消息包含具体信息，提取为一条记忆
2. 消息只是打招呼或无关内容时返回 []
3. name 使用 kebab-case
"""


class MemoryExtractor:
    """LLM 驱动的记忆提取器。

    Args:
        store: MemoryStore 实例
        retriever: MemoryRetriever 实例
        llm_client: LLM 调用函数，签名 await llm_client(model=..., messages=...)
        model_name: LLM 模型名称
        auto_mode: True 直接保存；False 返回候选供确认
        min_confidence: 最低置信度阈值
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        llm_client: Callable[..., Any],
        model_name: str = "deepseek/deepseek-v4-flash",
        auto_mode: bool = False,
        min_confidence: float = 0.7,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._llm_client = llm_client
        self._model_name = model_name
        self._auto_mode = auto_mode
        self._min_confidence = min_confidence

    # ── 公共接口 ──────────────────────────────────────────────

    async def extract_from_conversation(
        self,
        conversation_history: list[dict[str, Any]],
        trigger: str = "task_complete",
    ) -> list[ExtractionCandidate]:
        """从对话历史中提取记忆候选。

        1. 调用 LLM 分析对话历史
        2. 解析 JSON 响应
        3. 去重检查 + 置信度过滤
        4. auto_mode=True 时自动保存

        Args:
            conversation_history: 对话历史 [{"role":..., "content":...}]
            trigger: 触发原因

        Returns:
            所有候选列表（含 is_duplicate 标记）
        """
        prompt = self._build_extraction_prompt(conversation_history, trigger)
        raw_json = await self._call_llm(prompt)
        if raw_json is None:
            return []

        candidates = self._parse_json_response(raw_json, trigger)
        return self._process_candidates(candidates)

    async def extract_explicit(
        self,
        user_message: str,
        context: str = "",
    ) -> list[ExtractionCandidate]:
        """处理用户显式"记住这个"指令。

        Args:
            user_message: 用户消息
            context: 额外上下文

        Returns:
            提取的候选列表
        """
        prompt = _EXPLICIT_PROMPT_TEMPLATE.format(
            user_message=user_message,
            context_text=f"上下文：{context}\n" if context else "",
        )
        raw_json = await self._call_llm(prompt)
        if raw_json is None:
            return []

        candidates = self._parse_json_response(raw_json, "explicit")
        return self._process_candidates(candidates)

    # ── 去重检测 ──────────────────────────────────────────────

    def _check_duplicate(
        self,
        candidate: ExtractionCandidate,
        threshold: float = 0.85,
    ) -> tuple[bool, Optional[str]]:
        """检查候选是否与现有记忆重复。

        检查规则：
        1. name 全局唯一，同名即重复
        2. 同类型内容高度相似（keyword_score >= threshold）视为重复

        Args:
            candidate: 候选条目
            threshold: 相似度阈值（默认 0.85）

        Returns:
            (is_duplicate, duplicate_name)
        """
        candidate_text = (
            f"{candidate.entry.description} {candidate.entry.body}"
        )
        query_words = MemoryRetriever._tokenize(candidate_text)
        if not query_words:
            return False, None

        existing = self._store.list_all()

        for existing_entry in existing:
            # 规则 1：name 全局唯一
            if existing_entry.name == candidate.entry.name:
                return True, existing_entry.name

        # 规则 2：同类型内容相似度
        same_type = [e for e in existing if e.memory_type == candidate.entry.memory_type]
        for existing_entry in same_type:
            # 计算相似度：用候选文本作为 query 去 match 现有条目
            score = self._retriever._keyword_score(
                " ".join(query_words), existing_entry
            )
            if score >= threshold:
                return True, existing_entry.name

        return False, None

    # ── 内部辅助 ──────────────────────────────────────────────

    def _build_extraction_prompt(
        self,
        conversation_history: list[dict[str, Any]],
        trigger: str,
    ) -> str:
        """构建 LLM 提取提示词。

        Args:
            conversation_history: 对话历史（最多保留最近 20 轮）
            trigger: 触发原因

        Returns:
            格式化后的提示词
        """
        # 截断到最近 20 轮
        truncated = conversation_history[-_MAX_CONVERSATION_ROUNDS * 2:]

        lines: list[str] = []
        for msg in truncated:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content[:500]}")

        conversation_text = "\n".join(lines) if lines else "(empty)"
        return _EXTRACTION_PROMPT_TEMPLATE.format(
            conversation_text=conversation_text,
            trigger=trigger,
        )

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """调用 LLM 获取响应文本。"""
        try:
            response = await self._llm_client(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content
        except Exception as exc:
            logger.warning("LLM call failed in MemoryExtractor: %s", exc)
            return None

    def _parse_json_response(
        self,
        raw: str,
        trigger: str,
    ) -> list[ExtractionCandidate]:
        """解析 LLM 返回的 JSON 响应。

        Args:
            raw: LLM 原始响应文本
            trigger: 触发原因

        Returns:
            解析后的候选列表（JSON 解析失败返回空列表）
        """
        try:
            cleaned = self._clean_json(raw)
            data = json.loads(cleaned)
            if not isinstance(data, list):
                logger.warning("LLM response is not a JSON array")
                return []
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM JSON response: %s", exc)
            return []

        candidates: list[ExtractionCandidate] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                memory_type = MemoryType(item.get("type", "session"))
                name = self._to_kebab_case(item.get("name", ""))
                description = item.get("description", "")
                body = item.get("body", "")
                confidence = float(item.get("confidence", 0.5))

                entry = MemoryEntry(
                    name=name,
                    memory_type=memory_type,
                    description=description,
                    body=body,
                    confidence=confidence,
                )

                candidates.append(ExtractionCandidate(
                    entry=entry,
                    confidence=confidence,
                    trigger=trigger,
                ))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping invalid extraction item: %s", exc)

        return candidates

    def _process_candidates(
        self,
        candidates: list[ExtractionCandidate],
    ) -> list[ExtractionCandidate]:
        """处理候选列表：去重检测 + 置信度过滤。

        Args:
            candidates: 原始候选列表

        Returns:
            处理后的候选列表
        """
        for cand in candidates:
            # 置信度过滤
            if cand.confidence < self._min_confidence:
                cand.confidence = 0.0  # 标记为无效
                continue

            # 去重检测
            is_dup, dup_name = self._check_duplicate(cand)
            if is_dup:
                cand.is_duplicate = True
                cand.duplicate_name = dup_name

            # auto_mode 保存
            if self._auto_mode and not is_dup:
                try:
                    self._store.save(cand.entry)
                    logger.debug("Auto-saved memory: %s", cand.entry.name)
                except Exception as exc:
                    logger.warning("Failed to auto-save memory: %s", exc)

        return candidates

    @staticmethod
    def _clean_json(content: str) -> str:
        """清理 LLM 输出，提取 JSON 数组。

        移除 markdown 代码块标记和无关文本。
        """
        content = content.strip()

        # 尝试提取 ```json ... ``` 块
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    return part

        # 尝试直接解析
        if content.startswith("[") or content.startswith("{"):
            return content

        return content

    @staticmethod
    def _to_kebab_case(name: str) -> str:
        """将名称转为 kebab-case。

        规则：空格→连字符，大写转小写，移除非字母数字/连字符字符。
        """
        name = name.lower()
        name = name.replace(" ", "-")
        name = re.sub(r"[^a-zA-Z0-9一-鿿-]", "", name)
        name = re.sub(r"-+", "-", name)
        name = name.strip("-")
        return name if name else "untitled"
