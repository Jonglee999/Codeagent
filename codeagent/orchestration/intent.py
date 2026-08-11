"""Deterministic, shared response-mode classification."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

ResponseMode = Literal["chat", "execute"]

_ZH_ACTION_WORDS = (
    "修复", "创建", "新建", "新增", "修改", "删除", "运行", "测试", "重构",
    "实现", "优化", "安装", "配置", "回滚", "提交", "生成", "写", "编写",
    "制作", "开发", "搭建",
)
_EN_ACTION = re.compile(
    r"\b(fix|add|create|change|edit|delete|remove|run|test|refactor|implement|"
    r"optimize|install|configure|rollback|commit|generate|write|update|build)\b",
    re.I,
)
_ZH_QUESTION = ("？", "?", "哪些", "什么", "为什么", "怎么", "如何", "是否", "吗")
_REQUEST_CUE = re.compile(
    r"请|帮我|请帮|麻烦|能否|可否|可以.{0,4}吗|给我|替我|继续|再(?:次|运行|试|修)|务必|"
    r"\b(please|can you|could you|would you|go ahead|try again|rerun)\b",
    re.I,
)
_EN_NEGATED_ACTION = re.compile(
    r"\b(do not|don't|dont|no need to|without)\s+"
    r"(fix|add|create|change|edit|delete|remove|run|test|write|update|build)\b",
    re.I,
)
_ZH_ACTION_NOUN = re.compile(r"开发(?:者|人员|工程师)")


@dataclass(frozen=True)
class IntentDecision:
    mode: ResponseMode
    reason: str
    action_detected: bool
    question_detected: bool


def classify_intent(query: str, requested_mode: str = "auto") -> IntentDecision:
    """Classify a turn without an extra model call.

    Explicit modes win. For auto mode, negated actions are removed before
    detection, polite/indirect action requests execute, and explanatory
    questions remain tool-free chat.
    """
    if requested_mode in {"chat", "execute"}:
        return IntentDecision(
            mode=requested_mode,  # type: ignore[arg-type]
            reason="explicit response mode",
            action_detected=requested_mode == "execute",
            question_detected=False,
        )

    cleaned = query
    for prefix in ("不要", "不用", "无需", "不需要"):
        for word in _ZH_ACTION_WORDS:
            cleaned = cleaned.replace(f"{prefix}{word}", "")
    cleaned = _EN_NEGATED_ACTION.sub("", cleaned)

    action_source = _ZH_ACTION_NOUN.sub("", cleaned)
    action = any(word in action_source for word in _ZH_ACTION_WORDS) or bool(
        _EN_ACTION.search(action_source)
    )
    question = any(marker in query for marker in _ZH_QUESTION) or bool(
        re.search(r"\b(why|what|which|how|is|are|does|did)\b", query, re.I)
    )
    request_cue = bool(_REQUEST_CUE.search(query))

    if action and (request_cue or not question):
        return IntentDecision("execute", "action request", True, question)
    if question:
        return IntentDecision("chat", "informational question", action, True)
    if action:
        return IntentDecision("execute", "imperative action", True, False)
    return IntentDecision("chat", "conversation without action intent", False, False)


def classify_response_mode(query: str, requested_mode: str = "auto") -> ResponseMode:
    return classify_intent(query, requested_mode).mode
