"""Adaptive run policy for foreground coding-agent work.

The policy is deliberately deterministic and cheap.  It decides which pieces of
the orchestration graph are useful for the current request instead of making
every task pay for planning, broad context retrieval, memory, and learning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Literal


WorkflowMode = Literal["direct", "planned"]
ContextMode = Literal["minimal", "full"]
MemoryMode = Literal["off", "relevant"]
LearningMode = Literal["off", "capture"]


@dataclass(frozen=True)
class RunProfile:
    workflow: WorkflowMode
    context: ContextMode
    memory: MemoryMode
    learning: LearningMode
    benchmark: bool = False
    reflection: Literal["on_failure"] = "on_failure"
    reason: str = "focused task"

    def public_metadata(self) -> dict[str, Any]:
        return asdict(self)


_COMPLEX_PATTERNS = (
    r"架构|迁移|重构|全局|跨模块|端到端|完整功能|系统性|安全审计|性能优化",
    r"\b(architect|migration|migrate|refactor|cross[- ]module|end[- ]to[- ]end|"
    r"security audit|performance optimization|whole codebase)\b",
)
_DURABLE_MEMORY_PATTERNS = (
    r"记住|以后都|始终|永远不要|偏好|团队约定|项目规范",
    r"\b(remember|from now on|always|never|preference|team convention)\b",
)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def requests_durable_memory(query: str) -> bool:
    """Whether the user explicitly asks to persist a durable preference/rule."""
    return _matches(_DURABLE_MEMORY_PATTERNS, query.strip())


def select_run_profile(
    query: str,
    *,
    conversation_history: list[dict[str, Any]] | None = None,
    direct_execution: bool = False,
    benchmark_instance_id: str | None = None,
) -> RunProfile:
    """Choose the lightest workflow that still fits the request.

    An explicit follow-up/direct hint always wins. Benchmark examples keep the
    planned path because their issue statements are intentionally broad. Other
    requests use a small complexity score; this avoids another classifier LLM
    call and keeps the decision inspectable.
    """

    text = query.strip()
    durable_memory = requests_durable_memory(text)

    if direct_execution:
        return RunProfile(
            workflow="direct",
            context="minimal",
            # The turn history already supplies short-term conversational
            # context. Durable/vector recall is reserved for an explicit memory
            # request, avoiding an unrelated retrieval on every follow-up.
            memory="relevant" if durable_memory else "off",
            learning="capture" if durable_memory else "off",
            reason="follow-up continues in the existing conversation context",
        )

    if benchmark_instance_id:
        return RunProfile(
            # SWE issues cannot be planned reliably before repository inspection.
            # Use the single evidence-preserving tool loop and avoid spending a
            # provider call on brittle JSON plan generation.
            workflow="direct",
            context="minimal",
            memory="off",
            learning="off",
            benchmark=True,
            reason="benchmark uses a deterministic controller and just-in-time repository tools",
        )

    complexity = 0
    if len(text) >= 600:
        complexity += 2
    elif len(text) >= 320:
        complexity += 1
    if text.count("\n") >= 5:
        complexity += 1
    if _matches(_COMPLEX_PATTERNS, text):
        complexity += 2
    if len(re.findall(r"(?:并且|同时|然后|以及|and then|as well as)", text, re.I)) >= 2:
        complexity += 1

    if complexity >= 2:
        return RunProfile(
            workflow="planned",
            context="full",
            memory="relevant",
            learning="capture" if durable_memory else "off",
            reason="request spans multiple concerns or needs broad repository context",
        )

    return RunProfile(
        workflow="direct",
        context="minimal",
        memory="relevant" if durable_memory else "off",
        learning="capture" if durable_memory else "off",
        reason="focused request can be solved through just-in-time tool use",
    )
