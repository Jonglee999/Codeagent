"""ExecutionNode — 执行节点（Phase 3.5 升级版：偏离检测 + 目标追踪）。

核心功能：
1. Plan-aware: 按 PlanStep 逐步执行（state.plan 存在时）
2. Phase 1a 兼容: 直连 LLM 调用模式仍支持
3. 每次 write_file 后立即调用语法检查
4. 语法错误 → 反馈 LLM 重新生成修复代码（最多 3 次）
5. 单步骤超过 16 次 tool_calls → 强制结束
6. 进度上报：通过 callback 实时报告执行进度
7. TaskFocus: 偏离检测 + 目标追踪 + System Prompt 注入
"""

from __future__ import annotations

import inspect
import html
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from codeagent import config as codeagent_config
from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.orchestration.budget import preflight_model_call
from codeagent.orchestration.state import AgentState, PlanStep, RepairContext, StructuredError
from codeagent.tracing import trace_node

logger = logging.getLogger(__name__)

_MAX_TOOL_OBSERVATION_CHARS = 12_000
_MAX_PRESERVED_REASONING_CHARS = 8_000
_MAX_IDENTICAL_FAILED_TOOL_CALLS = 2
_MAX_UNEXPOSED_TOOL_CALLS = 4
_MAX_MUTATION_CONTEXT_READS = 2
_MISSING_VALIDATION_DEPENDENCY_PATTERNS = (
    "modulenotfounderror: no module named",
    "command not found",
    "is not recognized as an internal or external command",
    "could not find a version that satisfies the requirement",
)
_TEST_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:"
    r"pytest|py\.test|tox|nox|"
    r"python(?:3(?:\.\d+)?)?\s+-m\s+(?:pytest|unittest)|"
    r"python(?:3(?:\.\d+)?)?\s+[^;&|]*runtests\.py|"
    r"manage\.py\s+test|npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|"
    r"yarn\s+test|go\s+test|cargo\s+test|mvn(?:w)?\s+test|gradle(?:w)?\s+test"
    r")(?:\s|$)",
    re.IGNORECASE,
)
_SOURCE_PATH_PATTERN = re.compile(
    r"(?<![\w./-])([\w.-]+(?:/[\w.-]+)+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|rb|php))"
)


def _is_validation_dependency_failure(
    *,
    step: PlanStep,
    tool_name: str,
    tool_result: Any,
) -> bool:
    """Recognize validation that ran but was blocked by the local environment."""

    if (
        step.action != "command"
        or tool_name != "run_terminal"
        or tool_result.success
        or tool_result.error_code != "NON_ZERO_EXIT"
    ):
        return False
    data = tool_result.data if isinstance(tool_result.data, dict) else {}
    diagnostic = "\n".join(
        str(value)
        for value in (
            tool_result.error_message,
            data.get("stdout"),
            data.get("stderr"),
        )
        if value
    ).lower()
    return any(pattern in diagnostic for pattern in _MISSING_VALIDATION_DEPENDENCY_PATTERNS)


def _is_test_command(command: object) -> bool:
    """Return whether a terminal command invokes a test runner, not mere discovery."""
    return isinstance(command, str) and bool(_TEST_COMMAND_PATTERN.search(command))


def _mentioned_source_paths(request: str) -> set[str]:
    return {
        match.replace("\\", "/").lower()
        for match in _SOURCE_PATH_PATTERN.findall(request)
    }


def _record_failed_tool_call(
    counts: dict[str, int],
    *,
    scope: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: Any,
) -> tuple[str, int] | None:
    """Count a normalized failed tool request without retaining its arguments."""

    if tool_result.success:
        return None
    error_identity = tool_result.error_code or tool_result.error_message or "UNKNOWN"
    canonical = json.dumps(
        {
            "scope": scope,
            "tool": tool_name,
            "arguments": tool_args,
            "error": error_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    count = counts.get(signature, 0) + 1
    counts[signature] = count
    return signature, count


def _recover_dsml_tool_calls(message: Any) -> list[Any]:
    """Recover provider-emitted DSML tool calls returned as plain text.

    Some OpenAI-compatible providers occasionally serialize a valid tool request in
    ``message.content`` while leaving ``message.tool_calls`` empty. Treat only the
    explicit DSML envelope as a tool request; ordinary prose and fenced examples are
    left untouched.
    """
    if getattr(message, "tool_calls", None):
        return list(message.tool_calls)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        content = getattr(message, "reasoning_content", None)
    if not isinstance(content, str) or not content.strip():
        return []
    normalized = content
    for marker in ("｜｜DSML｜｜", "｜DSML｜", "|DSML|"):
        normalized = normalized.replace(marker, "")
    if not re.fullmatch(r"\s*<tool_calls>.*</tool_calls>\s*", normalized, re.DOTALL):
        return []
    recovered: list[Any] = []
    for index, match in enumerate(
        re.finditer(
            r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>',
            normalized,
            re.DOTALL,
        )
    ):
        name, body = match.groups()
        arguments = {
            parameter_name: html.unescape(value.strip())
            for parameter_name, value in re.findall(
                r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>',
                body,
                re.DOTALL,
            )
        }
        if not name or not arguments:
            continue
        recovered.append(SimpleNamespace(
            id=f"recovered_dsml_{index}",
            type="function",
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        ))
    return recovered


def _restore_provider_tool_calls(message: Any) -> None:
    recovered = _recover_dsml_tool_calls(message)
    if recovered and not getattr(message, "tool_calls", None):
        message.tool_calls = recovered


def _provider_response_diagnostic(
    message: Any,
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Return metadata-only evidence without persisting private model reasoning."""
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    return {
        "type": "provider_response_diagnostic",
        "content_present": bool(content),
        "content_chars": len(content) if isinstance(content, str) else 0,
        "reasoning_present": bool(reasoning),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "finish_reason": finish_reason,
        "timestamp": time.time(),
    }


def _assistant_retry_message(message: Any, fallback: str) -> dict[str, Any]:
    """Preserve provider thinking state when asking for a missing action.

    Thinking-mode providers can return only ``reasoning_content`` when an output
    limit is reached. Dropping it makes the next turn repeat the same analysis
    from scratch and can exhaust the task budget without ever taking action.
    """
    retry_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.content if message.content is not None else None,
    }
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        retry_message["reasoning_content"] = str(reasoning)[
            -_MAX_PRESERVED_REASONING_CHARS:
        ]
    elif not message.content:
        retry_message["content"] = fallback
    return retry_message


def _tool_result_message(tool_result: Any) -> str:
    """Serialize a bounded observation while retaining recovery and tail output."""
    payload = (
        tool_result.data
        if tool_result.success
        else {
            "error": tool_result.error_message,
            "error_code": tool_result.error_code,
            "retryable": bool(tool_result.retryable),
            "suggested_recovery": tool_result.suggested_recovery,
            "bounded_output": True,
            "details": tool_result.data,
        }
    )
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) <= _MAX_TOOL_OBSERVATION_CHARS:
        return encoded

    marker = (
        "\n...[TOOL_OBSERVATION_TRUNCATED; "
        f"original_chars={len(encoded)}]...\n"
    )
    available = _MAX_TOOL_OBSERVATION_CHARS - len(marker)
    head_size = available * 2 // 3
    return encoded[:head_size] + marker + encoded[-(available - head_size):]

# 最大连续 tool_calls 次数
_MAX_TOOL_CALLS = 16

# 单步最大重试次数
_MAX_RETRIES = 3

# 修复模式常量
_MAX_REPAIR_TOOL_CALLS = 5  # 修复模式最大 tool_calls 次数
_MAX_VALIDATION_ROUNDS = 3  # 最大验证重试轮次

# 修复模式提示词模板
_REPAIR_MODE_PROMPT = """\
## 验证修复模式

上一次修改后验证失败，需要你修复以下错误。

### 验证报告
{validation_report}

### 修复建议
{fix_suggestions_text}

请分析以上错误，读取相关文件，修复代码中的问题。
注意：
1. 只修复验证报告中指出的问题
2. 不要修改与验证失败无关的文件
3. 修复后代码必须通过所有验证
4. 尽量在 {max_repair_tool_calls} 次工具调用内完成修复
"""

# 每个 action 类型的合理工具列表
_ALLOWED_TOOLS_BY_ACTION: dict[str, set[str]] = {
    "create": {"write_file", "read_file", "list_files", "search_code", "get_diagnostics"},
    "modify": {"read_file", "list_files", "write_file", "apply_patch", "search_code", "get_diagnostics"},
    "delete": {"read_file", "list_files", "search_code", "delete_file"},
    "read": {"read_file", "list_files", "search_code", "get_diagnostics"},
    "command": {"run_terminal", "read_file", "list_files", "search_code", "get_diagnostics"},
}

_REQUIRED_EVIDENCE_TOOLS_BY_ACTION: dict[str, set[str]] = {
    "create": {"write_file"},
    "modify": {"write_file", "apply_patch"},
    "delete": {"delete_file"},
    "read": {"read_file", "list_files", "search_code", "get_diagnostics"},
    "command": {"run_terminal"},
}

_MAX_NO_TOOL_REPROMPTS = 1
_MAX_BENCHMARK_COMPLETION_REPROMPTS = 2
_MAX_MUTATION_FORCE_REPROMPTS = 2
_MAX_DISCOVERY_CALLS_BEFORE_MUTATION = 3
_MAX_BENCHMARK_DISCOVERY_WITH_EXPLICIT_TARGET = 4
_MAX_BENCHMARK_DISCOVERY_WITHOUT_TARGET = 6

# 偏离检测阈值
_DEVIATION_WARN_LIMIT = 1  # 首次偏离 → 警告
_DEVIATION_UPGRADE_LIMIT = 2  # 连续 2 次 → 升级警告
_DEVIATION_HUMAN_LIMIT = 3  # 连续 3 次 → 请求人工审核


class ExecutionNode:
    """执行节点 — 驱动 LLM 执行工具调用循环。

    升级版支持两种模式：
    - Plan-aware 模式：按 PlanStep 列表逐步执行
    - 直连模式（Phase 1a 兼容）：直接通过 LLM tool calling 循环
    """

    def __init__(
        self,
        llm: Callable[..., Any],
        tool_gateway: IToolGateway,
        validation_gateway: IValidationGateway,
        model_name: str = "deepseek/deepseek-v4-flash",
        max_tool_calls: int = _MAX_TOOL_CALLS,
        max_retries: int = _MAX_RETRIES,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        memory_gateway: Optional[IMemoryGateway] = None,
        steering_provider: Callable[[], Any] | None = None,
    ) -> None:
        """初始化 ExecutionNode。

        Args:
            llm: LLM 调用函数（如 litellm.completion）
            tool_gateway: 工具系统 Gateway
            validation_gateway: 验证 Gateway
            model_name: 模型名称
            max_tool_calls: 单次执行最大 tool_calls 次数
            max_retries: 单步语法错误最大重试次数
            progress_callback: 进度回调函数
            memory_gateway: 记忆系统 Gateway，None 时跳过记忆检索
        """
        self._llm = llm
        self._tool_gateway = tool_gateway
        self._validation_gateway = validation_gateway
        self._model_name = model_name
        self._max_tool_calls = max_tool_calls
        self._max_retries = max_retries
        self._progress_callback = progress_callback
        self._memory_gateway = memory_gateway
        self._steering_provider = steering_provider

    @trace_node("execution")
    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """执行主入口。

        新增 Phase 4.A.5 修复模式：当 state.validation_results 有失败
        且无待执行计划步骤时，进入验证反馈驱动的自动修复循环。

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新的状态字段（execution_log, errors, plan 等）
        """
        # Phase 4.A.5: 优先检查修复模式
        if self._needs_repair(state):
            result = await self._execute_repair_mode(state)
        elif state.plan and state.current_step_index < len(state.plan):
            result = await self._execute_with_plan(state)
        else:
            result = await self._execute_direct(state)

        return result

    # ── Plan-aware 执行 ──────────────────────────────────────

    async def _execute_with_plan(self, state: AgentState) -> dict[str, Any]:
        """按计划逐步执行（含 TaskFocus 进度追踪）。

        遍历 state.plan[state.current_step_index:]，对每个步骤：
        1. 构建步骤提示词 → 调用 LLM
        2. 执行工具调用循环（含偏离检测）
        3. 语法检查
        4. 记录变更
        5. 上报进度
        6. 更新完成/剩余步骤摘要
        """
        start_time = time.monotonic()
        # Validation/reflection can route back into planned execution. Preserve
        # prior evidence so final benchmark metrics include earlier mutations.
        execution_log: list[dict] = list(state.execution_log)
        errors: list[str] = list(state.errors)
        accumulated_changes: list[dict] = list(state.accumulated_changes)
        current_step_index = state.current_step_index
        deviation_count = state.deviation_count
        deviation_detected = state.deviation_detected
        human_review_required = state.human_review_required
        review_request = state.review_request
        trajectory_steps: list[dict] = list(state.trajectory_steps)
        validation_state = state.validation_state
        warnings = list(state.warnings)

        memory_section = await self._assemble_memory_section(state)
        remaining_steps = state.plan[current_step_index:]

        # 构建完成/剩余步骤摘要
        completed_descriptions = [s.description for s in state.plan[:current_step_index]]
        completed_steps_summary = (
            "; ".join(completed_descriptions) if completed_descriptions else ""
        )

        # 上报 node_start 事件
        for i, step in enumerate(remaining_steps):
            # 如果偏离触发 Human Review，停止执行
            if human_review_required:
                break

            self._report_progress(
                {
                    "type": "node_start",
                    "node": "execution",
                    "step_id": step.step_id,
                    "status": "running",
                    "summary": step.description,
                }
            )

            # 计算剩余步骤描述（不含当前步骤）
            remaining_descriptions = [
                f"[{s.step_id}] {s.description}" for s in remaining_steps[i + 1 :]
            ]

            # 按步骤 action 裁剪工具面：schema 只携带该步骤允许的工具 + MCP
            step_tools = self._available_tools(state, step.action)

            # 构建步骤提示词
            step_prompt = self._build_step_prompt(
                state, step, step_tools, memory_section=memory_section
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": step_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Execute only plan step {step.step_id} now. "
                        f"For action '{step.action}', successful completion requires "
                        f"calling one of: {', '.join(sorted(_REQUIRED_EVIDENCE_TOOLS_BY_ACTION.get(step.action, set()))) or 'an appropriate read tool'}. "
                        f"Use the exact target path {step.target_file!r} when a target is specified."
                    ),
                },
            ]

            # 执行该步骤的工具调用循环（含偏离检测）
            step_log_start = len(execution_log)
            (
                step_result,
                step_deviation_count,
                step_human_review,
                step_review_request,
                step_trajectory_steps,
                step_evidence,
            ) = await self._execute_step_tool_loop(
                state=state,
                step=step,
                messages=messages,
                tool_definitions=step_tools,
                execution_log=execution_log,
                errors=errors,
                accumulated_changes=accumulated_changes,
                deviation_count=deviation_count,
                remaining_descriptions=remaining_descriptions,
            )

            if step_result is not None:
                accumulated_changes = step_result

            trajectory_steps.extend(step_trajectory_steps)

            step.evidence = [
                {
                    "tool_name": entry.get("tool_name"),
                    "success": bool(entry.get("success")),
                    "validation_degraded": bool(
                        entry.get("validation_degraded")
                    ),
                    "duration_ms": entry.get("duration_ms", 0),
                    "file_path": entry.get("arguments", {}).get("file_path"),
                }
                for entry in execution_log[step_log_start:]
                if entry.get("type") == "tool_call"
                and (
                    entry.get("success") is True
                    or entry.get("validation_degraded") is True
                )
            ]

            degraded_entries = [
                entry
                for entry in execution_log[step_log_start:]
                if entry.get("validation_degraded") is True
            ]
            if degraded_entries:
                validation_state = "validation_degraded"
                warning = (
                    f"Validation for step {step.step_id} was blocked by a missing "
                    "local dependency; official evaluation remains authoritative."
                )
                if warning not in warnings:
                    warnings.append(warning)

            deviation_count = step_deviation_count
            if step_human_review:
                if state.auto_mode and step_evidence:
                    human_review_required = False
                    review_request = None
                else:
                    human_review_required = True
                    review_request = step_review_request

            if not step_evidence:
                self._report_progress(
                    {
                        "type": "node_complete",
                        "node": "execution",
                        "step_id": step.step_id,
                        "status": "failed",
                        "success": False,
                        "summary": f"Missing completion evidence: {step.description}",
                    }
                )
                break

            current_step_index += 1

            self._report_progress(
                {
                    "type": "node_complete",
                    "node": "execution",
                    "step_id": step.step_id,
                    "status": "completed",
                    "summary": step.description,
                }
            )

            # 更新完成摘要
            completed_descriptions.append(step.description)
            completed_steps_summary = "; ".join(completed_descriptions)

            # 如果偏离触发 Human Review，停止执行
            if human_review_required:
                break

        # 计算剩余任务描述
        remaining_descriptions = [
            f"[{s.step_id}] {s.description}" for s in (state.plan or [])[current_step_index:]
        ]

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ExecutionNode (plan) completed in %.1fms (%d steps)",
            total_duration,
            current_step_index - state.current_step_index,
        )

        return {
            "execution_log": execution_log,
            "errors": errors,
            "current_step_index": current_step_index,
            "accumulated_changes": accumulated_changes,
            "completed_steps_summary": completed_steps_summary,
            "tasks_remaining": remaining_descriptions,
            "deviation_detected": deviation_detected,
            "deviation_count": deviation_count,
            "human_review_required": human_review_required,
            "review_request": review_request,
            "trajectory_steps": trajectory_steps,
            "llm_call_count": state.llm_call_count,
            "estimated_tokens": state.estimated_tokens,
            "steering_instructions": list(state.steering_instructions),
            "memory_hits": list(state.memory_hits),
            "memory_recalled": state.memory_recalled,
            "memory_context": state.memory_context,
            "validation_state": validation_state,
            "warnings": warnings,
        }

    def _build_step_prompt(
        self,
        state: AgentState,
        step: PlanStep,
        tool_definitions: list[Any],
        memory_section: str = "",
    ) -> str:
        """为单个 PlanStep 构建提示词（含 TaskFocus 上下文注入）。

        Args:
            state: Agent 状态
            step: 当前执行步骤
            tool_definitions: 工具定义列表
            memory_section: Phase 6.5 记忆层文本，空字符串时跳过
        """
        total_steps = len(state.plan or [])
        parts: list[str] = [
            "You are a coding agent executing a specific step of a plan.",
            "",
            f"## Current Step ({step.step_id}/{total_steps})",
            f"Description: {step.description}",
            f"Action: {step.action}",
            f"Target: {step.target_file or 'N/A'}",
            f"Acceptance criteria: {step.acceptance_criteria or 'successful tool evidence'}",
            "",
        ]

        # ── Phase 3.5 TaskFocus: 注入当前任务状态 ──────────────
        if state.original_goal_summary:
            parts.append(f"## 当前任务状态\n原始目标: {state.original_goal_summary}\n")
        if state.completed_steps_summary:
            parts.append(f"进度: {state.completed_steps_summary}")
        if state.tasks_remaining:
            parts.append("剩余步骤:\n" + "\n".join(f"- {t}" for t in state.tasks_remaining))
        if state.original_goal_summary or state.completed_steps_summary or state.tasks_remaining:
            parts.append("请严格围绕上述目标执行，不要偏离到未规划的方向。\n")

        if state.context:
            parts.append(f"## Project Context\n{state.context}\n")

        if state.benchmark_instance_id:
            fail_to_pass = "\n".join(
                f"- {test}" for test in state.benchmark_fail_to_pass[:50]
            ) or "- (not provided)"
            pass_to_pass = "\n".join(
                f"- {test}" for test in state.benchmark_pass_to_pass[:50]
            ) or "- (not provided)"
            parts.append(
                "## Benchmark Contract\n"
                f"Instance: {state.benchmark_instance_id}\n"
                "This run must produce a real code patch and executed test output.\n"
                f"FAIL_TO_PASS tests:\n{fail_to_pass}\n"
                f"PASS_TO_PASS tests:\n{pass_to_pass}\n"
                "Use these tests for local feedback when the repository supports them. "
                "Only the official SWE-bench harness determines the final score."
            )

        # Phase 6.5: Memory 层注入
        if memory_section:
            parts.append(memory_section)
            parts.append("")

        parts.append("## Available Tools")
        for td in tool_definitions:
            parts.append(f"- {td.name}: {td.description}")
        parts.append("")

        parts.extend(
            [
                "## Guidelines",
                "- Every file_path/path/cwd argument is relative to the workspace root",
                "- Never prefix paths with the workspace directory name or use absolute paths",
                "- run_terminal already starts at the workspace root; never use cd, /workspace discovery, command substitution $(), or backticks",
                "- Use tools to accomplish this single step",
                "  (not the entire plan)",
                "- For create/modify steps, use at most 3 read/search calls before making the required file mutation",
                "- After writing a file, syntax will be checked automatically",
                "- If syntax errors are reported, fix them promptly",
                "- For frontend or browser UI work, leave a directly renderable index.html entrypoint; use relative asset URLs so the live workspace preview can render HTML/CSS/JS",
                "- When this step is done, respond with a summary",
            ]
        )

        return "\n".join(parts)

    async def _execute_step_tool_loop(
        self,
        state: AgentState,
        step: PlanStep,
        messages: list[dict[str, Any]],
        tool_definitions: list[Any],
        execution_log: list[dict],
        errors: list[str],
        accumulated_changes: list[dict],
        deviation_count: int = 0,
        remaining_descriptions: list[str] | None = None,
        trajectory_steps: list[dict] | None = None,
    ) -> tuple[list[dict] | None, int, bool, dict | None, list[dict], bool]:
        """执行单个步骤的 tool calling 循环（含偏离检测）。

        Returns:
            tuple[accumulated_changes, deviation_count, human_review_required, review_request]
        """
        if trajectory_steps is None:
            trajectory_steps = []
        tool_call_count = 0
        syntax_retries = 0
        openai_tools = self._to_openai_tools(tool_definitions)
        failed_tool_calls: dict[str, int] = {}
        duplicate_failure_stopped = False
        human_review_required = False
        review_request: dict | None = None
        successful_tools: set[str] = set()
        completion_evidence_observed = False
        no_tool_reprompts = 0
        mutation_tools_forced = False
        validation_degraded_observed = False
        required_evidence_tools = _REQUIRED_EVIDENCE_TOOLS_BY_ACTION.get(step.action, set())
        if step.action == "read" and not step.acceptance_criteria:
            # Backward compatibility for persisted plans created before PlanStep
            # carried explicit acceptance criteria. Newly planned read steps are strict.
            required_evidence_tools = set()

        while tool_call_count < self._max_tool_calls:
            active_tools = openai_tools
            mutation_required = bool(
                required_evidence_tools & {"write_file", "apply_patch"}
            )
            if (
                mutation_required
                and not completion_evidence_observed
                and tool_call_count >= _MAX_DISCOVERY_CALLS_BEFORE_MUTATION
            ):
                active_tools = [
                    tool
                    for tool in openai_tools
                    if tool.get("function", {}).get("name") in required_evidence_tools
                ]
                if not mutation_tools_forced:
                    mutation_tools_forced = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Discovery budget is exhausted for this modification step. "
                                "Your next action must mutate the exact target with write_file "
                                "or apply_patch. Do not read or search again."
                            ),
                        }
                    )
            try:
                response = await self._call_llm_with_limit(
                    state=state,
                    messages=messages,
                    tools=active_tools if active_tools else None,
                    tool_choice="auto" if active_tools else None,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                errors.append(f"LLM call failed: {e}")
                break

            if response is None:
                # 成本控制：超出 LLM 调用次数上限
                human_review_required = True
                review_request = state.review_request
                break

            choice = response.choices[0]
            msg = choice.message
            _restore_provider_tool_calls(msg)

            if not msg.tool_calls:
                if getattr(msg, "reasoning_content", None) or not msg.content:
                    diagnostic = _provider_response_diagnostic(
                        msg,
                        finish_reason=getattr(choice, "finish_reason", None),
                    )
                    diagnostic["step_id"] = step.step_id
                    execution_log.append(diagnostic)
                if msg.content:
                    execution_log.append(
                        {
                            "type": "llm_response",
                            "content": msg.content,
                            "step_id": step.step_id,
                            "timestamp": time.time(),
                        }
                    )
                missing_evidence = required_evidence_tools - successful_tools
                if missing_evidence and no_tool_reprompts < _MAX_NO_TOOL_REPROMPTS:
                    no_tool_reprompts += 1
                    messages.append(
                        _assistant_retry_message(msg, "I believe the step is complete.")
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue from your preserved reasoning; do not restart the "
                                "analysis. This step is not complete yet because there is no successful "
                                f"tool evidence. Use one of these required tools now: "
                                f"{', '.join(sorted(missing_evidence))}. Call it immediately; "
                                "do not only describe the work."
                            ),
                        }
                    )
                    continue
                break

            # assistant message with tool calls
            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            # DeepSeek requires explicit null content (not None) in assistant tool_call messages
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": msg.content if msg.content is not None else None,
                "tool_calls": assistant_tool_calls,
            }
            # DeepSeek v4 Flash thinking mode requires reasoning_content to be preserved
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            messages.append(assistant_msg)

            executed_tool_ids: list[str] = []
            pending_deviation_warnings: list[dict] = []

            for tool_call in msg.tool_calls:
                if tool_call_count >= self._max_tool_calls:
                    errors.append(f"Exceeded max tool calls per step ({self._max_tool_calls})")
                    break

                tool_call_count += 1
                tool_name = tool_call.function.name
                tool_args = self._normalize_workspace_args(
                    state, self._parse_tool_args(tool_call.function.arguments)
                )

                # ── Phase 3.5 偏离检测 ──────────────────────────
                is_deviation = self._check_deviation(step, tool_name, tool_args)
                if is_deviation:
                    deviation_count += 1
                    self._report_progress(
                        {
                            "type": "deviation_detected",
                            "node": "execution",
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "deviation_count": deviation_count,
                            "summary": f"Rejected out-of-step tool call: {tool_name}",
                        }
                    )

                    execution_log.append(
                        {
                            "type": "deviation_detected",
                            "step_id": step.step_id,
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "deviation_count": deviation_count,
                            "timestamp": time.time(),
                        }
                    )

                    if deviation_count >= _DEVIATION_HUMAN_LIMIT and state.auto_mode:
                        deviation_count = _DEVIATION_UPGRADE_LIMIT
                        pending_deviation_warnings.append(
                            {
                                "role": "user",
                                "content": (
                                    "Automatic mode rejected that out-of-step call. "
                                    f"Stay within step {step.step_id}; allowed tools are: "
                                    f"{', '.join(sorted(_ALLOWED_TOOLS_BY_ACTION.get(step.action, set())))}. "
                                    f"The exact target is {step.target_file!r}."
                                ),
                            }
                        )
                        continue

                    if deviation_count >= _DEVIATION_HUMAN_LIMIT:
                        # 连续 3 次 → 请求人工审核
                        human_review_required = True
                        review_request = {
                            "review_type": "deviation_detected",
                            "title": "执行偏离检测 — 需要人工审核",
                            "details": {
                                "step_id": step.step_id,
                                "step_description": step.description,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "consecutive_deviations": deviation_count,
                            },
                            "options": ["approve", "reject", "modify", "abort"],
                        }
                        execution_log.append(
                            {
                                "type": "human_review_required",
                                "reason": "repeated_deviation",
                                "deviation_count": deviation_count,
                                "timestamp": time.time(),
                            }
                        )
                        # DeepSeek 要求所有 tool_call_id 都有对应结果
                        for tc in msg.tool_calls:
                            if tc.id not in executed_tool_ids:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": "Tool execution skipped due to deviation limit requiring human review.",
                                    }
                                )

                        # 先 flush 已收集的偏离警告，再添加最终警告
                        for w in pending_deviation_warnings:
                            messages.append(w)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"警告：检测到连续 {deviation_count} 次偏离计划的行为。"
                                    f"执行已暂停，需要人工审核。"
                                ),
                            }
                        )
                        return (
                            accumulated_changes,
                            deviation_count,
                            human_review_required,
                            review_request,
                            trajectory_steps,
                            False,
                        )
                    elif deviation_count >= _DEVIATION_UPGRADE_LIMIT:
                        # 连续 2 次 → 升级警告（收集，在 tool 结果后追加）
                        pending_deviation_warnings.append(
                            {
                                "role": "user",
                                "content": (
                                    f"警告（{deviation_count}/{_DEVIATION_HUMAN_LIMIT}）："
                                    f"你正在偏离当前步骤的规划范围。"
                                    f"当前步骤目标: {step.description}。"
                                    f"请严格围绕当前步骤执行。"
                                ),
                            }
                        )
                        continue
                    else:
                        # 首次偏离 → 警告（收集，在 tool 结果后追加）
                        pending_deviation_warnings.append(
                            {
                                "role": "user",
                                "content": (
                                    f"注意：工具 '{tool_name}' 的调用似乎偏离了当前步骤的规划。"
                                    f"当前步骤目标: {step.description}。"
                                    f"请确保工具调用在步骤范围内。"
                                ),
                            }
                        )
                        continue
                logger.info(
                    "Step %d tool #%d: %s(%s)",
                    step.step_id,
                    tool_call_count,
                    tool_name,
                    tool_args,
                )

                # ── Rollback: write_file/delete_file 前保存原始内容 ──
                orig_file_path, original_content = self._save_original_content_before_tool(
                    tool_name, tool_args, state.project_root
                )

                tool_result = await self._tool_gateway.execute_tool(tool_name, tool_args)
                elapsed = tool_result.duration_ms

                log_entry: dict[str, Any] = {
                    "type": "tool_call",
                    "step_id": step.step_id,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "success": tool_result.success,
                    "duration_ms": elapsed,
                    "timestamp": time.time(),
                }

                if tool_result.success:
                    log_entry["result"] = tool_result.data
                    successful_tools.add(tool_name)
                else:
                    log_entry["error"] = tool_result.error_message
                    log_entry["error_code"] = tool_result.error_code
                    log_entry["retryable"] = bool(tool_result.retryable)
                    log_entry["suggested_recovery"] = tool_result.suggested_recovery
                    logger.warning(
                        "Step %d tool '%s' failed: %s",
                        step.step_id,
                        tool_name,
                        tool_result.error_message,
                    )

                if _is_validation_dependency_failure(
                    step=step,
                    tool_name=tool_name,
                    tool_result=tool_result,
                ):
                    validation_degraded_observed = True
                    log_entry["validation_degraded"] = True
                    log_entry["degradation_reason"] = "missing_dependency"

                execution_log.append(log_entry)

                # 上报 tool_call 和 tool_result 事件
                self._report_progress(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "params": tool_args,
                    }
                )
                summary = (
                    f"{'Succeeded' if tool_result.success else 'Failed'}: "
                    f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"
                )
                self._report_progress(
                    {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": tool_result.success,
                        "summary": summary,
                        "error_code": tool_result.error_code,
                        "retryable": bool(tool_result.retryable),
                        "suggested_recovery": tool_result.suggested_recovery,
                    }
                )

                # Phase 7.6: 记录工具调用到轨迹
                trajectory_steps.append(
                    {
                        "node_name": "execution",
                        "step_type": "tool_call",
                        "step_id": step.step_id,
                        "tool_name": tool_name,
                        "tool_args_summary": json.dumps(tool_args, ensure_ascii=False)[:100],
                        "success": tool_result.success,
                        "duration_ms": elapsed,
                    }
                )

                # 添加工具结果到消息
                result_content = _tool_result_message(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )
                executed_tool_ids.append(tool_call.id)

                failed_call = _record_failed_tool_call(
                    failed_tool_calls,
                    scope=f"plan-step-{step.step_id}",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result,
                )
                if (
                    failed_call is not None
                    and failed_call[1] >= _MAX_IDENTICAL_FAILED_TOOL_CALLS
                ):
                    signature, failure_count = failed_call
                    duplicate_failure_stopped = True
                    errors.append(
                        "Stopped repeated identical failed tool call "
                        f"'{tool_name}' after {failure_count} attempts"
                    )
                    execution_log.append(
                        {
                            "type": "duplicate_tool_failure_breaker",
                            "step_id": step.step_id,
                            "tool_name": tool_name,
                            "error_code": tool_result.error_code,
                            "failure_count": failure_count,
                            "signature": signature,
                            "timestamp": time.time(),
                        }
                    )
                    self._report_progress(
                        {
                            "type": "tool_loop_stopped",
                            "reason": "duplicate_tool_failure",
                            "tool": tool_name,
                            "failure_count": failure_count,
                        }
                    )
                    break

                # write_file → track change (含 original_content) + syntax check
                if tool_name in {"write_file", "apply_patch"} and tool_result.success:
                    if orig_file_path:
                        action = tool_args.get("mode", "modify")
                        accumulated_changes.append(
                            {
                                "file_path": orig_file_path,
                                "step_id": step.step_id,
                                "action": action,
                                "original_content": original_content,
                                "timestamp": time.time(),
                            }
                        )
                        syntax_retries = await self._run_syntax_check(
                            file_path=orig_file_path,
                            project_root=state.project_root,
                            messages=messages,
                            execution_log=execution_log,
                            retry_count=syntax_retries,
                            step_id=step.step_id,
                        )

                # delete_file → track change (含 original_content 用于重建)
                if tool_name == "delete_file" and tool_result.success:
                    if orig_file_path:
                        accumulated_changes.append(
                            {
                                "file_path": orig_file_path,
                                "step_id": step.step_id,
                                "action": "delete",
                                "original_content": original_content,
                                "timestamp": time.time(),
                            }
                        )

                if tool_result.success and tool_name in required_evidence_tools:
                    completion_evidence_observed = True
                    break

                if validation_degraded_observed and tool_name in required_evidence_tools:
                    execution_log.append(
                        {
                            "type": "validation_degraded",
                            "step_id": step.step_id,
                            "tool_name": tool_name,
                            "reason": "missing_dependency",
                            "error_code": tool_result.error_code,
                            "timestamp": time.time(),
                        }
                    )
                    break

                if tool_call_count >= self._max_tool_calls:
                    break

            # DeepSeek 要求所有 tool_call_id 都有对应结果（必须紧跟在 assistant msg 之后）
            for tc in msg.tool_calls:
                if tc.id not in executed_tool_ids:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Tool execution skipped (deviation or limit reached).",
                        }
                    )

            # 在 tool 结果之后追加偏离警告（确保 assistant→tool→user 的顺序）
            for w in pending_deviation_warnings:
                messages.append(w)

            if duplicate_failure_stopped:
                break

            if completion_evidence_observed:
                break

            if tool_call_count >= self._max_tool_calls:
                break

        evidence_satisfied = (
            not required_evidence_tools
            or bool(required_evidence_tools & successful_tools)
            or validation_degraded_observed
        )
        if not evidence_satisfied and not human_review_required:
            errors.append(
                f"Step evidence missing for step {step.step_id}: expected a successful "
                f"{', '.join(sorted(required_evidence_tools))} call"
            )

        return (
            accumulated_changes,
            deviation_count,
            human_review_required,
            review_request,
            trajectory_steps,
            evidence_satisfied,
        )

    # ── Phase 3.5 偏离检测 ────────────────────────────────────

    def _check_deviation(self, step: PlanStep, tool_name: str, tool_args: dict) -> bool:
        """检测当前工具调用是否偏离当前步骤的规划范围。

        检测规则：
        1. 如果 tool_name 为 write_file，检查 file_path 是否在当前步骤的
           target_file 范围内
        2. 如果 tool_name 不在当前步骤的合理工具列表中
        3. 如果 LLM 尝试修改计划外的文件（write_file 写入非目标文件）

        Args:
            step: 当前执行步骤
            tool_name: 工具名称
            tool_args: 工具参数字典

        Returns:
            True 表示检测到偏离
        """
        # 步骤 1: write_file 的 file_path 一致性检查
        if tool_name in {"write_file", "apply_patch"}:
            file_path = tool_args.get("file_path", "")
            if file_path and step.target_file:
                # 检查是否是命令行操作步骤（command 类型不检查文件路径）
                if step.action != "command":
                    # 文件路径应匹配步骤目标文件
                    norm_file = file_path.replace("\\", "/")
                    norm_target = step.target_file.replace("\\", "/")
                    if not (norm_file == norm_target or norm_file.endswith("/" + norm_target)):
                        return True

        # 步骤 2: 工具名称合理性检查
        allowed = _ALLOWED_TOOLS_BY_ACTION.get(step.action, set())
        if allowed and tool_name not in allowed and not tool_name.startswith("mcp__"):
            return True

        return False

    # ── Phase 4.A.5 修复模式 ────────────────────────────────────

    def _needs_repair(self, state: AgentState) -> bool:
        """检查是否需要进入验证反馈驱动的修复模式。

        当验证结果中有失败项，且所有计划步骤已完成（或无计划）时，
        进入修复模式让 LLM 根据验证报告自动修复代码。

        Returns:
            True 表示应进入修复模式
        """
        if not state.validation_results:
            return False

        any_failed = any(not r.passed for r in state.validation_results)
        if not any_failed:
            return False

        # 仅当所有计划步骤已完成（或无计划）时才进入修复模式
        plan_done = (
            state.plan is None
            or len(state.plan) == 0
            or state.current_step_index >= len(state.plan)
        )
        return plan_done

    async def _execute_repair_mode(self, state: AgentState) -> dict[str, Any]:
        """验证反馈驱动的自动修复模式。

        将验证报告和修复建议作为额外上下文注入 LLM，让 LLM 自动
        定位和修复问题。修复完成后递增 retry_count。

        Returns:
            dict: 更新的状态字段
        """
        start_time = time.monotonic()
        execution_log: list[dict] = list(state.execution_log)
        errors: list[str] = list(state.errors)
        accumulated_changes: list[dict] = list(state.accumulated_changes)
        trajectory_steps: list[dict] = list(state.trajectory_steps)

        # 上报 node_start 事件
        self._report_progress(
            {
                "type": "node_start",
                "node": "execution/repair",
                "data": {"retry_count": state.retry_count + 1},
            }
        )

        # ── Phase 5.4: 构建修复上下文 ──────────────────────────
        new_errors, pre_existing = self._extract_structured_errors(state.validation_results)
        attempt_number = state.retry_count + 1
        last_fix_summary = state.repair_context.last_fix_summary if state.repair_context else None
        ctx = RepairContext(
            attempt_number=attempt_number,
            errors=new_errors,
            pre_existing_errors=pre_existing,
            last_fix_summary=last_fix_summary,
        )
        state.repair_context = ctx

        # 构建结构化的修复提示词
        repair_prompt = self._build_repair_prompt(state)

        tool_definitions = self._available_tools(state)
        memory_section = await self._assemble_memory_section(state)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._build_system_prompt(
                    state,
                    tool_definitions,
                    memory_section=memory_section,
                ),
            },
            {"role": "user", "content": repair_prompt},
        ]

        # 记录修复模式进入日志
        new_retry_count = state.retry_count + 1
        execution_log.append(
            {
                "type": "repair_mode",
                "retry_count": new_retry_count,
                "validation_failure_count": sum(
                    1 for r in state.validation_results if not r.passed
                ),
                "timestamp": time.time(),
            }
        )

        logger.info(
            "Entering repair mode (retry %d/%d)",
            new_retry_count,
            _MAX_VALIDATION_ROUNDS,
        )

        # 执行修复工具调用循环（使用较低的工具调用上限）
        tool_call_count = 0
        syntax_retries = 0
        openai_tools = self._to_openai_tools(tool_definitions)
        failed_tool_calls: dict[str, int] = {}
        duplicate_failure_stopped = False

        while tool_call_count < _MAX_REPAIR_TOOL_CALLS:
            try:
                response = await self._call_llm_with_limit(
                    state=state,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                )
            except Exception as e:
                logger.error("Repair LLM call failed: %s", e)
                errors.append(f"Repair LLM call failed: {e}")
                break

            if response is None:
                # 成本控制：超出 LLM 调用次数上限
                break

            choice = response.choices[0]
            msg = choice.message
            _restore_provider_tool_calls(msg)

            if not msg.tool_calls:
                if getattr(msg, "reasoning_content", None) or not msg.content:
                    diagnostic = _provider_response_diagnostic(
                        msg,
                        finish_reason=getattr(choice, "finish_reason", None),
                    )
                    diagnostic["repair_mode"] = True
                    execution_log.append(diagnostic)
                if msg.content:
                    execution_log.append(
                        {
                            "type": "llm_response",
                            "content": msg.content,
                            "repair_mode": True,
                            "timestamp": time.time(),
                        }
                    )
                break

            # assistant message with tool calls
            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": msg.content if msg.content is not None else None,
                "tool_calls": assistant_tool_calls,
            }
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            messages.append(assistant_msg)

            executed_tool_ids: list[str] = []

            for tool_call in msg.tool_calls:
                if tool_call_count >= _MAX_REPAIR_TOOL_CALLS:
                    errors.append(f"Exceeded max repair tool calls ({_MAX_REPAIR_TOOL_CALLS})")
                    break

                tool_call_count += 1
                tool_name = tool_call.function.name
                tool_args = self._normalize_workspace_args(
                    state, self._parse_tool_args(tool_call.function.arguments)
                )

                logger.info(
                    "Repair tool #%d: %s(%s)",
                    tool_call_count,
                    tool_name,
                    tool_args,
                )

                # ── Rollback: write_file/delete_file 前保存原始内容 ──
                orig_file_path, original_content = self._save_original_content_before_tool(
                    tool_name, tool_args, state.project_root
                )

                tool_result = await self._tool_gateway.execute_tool(tool_name, tool_args)

                log_entry: dict[str, Any] = {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "success": tool_result.success,
                    "duration_ms": tool_result.duration_ms,
                    "repair_mode": True,
                    "timestamp": time.time(),
                }

                if tool_result.success:
                    log_entry["result"] = tool_result.data
                else:
                    log_entry["error"] = tool_result.error_message
                    log_entry["error_code"] = tool_result.error_code
                    log_entry["retryable"] = bool(tool_result.retryable)
                    log_entry["suggested_recovery"] = tool_result.suggested_recovery
                    logger.warning(
                        "Repair tool '%s' failed: %s",
                        tool_name,
                        tool_result.error_message,
                    )

                execution_log.append(log_entry)

                # 上报 tool_call 和 tool_result 事件
                self._report_progress(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "params": tool_args,
                    }
                )
                summary = (
                    f"{'Succeeded' if tool_result.success else 'Failed'}: "
                    f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"
                )
                self._report_progress(
                    {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": tool_result.success,
                        "summary": summary,
                        "error_code": tool_result.error_code,
                        "retryable": bool(tool_result.retryable),
                        "suggested_recovery": tool_result.suggested_recovery,
                    }
                )

                # Phase 7.6: 记录修复工具调用到轨迹
                trajectory_steps.append(
                    {
                        "node_name": "execution",
                        "step_type": "tool_call",
                        "step_id": -1,
                        "tool_name": tool_name,
                        "tool_args_summary": json.dumps(tool_args, ensure_ascii=False)[:100],
                        "success": tool_result.success,
                        "duration_ms": tool_result.duration_ms,
                        "repair_mode": True,
                    }
                )

                result_content = _tool_result_message(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )
                executed_tool_ids.append(tool_call.id)

                failed_call = _record_failed_tool_call(
                    failed_tool_calls,
                    scope="repair",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result,
                )
                if (
                    failed_call is not None
                    and failed_call[1] >= _MAX_IDENTICAL_FAILED_TOOL_CALLS
                ):
                    signature, failure_count = failed_call
                    duplicate_failure_stopped = True
                    errors.append(
                        "Stopped repeated identical failed repair tool call "
                        f"'{tool_name}' after {failure_count} attempts"
                    )
                    execution_log.append(
                        {
                            "type": "duplicate_tool_failure_breaker",
                            "repair_mode": True,
                            "tool_name": tool_name,
                            "error_code": tool_result.error_code,
                            "failure_count": failure_count,
                            "signature": signature,
                            "timestamp": time.time(),
                        }
                    )
                    break

                # write_file → track change (含 original_content) + syntax check
                if tool_name in {"write_file", "apply_patch"} and tool_result.success:
                    if orig_file_path:
                        action = tool_args.get("mode", "modify")
                        accumulated_changes.append(
                            {
                                "file_path": orig_file_path,
                                "step_id": -1,  # repair mode step
                                "action": action,
                                "original_content": original_content,
                                "repair_mode": True,
                                "timestamp": time.time(),
                            }
                        )
                        syntax_retries = await self._run_syntax_check(
                            file_path=orig_file_path,
                            project_root="",
                            messages=messages,
                            execution_log=execution_log,
                            retry_count=syntax_retries,
                            step_id=None,
                        )

                # delete_file → track change (含 original_content 用于重建)
                if tool_name == "delete_file" and tool_result.success:
                    if orig_file_path:
                        accumulated_changes.append(
                            {
                                "file_path": orig_file_path,
                                "step_id": -1,
                                "action": "delete",
                                "original_content": original_content,
                                "repair_mode": True,
                                "timestamp": time.time(),
                            }
                        )

                if tool_call_count >= _MAX_REPAIR_TOOL_CALLS:
                    break

            # fill missing tool responses
            for tc in msg.tool_calls:
                if tc.id not in executed_tool_ids:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Tool execution skipped (limit reached).",
                        }
                    )

            if duplicate_failure_stopped:
                break

            if tool_call_count >= _MAX_REPAIR_TOOL_CALLS:
                logger.info("Repair mode reached max tool calls (%d)", _MAX_REPAIR_TOOL_CALLS)
                break

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "Repair mode completed in %.1fms (%d tool calls, retry %d)",
            total_duration,
            tool_call_count,
            new_retry_count,
        )

        # ── Phase 5.4: 生成 last_fix_summary ────────────────────
        tool_calls_log = [
            e for e in execution_log if e.get("type") == "tool_call" and e.get("repair_mode")
        ]
        summary_parts: list[str] = []
        for entry in tool_calls_log:
            tn = entry.get("tool_name", "")
            args = entry.get("arguments", {})
            if tn in {"write_file", "apply_patch"}:
                summary_parts.append(f"Modified {args.get('file_path', 'unknown')}")
            elif tn == "read_file":
                summary_parts.append(f"Read {args.get('file_path', 'unknown')}")
            elif tn == "delete_file":
                summary_parts.append(f"Deleted {args.get('file_path', 'unknown')}")
        ctx.last_fix_summary = (
            "; ".join(summary_parts)
            if summary_parts
            else (ctx.last_fix_summary or "No changes made")
        )

        return {
            "execution_log": execution_log,
            "errors": errors,
            "accumulated_changes": accumulated_changes,
            "retry_count": new_retry_count,
            "repair_context": ctx,
            "trajectory_steps": trajectory_steps,
            "repair_rounds": state.repair_rounds + 1,
            "llm_call_count": state.llm_call_count,
            "estimated_tokens": state.estimated_tokens,
            "steering_instructions": list(state.steering_instructions),
            "memory_hits": list(state.memory_hits),
        }

    def _format_validation_report(
        self,
        validation_results: list[Any],
    ) -> str:
        """将 ValidationResult 列表格式化为可读的验证报告文本。"""
        lines: list[str] = []
        for i, r in enumerate(validation_results):
            status = "✅ PASSED" if r.passed else "❌ FAILED"
            label = ["Syntax", "Static Analysis", "Runtime"][i] if i < 3 else f"Layer {i + 1}"
            lines.append(f"### Layer {i + 1}: {label}")
            lines.append(f"Status: {status}")
            if r.errors:
                lines.append(f"Errors ({len(r.errors)}):")
                for err in r.errors[:5]:  # 最多显示 5 个
                    loc = (
                        f"{err.file_path}:{err.line}"
                        if hasattr(err, "file_path") and err.file_path
                        else ""
                    )
                    msg = err.message if hasattr(err, "message") else str(err)
                    lines.append(f"  - {loc}: {msg}" if loc else f"  - {msg}")
                if len(r.errors) > 5:
                    lines.append(f"  - ... and {len(r.errors) - 5} more")
            if r.warnings:
                lines.append(f"Warnings ({len(r.warnings)}):")
                for w in r.warnings[:3]:
                    lines.append(f"  - {w.message if hasattr(w, 'message') else w}")
            lines.append("")
        return "\n".join(lines)

    def _format_fix_suggestions(self, state: AgentState) -> str:
        """从 state 中提取 FixSuggestion 并格式化为文本。

        查找 execution_log 中的验证报告或 fix_suggestions 数据。
        """
        # 尝试从 execution_log 中提取 fix_suggestions
        suggestion_texts: list[str] = []
        for entry in reversed(state.execution_log):
            if entry.get("type") == "validation_report":
                fix_count = entry.get("fix_suggestion_count", 0)
                if fix_count > 0:
                    suggestion_texts.append(
                        f"Found {fix_count} fix suggestion(s) in validation report."
                    )
                break

        if not suggestion_texts:
            return "No specific fix suggestions available."

        return "\n".join(suggestion_texts)

    # ── Phase 5.4: 修复上下文结构化 ────────────────────────────

    def _extract_structured_errors(
        self,
        validation_results: list[Any],
    ) -> tuple[list[StructuredError], list[str]]:
        """从 ValidationResult 列表中提取结构化错误。

        validation_results 顺序：0=syntax, 1=lint, 2=runtime。
        当前系统中尚无 is_pre_existing 机制，所有错误均标记为新的。

        Returns:
            tuple[list[StructuredError], list[str]]:
                (需要修复的新错误列表, 预存在的错误描述列表)
        """
        error_types: list[str] = ["syntax", "lint", "runtime"]
        new_errors: list[StructuredError] = []
        pre_existing: list[str] = []

        for i, vr in enumerate(validation_results):
            error_type = error_types[i] if i < len(error_types) else "runtime"
            for err in vr.errors:
                file_path = err.file_path if hasattr(err, "file_path") and err.file_path else ""
                line_number: int | None = err.line if hasattr(err, "line") and err.line else None
                message = err.message if hasattr(err, "message") else str(err)
                new_errors.append(
                    StructuredError(
                        file_path=file_path,
                        line_number=line_number,
                        error_type=error_type,  # type: ignore[arg-type]
                        message=message,
                    )
                )

        return new_errors, pre_existing

    def _build_repair_prompt(self, state: AgentState) -> str:
        """构建结构化的修复提示词。

        当 state.repair_context 存在时使用结构化格式，
        否则回退到原有的纯文本验证报告。
        """
        ctx = state.repair_context
        if ctx is None:
            # 兼容旧逻辑：无 RepairContext 时使用纯文本格式
            validation_report = self._format_validation_report(state.validation_results)
            fix_suggestions_text = self._format_fix_suggestions(state)
            return _REPAIR_MODE_PROMPT.format(
                validation_report=validation_report,
                fix_suggestions_text=fix_suggestions_text,
                max_repair_tool_calls=_MAX_REPAIR_TOOL_CALLS,
            )

        lines: list[str] = [
            f"## 修复尝试 #{ctx.attempt_number}",
            "",
        ]
        if ctx.last_fix_summary:
            lines += [f"**上次修复内容：** {ctx.last_fix_summary}", ""]

        if state.reflection:
            guidance = state.reflection.get("revision_guidance", {})
            lines += [
                f"**反思假设：** {guidance.get('hypothesis', '')}",
                f"**修订要求：** {guidance.get('instruction', '')}",
                f"**避免原样重试：** {'是' if state.reflection.get('avoid_repeat') else '否'}",
                "",
            ]

        lines += [
            f"**需要修复的错误（{len(ctx.errors)} 个）：**",
        ]
        for err in ctx.errors:
            loc = (
                f"{err.file_path}:{err.line_number}"
                if err.line_number is not None
                else err.file_path
            )
            lines.append(f"- `{loc}` [{err.error_type}] {err.message}")

        if ctx.pre_existing_errors:
            lines += [
                "",
                f"**以下是预存在的错误，请勿修复（{len(ctx.pre_existing_errors)} 个）：**",
            ]
            for e in ctx.pre_existing_errors:
                lines.append(f"- {e}")

        lines += [
            "",
            "请分析以上错误，读取相关文件，修复代码中的问题。",
            "注意：",
            "1. 只修复列表中指出的问题",
            "2. 不要修改与验证失败无关的文件",
            "3. 修复后代码必须通过所有验证",
            f"4. 尽量在 {_MAX_REPAIR_TOOL_CALLS} 次工具调用内完成修复",
        ]

        return "\n".join(lines)

    # ── Phase 1a 直连模式 ────────────────────────────────────

    async def _execute_direct(self, state: AgentState) -> dict[str, Any]:
        """Phase 1a 直连模式：直接通过 LLM tool calling 循环执行。

        copy from Phase 1a 的 __call__ 实现。
        """
        start_time = time.monotonic()
        execution_log: list[dict] = list(state.execution_log)
        errors: list[str] = list(state.errors)
        trajectory_steps: list[dict] = list(state.trajectory_steps)
        accumulated_changes: list[dict] = list(state.accumulated_changes)

        # 上报 node_start 事件
        self._report_progress(
            {
                "type": "node_start",
                "node": "execution/direct",
                "data": {"query": state.user_request[:200]},
            }
        )

        tool_definitions = self._available_tools(state)
        memory_section = await self._assemble_memory_section(state)
        system_prompt = self._build_system_prompt(
            state,
            tool_definitions,
            memory_section=memory_section,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in state.conversation_history[-20:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content[:12000]})
        if not messages or messages[-1].get("content") != state.user_request:
            messages.append({"role": "user", "content": state.user_request})

        tool_call_count = 0
        syntax_retries = 0
        no_tool_reprompts = 0
        benchmark_completion_reprompts = 0
        mutation_force_reprompts = 0
        benchmark_test_attempted = any(
            entry.get("type") == "tool_call"
            and entry.get("tool_name") == "run_terminal"
            and _is_test_command((entry.get("arguments") or {}).get("command"))
            for entry in execution_log
        )
        benchmark_discovery_calls = 0
        mentioned_source_paths = _mentioned_source_paths(state.user_request)
        inspected_explicit_target = False
        mutation_tools_forced = False
        patch_conflict_recovery = False
        mutation_context_reads = 0
        benchmark_stage = "locate"
        if state.benchmark_instance_id:
            execution_log.append({
                "type": "benchmark_stage",
                "stage": benchmark_stage,
                "reason": "begin repository localization",
                "timestamp": time.time(),
            })
        failed_tool_calls: dict[str, int] = {}
        unexposed_tool_calls = 0
        duplicate_failure_stopped = False
        capability_violation_stopped = False
        while tool_call_count < self._max_tool_calls:
            active_tool_definitions = tool_definitions
            discovery_limit = (
                _MAX_BENCHMARK_DISCOVERY_WITH_EXPLICIT_TARGET
                if inspected_explicit_target
                else _MAX_BENCHMARK_DISCOVERY_WITHOUT_TARGET
            )
            if (
                state.benchmark_instance_id
                and not accumulated_changes
                and benchmark_discovery_calls >= discovery_limit
            ):
                allowed_names = {"write_file", "apply_patch"}
                if (
                    mutation_context_reads < _MAX_MUTATION_CONTEXT_READS
                    or patch_conflict_recovery
                ):
                    allowed_names.add("read_file")
                active_tool_definitions = [
                    tool
                    for tool in tool_definitions
                    if tool.name in allowed_names
                ]
                if not mutation_tools_forced:
                    mutation_tools_forced = True
                    execution_log.append({
                        "type": "benchmark_stagnation_guard",
                        "discovery_calls": benchmark_discovery_calls,
                        "explicit_target_inspected": inspected_explicit_target,
                        "timestamp": time.time(),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            "Repository discovery is no longer producing a candidate change. "
                            "Use the source context already collected to apply the smallest "
                            "justified patch now. Change the hypothesis only after test evidence."
                        ),
                    })
            openai_tools = self._to_openai_tools(active_tool_definitions)
            active_tool_names = {tool.name for tool in active_tool_definitions}

            try:
                response = await self._call_llm_with_limit(
                    state=state,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                errors.append(f"LLM call failed: {e}")
                break

            if response is None:
                # 成本控制：超出 LLM 调用次数上限
                break

            choice = response.choices[0]
            msg = choice.message
            _restore_provider_tool_calls(msg)

            if not msg.tool_calls:
                if getattr(msg, "reasoning_content", None) or not msg.content:
                    execution_log.append(_provider_response_diagnostic(
                        msg,
                        finish_reason=getattr(choice, "finish_reason", None),
                    ))
                if msg.content:
                    execution_log.append(
                        {
                            "type": "llm_response",
                            "content": msg.content,
                            "timestamp": time.time(),
                        }
                    )
                is_action_follow_up = any(
                    item.get("role") == "assistant"
                    for item in state.conversation_history[:-1]
                )
                if (
                    is_action_follow_up
                    and tool_definitions
                    and tool_call_count == 0
                    and no_tool_reprompts < _MAX_NO_TOOL_REPROMPTS
                ):
                    no_tool_reprompts += 1
                    messages.append(
                        _assistant_retry_message(
                            msg, "I can answer from the earlier turn."
                        )
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have not acted on the latest follow-up yet. Do not repeat the "
                            "previous answer. Use the smallest relevant tool now and base the "
                            "final answer on fresh evidence from this turn."
                        ),
                    })
                    continue
                if (
                    state.benchmark_instance_id
                    and mutation_tools_forced
                    and not accumulated_changes
                    and mutation_force_reprompts < _MAX_MUTATION_FORCE_REPROMPTS
                ):
                    mutation_force_reprompts += 1
                    messages.append(
                        _assistant_retry_message(msg, "I did not apply a patch.")
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "The current mutation phase requires an actual tool call. "
                            "Do not return prose or more analysis. Call exactly one of "
                            "the exposed write_file/apply_patch tools now, using the "
                            "smallest change justified by the source already inspected."
                        ),
                    })
                    continue
                if state.benchmark_instance_id and (
                    not accumulated_changes or not benchmark_test_attempted
                ) and benchmark_completion_reprompts < _MAX_BENCHMARK_COMPLETION_REPROMPTS:
                    benchmark_completion_reprompts += 1
                    messages.append(
                        _assistant_retry_message(msg, "I believe the task is complete.")
                    )
                    if not accumulated_changes:
                        instruction = (
                            "This benchmark task is not complete: no successful file "
                            "mutation has been recorded. Inspect the exact source context, "
                            "then use write_file or apply_patch for the smallest justified fix."
                        )
                    else:
                        instruction = (
                            "The candidate patch has not been tested. Use run_terminal now "
                            "to execute the most targeted FAIL_TO_PASS test available. "
                            "A non-zero test result is still useful evidence; inspect it and "
                            "revise the patch if necessary."
                        )
                    messages.append({
                        "role": "user",
                        "content": instruction,
                    })
                    continue
                break

            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            assistant_msg: dict[str, object] = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": assistant_tool_calls,
            }
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            messages.append(assistant_msg)

            executed_tool_ids: list[str] = []

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_is_exposed = tool_name in active_tool_names
                if tool_is_exposed and tool_call_count >= self._max_tool_calls:
                    errors.append(f"Exceeded max tool calls ({self._max_tool_calls})")
                    break

                if tool_is_exposed:
                    tool_call_count += 1
                else:
                    unexposed_tool_calls += 1
                tool_args = self._normalize_workspace_args(
                    state, self._parse_tool_args(tool_call.function.arguments)
                )

                logger.info(
                    "Tool call #%d: %s(%s)",
                    tool_call_count + unexposed_tool_calls,
                    tool_name,
                    tool_args,
                )

                if not tool_is_exposed:
                    orig_file_path, original_content = None, None
                    tool_result = ToolResult(
                        success=False,
                        error_message=(
                            f"Tool '{tool_name}' is not exposed in the current capability "
                            f"phase. Allowed tools: {', '.join(sorted(active_tool_names)) or 'none'}."
                        ),
                        error_code="CAPABILITY_NOT_EXPOSED",
                        retryable=True,
                        suggested_recovery=(
                            "Choose one of the currently exposed tools and use the evidence "
                            "already collected."
                        ),
                    )
                else:
                    # ── Rollback: write_file/delete_file 前保存原始内容 ──
                    orig_file_path, original_content = self._save_original_content_before_tool(
                        tool_name, tool_args, state.project_root
                    )
                    tool_result = await self._tool_gateway.execute_tool(tool_name, tool_args)
                elapsed = tool_result.duration_ms

                log_entry: dict[str, Any] = {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "success": tool_result.success,
                    "duration_ms": elapsed,
                    "timestamp": time.time(),
                }

                if tool_result.success:
                    log_entry["result"] = tool_result.data
                    logger.info(
                        "Tool '%s' succeeded in %.1fms",
                        tool_name,
                        elapsed,
                    )
                else:
                    log_entry["error"] = tool_result.error_message
                    log_entry["error_code"] = tool_result.error_code
                    log_entry["retryable"] = bool(tool_result.retryable)
                    log_entry["suggested_recovery"] = tool_result.suggested_recovery
                    logger.warning(
                        "Tool '%s' failed: %s",
                        tool_name,
                        tool_result.error_message,
                    )

                execution_log.append(log_entry)

                if state.benchmark_instance_id:
                    next_stage = benchmark_stage
                    stage_reason = ""
                    terminal_is_test = tool_name == "run_terminal" and _is_test_command(
                        tool_args.get("command")
                    )
                    if terminal_is_test:
                        benchmark_test_attempted = True
                        next_stage = "regression_test" if tool_result.success else "evidence_revise"
                        stage_reason = "targeted test command executed"
                    elif tool_name in {"write_file", "apply_patch", "delete_file"} and tool_result.success:
                        next_stage = "target_test"
                        stage_reason = "candidate mutation recorded"
                    elif tool_name == "read_file" and tool_result.success:
                        next_stage = "hypothesize"
                        stage_reason = "source context inspected"
                    elif tool_name in {
                        "list_files", "search_code", "get_diagnostics", "run_terminal"
                    } and tool_result.success:
                        next_stage = "inspect"
                        stage_reason = "repository evidence discovered"
                    if next_stage != benchmark_stage:
                        benchmark_stage = next_stage
                        execution_log.append({
                            "type": "benchmark_stage",
                            "stage": benchmark_stage,
                            "reason": stage_reason,
                            "timestamp": time.time(),
                        })

                    is_discovery = tool_name in {
                        "read_file", "list_files", "search_code", "get_diagnostics"
                    } or (tool_name == "run_terminal" and not terminal_is_test)
                    if is_discovery:
                        benchmark_discovery_calls += 1
                    if tool_name == "read_file" and tool_result.success:
                        read_path = str(tool_args.get("file_path", "")).replace("\\", "/").lower()
                        inspected_explicit_target = (
                            inspected_explicit_target
                            or read_path in mentioned_source_paths
                        )

                # 上报 tool_call 和 tool_result 事件
                self._report_progress(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "params": tool_args,
                    }
                )
                summary = (
                    f"{'Succeeded' if tool_result.success else 'Failed'}: "
                    f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"
                )
                self._report_progress(
                    {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": tool_result.success,
                        "summary": summary,
                        "error_code": tool_result.error_code,
                        "retryable": bool(tool_result.retryable),
                        "suggested_recovery": tool_result.suggested_recovery,
                    }
                )

                # Phase 7.6: 记录工具调用到轨迹
                trajectory_steps.append(
                    {
                        "node_name": "execution",
                        "step_type": "tool_call",
                        "step_id": 0,
                        "tool_name": tool_name,
                        "tool_args_summary": json.dumps(tool_args, ensure_ascii=False)[:100],
                        "success": tool_result.success,
                        "duration_ms": elapsed,
                    }
                )

                result_content = _tool_result_message(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content,
                    }
                )
                executed_tool_ids.append(tool_call.id)

                failed_call = _record_failed_tool_call(
                    failed_tool_calls,
                    scope="direct",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=tool_result,
                )
                if (
                    failed_call is not None
                    and failed_call[1] >= _MAX_IDENTICAL_FAILED_TOOL_CALLS
                ):
                    signature, failure_count = failed_call
                    duplicate_failure_stopped = True
                    errors.append(
                        "Stopped repeated identical failed tool call "
                        f"'{tool_name}' after {failure_count} attempts"
                    )
                    execution_log.append(
                        {
                            "type": "duplicate_tool_failure_breaker",
                            "tool_name": tool_name,
                            "error_code": tool_result.error_code,
                            "failure_count": failure_count,
                            "signature": signature,
                            "timestamp": time.time(),
                        }
                    )
                    break

                if tool_result.error_code == "PATCH_CONFLICT":
                    patch_conflict_recovery = True
                    execution_log.append(
                        {
                            "type": "benchmark_patch_conflict_recovery",
                            "tool_name": tool_name,
                            "timestamp": time.time(),
                        }
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "The candidate patch did not match the current file. On the "
                            "next turn, re-read only the target range, then regenerate "
                            "one exact minimal patch from that current text."
                        ),
                    })

                if unexposed_tool_calls >= _MAX_UNEXPOSED_TOOL_CALLS:
                    capability_violation_stopped = True
                    errors.append(
                        "Stopped repeated requests for tools outside the active "
                        f"capability phase after {unexposed_tool_calls} attempts"
                    )
                    execution_log.append(
                        {
                            "type": "capability_violation_breaker",
                            "failure_count": unexposed_tool_calls,
                            "timestamp": time.time(),
                        }
                    )
                    break

                if (
                    mutation_tools_forced
                    and tool_name == "read_file"
                    and tool_result.success
                ):
                    mutation_context_reads += 1

                if tool_name in {"write_file", "apply_patch"} and tool_result.success:
                    patch_conflict_recovery = False
                    file_path = tool_args.get("file_path", "")
                    if file_path:
                        accumulated_changes.append(
                            {
                                "file_path": file_path,
                                "change_type": "modified"
                                if original_content is not None
                                else "created",
                                "tool": tool_name,
                            }
                        )
                        syntax_retries = await self._run_syntax_check(
                            file_path=file_path,
                            project_root=state.project_root,
                            messages=messages,
                            execution_log=execution_log,
                            retry_count=syntax_retries,
                            step_id=None,
                        )

                if tool_name == "delete_file" and tool_result.success and orig_file_path:
                    accumulated_changes.append({
                        "file_path": orig_file_path,
                        "change_type": "deleted",
                        "tool": tool_name,
                        "original_content": original_content,
                    })

                if tool_call_count >= self._max_tool_calls:
                    break

            # DeepSeek 要求所有 tool_call_id 都有对应结果
            for tc in msg.tool_calls:
                if tc.id not in executed_tool_ids:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Tool execution skipped (limit reached).",
                        }
                    )

            if duplicate_failure_stopped or capability_violation_stopped:
                break

            if tool_call_count >= self._max_tool_calls:
                break

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ExecutionNode (direct) completed in %.1fms (%d tool calls)",
            total_duration,
            tool_call_count,
        )

        return {
            "execution_log": execution_log,
            "errors": errors,
            "trajectory_steps": trajectory_steps,
            "accumulated_changes": accumulated_changes,
            "llm_call_count": state.llm_call_count,
            "estimated_tokens": state.estimated_tokens,
            "steering_instructions": list(state.steering_instructions),
            "memory_hits": list(state.memory_hits),
            "memory_recalled": state.memory_recalled,
            "memory_context": state.memory_context,
        }

    # ── 语法检查 ──────────────────────────────────────────────

    async def _run_syntax_check(
        self,
        file_path: str,
        project_root: str,
        messages: list[dict[str, Any]],
        execution_log: list[dict],
        retry_count: int,
        step_id: int | None = None,
    ) -> int:
        """对写入的文件运行语法检查，若失败则重试。"""
        from pathlib import Path

        full_path = file_path
        if not os.path.isabs(file_path):
            full_path = str(Path(project_root) / file_path) if project_root else file_path

        validation_result = await self._validation_gateway.run_syntax_check(full_path)

        log_entry: dict[str, Any] = {
            "type": "syntax_check",
            "file_path": file_path,
            "step_id": step_id,
            "passed": validation_result.passed,
            "errors": [
                {"line": e.line, "column": e.column, "message": e.message}
                for e in validation_result.errors
            ],
            "duration_ms": validation_result.duration_ms,
            "timestamp": time.time(),
        }
        execution_log.append(log_entry)

        if not validation_result.passed and retry_count < self._max_retries:
            retry_count += 1
            error_summary = "; ".join(
                f"L{e.line}:{e.column} {e.message}" for e in validation_result.errors
            )
            logger.info(
                "Syntax check failed (attempt %d/%d): %s",
                retry_count,
                self._max_retries,
                error_summary,
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The file '{file_path}' has syntax errors "
                        f"(attempt {retry_count}/{self._max_retries}):\n"
                        f"{error_summary}\n\n"
                        "Please fix the syntax errors and try again."
                    ),
                }
            )

        return retry_count

    # ── 辅助方法 ──────────────────────────────────────────────

    def _build_system_prompt(
        self,
        state: AgentState,
        tool_definitions: list[Any],
        memory_section: str = "",
    ) -> str:
        """构建系统提示词。

        Args:
            state: Agent 状态
            tool_definitions: 工具定义列表
            memory_section: Phase 6.5 记忆层文本，空字符串时跳过
        """
        parts = [
            "You are a coding agent. Your task is to help the user with their coding request.",
            "",
            "You have access to the following tools. Use them to read and write files as needed.",
            "",
            "## Conversation turn semantics",
            "- Earlier user and assistant messages are context, not tasks to repeat",
            "- The latest user message is the only current request; answer that delta directly",
            "- Never copy or paraphrase the previous assistant answer as proof of new work",
            "- Requests to run, retry, check, or verify something require fresh tool evidence from this turn",
            "",
        ]

        if state.context:
            parts.append(f"## Context\n{state.context}\n")

        if state.benchmark_instance_id:
            fail_to_pass = "\n".join(
                f"- {test}" for test in state.benchmark_fail_to_pass[:50]
            ) or "- (not provided)"
            pass_to_pass = "\n".join(
                f"- {test}" for test in state.benchmark_pass_to_pass[:50]
            ) or "- (not provided)"
            parts.append(
                "## Benchmark Contract\n"
                f"Instance: {state.benchmark_instance_id}\n"
                "Do not finish until you have produced a real file change. "
                "Final validation must execute the repository tests.\n"
                f"FAIL_TO_PASS tests:\n{fail_to_pass}\n"
                f"PASS_TO_PASS tests:\n{pass_to_pass}\n"
                "The official SWE-bench harness is the sole source of the final score.\n"
            )

        if tool_definitions:
            parts.append("## Available Tools")
            for td in tool_definitions:
                parts.append(f"- {td.name}: {td.description}")
            parts.append("")

        # Phase 6.5: Memory 层注入
        if memory_section:
            parts.append(memory_section)
            parts.append("")

        parts.extend(
            [
                "## Guidelines",
                "- Every file_path/path/cwd argument is relative to the workspace root",
                "- Never prefix paths with the workspace directory name or use absolute paths",
                "- run_terminal already starts at the workspace root; never use cd, /workspace discovery, command substitution $(), or backticks",
                "- Use tools step by step to accomplish the task",
                "- For benchmark work, use search_code/read_file for discovery; grep or listing commands are not test evidence",
                "- After writing a file, the system will check syntax automatically",
                "- If syntax errors are reported, fix them promptly",
                "- For frontend or browser UI work, leave a directly renderable index.html entrypoint; use relative asset URLs so the live workspace preview can render HTML/CSS/JS",
                "- You can make multiple tool calls in sequence",
                "- When you are done, respond with a summary of what you did",
            ]
        )

        return "\n".join(parts)

    # ── Phase 5.5: LLM 成本控制 ────────────────────────────────

    async def _call_llm_with_limit(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> Any | None:
        """调用 LLM 并检查成本限制。

        Args:
            state: Agent 状态（读取/更新 llm_call_count）
            messages: 对话消息列表
            tools: OpenAI 兼容 tools 格式
            tool_choice: tool_choice 参数

        Returns:
            LLM 响应对象，或 None（超出限制时，同时设置 human_review_required）
        """
        await self._apply_runtime_steering(state, messages)
        max_calls = codeagent_config.get_max_llm_calls_per_task()
        current_count = state.llm_call_count

        if current_count >= max_calls:
            state.human_review_required = True
            state.review_type = "cost_limit_reached"
            state.review_request = {
                "review_type": "cost_limit_reached",
                "title": "LLM 调用次数已达上限",
                "details": {
                    "current_count": current_count,
                    "max_calls": max_calls,
                },
                "options": ["approve", "abort"],
            }
            return None

        max_tokens = codeagent_config.get_max_tokens_per_task()
        budget = preflight_model_call(
            consumed_tokens=state.estimated_tokens,
            max_task_tokens=max_tokens,
            max_completion_tokens=(
                codeagent_config.get_max_completion_tokens_per_call()
            ),
            messages=messages,
            tools=tools,
        )
        if not budget.allowed:
            state.human_review_required = True
            state.review_type = "token_budget_exhausted"
            state.review_request = {
                "review_type": "token_budget_exhausted",
                "title": "Token budget exhausted",
                "details": {
                    "estimated_tokens": state.estimated_tokens,
                    "max_tokens": max_tokens,
                    "remaining_tokens": budget.remaining_tokens,
                    "estimated_input_tokens": budget.estimated_input_tokens,
                    "minimum_completion_tokens": budget.minimum_completion_tokens,
                    "reason": budget.reason,
                },
                "options": ["approve", "abort"],
            }
            return None

        response = await self._llm(
            model=self._model_name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=budget.max_completion_tokens,
            model_role="execution",
            routing_context={
                "request": state.user_request,
                "retry_count": state.retry_count,
                "validation_failures": sum(
                    1 for item in state.validation_results
                    if not getattr(item, "success", False)
                ),
                "affected_file_count": len({
                    str(item.get("file_path") or item.get("path") or "")
                    for item in state.accumulated_changes
                    if isinstance(item, dict)
                }),
                "high_risk": any(
                    getattr(step, "risk", "low") == "high"
                    for step in (state.plan or [])
                ),
            },
        )

        state.llm_call_count = current_count + 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            state.estimated_tokens += int(
                getattr(usage, "total_tokens", 0)
                or (getattr(usage, "prompt_tokens", 0) + getattr(usage, "completion_tokens", 0))
            )

        return response

    async def _apply_runtime_steering(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
    ) -> None:
        """Consume queued user instructions at an LLM/tool-loop boundary."""
        if self._steering_provider is None:
            return
        try:
            pending = self._steering_provider()
            if inspect.isawaitable(pending):
                pending = await pending
        except Exception as exc:
            logger.warning("Steering poll degraded: %s", exc)
            return
        if isinstance(pending, str):
            pending = [pending]
        if not isinstance(pending, list):
            return
        for raw in pending[:20]:
            instruction = str(raw).strip()[:4000]
            if not instruction:
                continue
            state.steering_instructions.append(instruction)
            messages.append({
                "role": "user",
                "content": (
                    "## Runtime steering update\n"
                    "The user added this instruction while the task was running. "
                    "Apply it from the next safe action, and let it supersede conflicting "
                    f"earlier details without discarding completed valid work:\n{instruction}"
                ),
            })
            self._report_progress({
                "type": "steering_applied",
                "summary": instruction[:200],
                "data": {"instruction_index": len(state.steering_instructions)},
            })

    def _to_openai_tools(self, tool_definitions: list[Any]) -> list[dict[str, Any]]:
        """将 ToolDefinition 转换为 OpenAI-compatible tools 格式。"""
        tools = []
        for td in tool_definitions:
            tool = {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters_schema,
                },
            }
            tools.append(tool)
        return tools

    def _parse_tool_args(self, arguments: str) -> dict[str, Any]:
        """解析工具参数字符串（JSON）。"""
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse tool arguments: %s", arguments)
            return {}

    def _available_tools(self, state: AgentState, action: str | None = None) -> list[Any]:
        """Return tool definitions available to the model.

        When a plan-step ``action`` is given, the schema surface is pruned to the
        action's allowed set (``_ALLOWED_TOOLS_BY_ACTION``) plus any MCP tools.
        The run manifest, action policy, and Skill allowlist are independent
        upper bounds. Their intersection is exposed so a Skill cannot restore
        a capability already removed by intent/policy selection.
        """
        tools = self._tool_gateway.list_tools()
        if action and action not in _ALLOWED_TOOLS_BY_ACTION:
            logger.warning("Refusing tool exposure for unknown plan action: %s", action)
            return []
        manifest_scoped = "selected_names" in state.tool_manifest
        selected_names = set(state.tool_manifest.get("selected_names", []))
        pruned_names = (
            _ALLOWED_TOOLS_BY_ACTION.get(action, set()) if action else None
        )
        skill_allowed = set(state.allowed_tools)
        skill_allows_mcp = "mcp:*" in skill_allowed

        result: list[Any] = []
        for tool in tools:
            if manifest_scoped and tool.name not in selected_names:
                continue
            if (
                not tool.name.startswith("mcp__")
                and pruned_names is not None
                and tool.name not in pruned_names
            ):
                continue
            if skill_allowed and not (
                tool.name in skill_allowed
                or (skill_allows_mcp and tool.name.startswith("mcp__"))
            ):
                continue
            result.append(tool)
        return result

    @staticmethod
    def _normalize_workspace_args(
        state: AgentState,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Strip a duplicated workspace basename from model-provided paths."""
        root_name = Path(state.project_root).resolve().name
        normalized = dict(arguments)
        for key in ("file_path", "path", "cwd"):
            value = normalized.get(key)
            if not isinstance(value, str):
                continue
            portable = value.replace("\\", "/").lstrip("./")
            if portable == root_name:
                normalized[key] = "."
            elif portable.startswith(f"{root_name}/"):
                normalized[key] = portable[len(root_name) + 1 :]
        return normalized

    def _report_progress(self, progress: dict[str, Any]) -> None:
        """上报进度（如果设置了 callback）。"""
        if self._progress_callback:
            try:
                self._progress_callback(progress)
            except Exception as exc:
                logger.warning("Progress callback failed: %s", exc)

    # ── Phase 6.5: 记忆检索 ──────────────────────────────────

    async def _assemble_memory_section(self, state: AgentState) -> str:
        """检索相关记忆并组装为 Memory 层文本。

        使用 state.user_request 作为检索查询。

        Args:
            state: Agent 状态

        Returns:
            "## Relevant Memories\n{memory_xml}" 或 ""
        """
        if state.memory_mode == "off" or not self._memory_gateway:
            return ""
        if state.memory_recalled:
            return state.memory_context

        try:
            token_budget = codeagent_config.get_memory_token_budget()
            memory_xml = await self._memory_gateway.recall(
                query=state.user_request,
                token_budget=token_budget,
            )
            state.memory_recalled = True
            if not memory_xml:
                state.memory_context = ""
                return ""
            from codeagent.memory.audit import parse_memory_hits

            hits = parse_memory_hits(memory_xml, token_budget)
            existing = {str(hit.get("name")) for hit in state.memory_hits}
            state.memory_hits.extend(hit for hit in hits if str(hit.get("name")) not in existing)
            state.memory_context = f"## Relevant Memories\n\n{memory_xml}"
            self._report_progress(
                {
                    "type": "memory_recalled",
                    "visibility": "internal",
                    "summary": f"Recalled {len(hits)} relevant memory item(s)",
                    "data": {"hits": hits, "token_budget": token_budget},
                }
            )
            return state.memory_context
        except Exception as exc:
            logger.warning("Memory recall failed in ExecutionNode (non-blocking): %s", exc)
            return ""

    # ── Rollback 支持 ──────────────────────────────────────────

    def _save_original_content_before_tool(
        self, tool_name: str, tool_args: dict, project_root: str | None = None
    ) -> tuple[str, Optional[str]]:
        """在 write_file/delete_file 前保存原始内容。

        Args:
            tool_name: 工具名称
            tool_args: 工具参数字典

        Returns:
            tuple[file_path, original_content]:
                file_path: 目标文件路径
                original_content: 原始内容（文件不存在时为 None）
        """
        file_path = tool_args.get("file_path", "")
        if not file_path or tool_name not in ("write_file", "apply_patch", "delete_file"):
            return ("", None)

        # 尝试读取文件原始内容
        try:
            path = Path(file_path)
            if project_root is not None and not path.is_absolute():
                path = Path(project_root) / path
            if path.is_file():
                original_content = path.read_text(encoding="utf-8")
            else:
                original_content = None
        except (OSError, PermissionError):
            original_content = None

        return (file_path, original_content)
