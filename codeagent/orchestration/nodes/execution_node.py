"""ExecutionNode — 执行节点（Phase 1b 升级版：支持按计划逐步执行）。

核心功能：
1. Plan-aware: 按 PlanStep 逐步执行（state.plan 存在时）
2. Phase 1a 兼容: 直连 LLM 调用模式仍支持
3. 每次 write_file 后立即调用语法检查
4. 语法错误 → 反馈 LLM 重新生成修复代码（最多 3 次）
5. 单步骤超过 10 次 tool_calls → 强制结束
6. 进度上报：通过 callback 实时报告执行进度
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.orchestration.state import AgentState, PlanStep

logger = logging.getLogger(__name__)

# 最大连续 tool_calls 次数
_MAX_TOOL_CALLS = 10

# 单步最大重试次数
_MAX_RETRIES = 3


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
    ) -> None:
        """初始化 ExecutionNode。

        Args:
            llm: LLM 调用函数（如 litellm.completion）
            tool_gateway: 工具系统 Gateway
            validation_gateway: 验证 Gateway
            model_name: 模型名称
            max_tool_calls: 单次执行最大 tool_calls 次数
            max_retries: 单步语法错误最大重试次数
            progress_callback: 进度回调函数，接收 dict: {step_id, status, ...}
        """
        self._llm = llm
        self._tool_gateway = tool_gateway
        self._validation_gateway = validation_gateway
        self._model_name = model_name
        self._max_tool_calls = max_tool_calls
        self._max_retries = max_retries
        self._progress_callback = progress_callback

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """执行主入口。

        如果 state.plan 存在且包含未执行步骤，按计划逐步执行；
        否则走 Phase 1a 直连模式。

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新的状态字段（execution_log, errors, plan 等）
        """
        if state.plan and state.current_step_index < len(state.plan):
            return await self._execute_with_plan(state)
        return await self._execute_direct(state)

    # ── Plan-aware 执行 ──────────────────────────────────────

    async def _execute_with_plan(self, state: AgentState) -> dict[str, Any]:
        """按计划逐步执行。

        遍历 state.plan[state.current_step_index:]，对每个步骤：
        1. 构建步骤提示词 → 调用 LLM
        2. 执行工具调用循环
        3. 语法检查
        4. 记录变更
        5. 上报进度
        """
        start_time = time.monotonic()
        execution_log: list[dict] = []
        errors: list[str] = list(state.errors)
        accumulated_changes: list[dict] = list(state.accumulated_changes)
        current_step_index = state.current_step_index

        tool_definitions = self._tool_gateway.list_tools()
        remaining_steps = state.plan[current_step_index:]

        for step in remaining_steps:
            self._report_progress({
                "step_id": step.step_id,
                "status": "running",
                "file_path": step.target_file,
                "operation": step.action,
                "message": step.description,
            })

            # 构建步骤提示词
            step_prompt = self._build_step_prompt(state, step, tool_definitions)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": step_prompt},
            ]

            # 执行该步骤的工具调用循环
            step_result = await self._execute_step_tool_loop(
                step=step,
                messages=messages,
                tool_definitions=tool_definitions,
                execution_log=execution_log,
                errors=errors,
                accumulated_changes=accumulated_changes,
            )

            if step_result:
                accumulated_changes = step_result

            current_step_index += 1

            self._report_progress({
                "step_id": step.step_id,
                "status": "completed",
                "file_path": step.target_file,
                "operation": step.action,
                "message": f"Step {step.step_id} completed",
            })

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ExecutionNode (plan) completed in %.1fms (%d steps)",
            total_duration, current_step_index - state.current_step_index,
        )

        return {
            "execution_log": execution_log,
            "errors": errors,
            "current_step_index": current_step_index,
            "accumulated_changes": accumulated_changes,
        }

    def _build_step_prompt(
        self,
        state: AgentState,
        step: PlanStep,
        tool_definitions: list[Any],
    ) -> str:
        """为单个 PlanStep 构建提示词。"""
        parts: list[str] = [
            "You are a coding agent executing a specific step of a plan.",
            "",
            f"## Current Step ({step.step_id}/{len(state.plan or [])})",
            f"Description: {step.description}",
            f"Action: {step.action}",
            f"Target: {step.target_file or 'N/A'}",
            "",
        ]

        if state.context:
            parts.append(f"## Project Context\n{state.context}\n")

        parts.append("## Available Tools")
        for td in tool_definitions:
            parts.append(f"- {td.name}: {td.description}")
        parts.append("")

        parts.extend([
            "## Guidelines",
            "- Use tools to accomplish this single step",
            "  (not the entire plan)",
            "- After writing a file, syntax will be checked automatically",
            "- If syntax errors are reported, fix them promptly",
            "- When this step is done, respond with a summary",
        ])

        return "\n".join(parts)

    async def _execute_step_tool_loop(
        self,
        step: PlanStep,
        messages: list[dict[str, Any]],
        tool_definitions: list[Any],
        execution_log: list[dict],
        errors: list[str],
        accumulated_changes: list[dict],
    ) -> list[dict] | None:
        """执行单个步骤的 tool calling 循环。"""
        tool_call_count = 0
        syntax_retries = 0
        pending_syntax_check: str | None = None
        openai_tools = self._to_openai_tools(tool_definitions)

        while tool_call_count < self._max_tool_calls:
            try:
                response = await self._llm(
                    model=self._model_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                errors.append(f"LLM call failed: {e}")
                break

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                if msg.content:
                    execution_log.append({
                        "type": "llm_response",
                        "content": msg.content,
                        "step_id": step.step_id,
                        "timestamp": time.time(),
                    })
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
                "content": msg.content,
                "tool_calls": assistant_tool_calls,
            }
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                assistant_msg["reasoning_content"] = msg.reasoning_content
            messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                if tool_call_count >= self._max_tool_calls:
                    errors.append(
                        f"Exceeded max tool calls per step ({self._max_tool_calls})"
                    )
                    break

                tool_call_count += 1
                tool_name = tool_call.function.name
                tool_args = self._parse_tool_args(tool_call.function.arguments)

                logger.info(
                    "Step %d tool #%d: %s(%s)",
                    step.step_id, tool_call_count, tool_name, tool_args,
                )

                tool_result = await self._tool_gateway.execute_tool(
                    tool_name, tool_args
                )
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
                else:
                    log_entry["error"] = tool_result.error_message
                    logger.warning(
                        "Step %d tool '%s' failed: %s",
                        step.step_id, tool_name, tool_result.error_message,
                    )

                execution_log.append(log_entry)

                # 添加工具结果到消息
                result_content = (
                    json.dumps(tool_result.data, ensure_ascii=False)
                    if tool_result.success
                    else f"Error: {tool_result.error_message}"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content,
                })

                # write_file → track change + syntax check
                if tool_name == "write_file" and tool_result.success:
                    file_path = tool_args.get("file_path", "")
                    if file_path:
                        accumulated_changes.append({
                            "file_path": file_path,
                            "step_id": step.step_id,
                            "action": tool_args.get("mode", "modify"),
                            "timestamp": time.time(),
                        })
                        syntax_retries = await self._run_syntax_check(
                            file_path=file_path,
                            project_root="",  # will resolve in _run_syntax_check
                            messages=messages,
                            execution_log=execution_log,
                            retry_count=syntax_retries,
                            step_id=step.step_id,
                        )

                if tool_call_count >= self._max_tool_calls:
                    break

            if tool_call_count >= self._max_tool_calls:
                break

        return accumulated_changes

    # ── Phase 1a 直连模式 ────────────────────────────────────

    async def _execute_direct(self, state: AgentState) -> dict[str, Any]:
        """Phase 1a 直连模式：直接通过 LLM tool calling 循环执行。

        copy from Phase 1a 的 __call__ 实现。
        """
        start_time = time.monotonic()
        execution_log: list[dict] = []
        errors: list[str] = list(state.errors)

        tool_definitions = self._tool_gateway.list_tools()
        system_prompt = self._build_system_prompt(
            state, tool_definitions
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state.user_request},
        ]

        tool_call_count = 0
        syntax_retries = 0
        pending_syntax_check: str | None = None

        while tool_call_count < self._max_tool_calls:
            openai_tools = self._to_openai_tools(tool_definitions)

            try:
                response = await self._llm(
                    model=self._model_name,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                errors.append(f"LLM call failed: {e}")
                break

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                if msg.content:
                    execution_log.append({
                        "type": "llm_response",
                        "content": msg.content,
                        "timestamp": time.time(),
                    })
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

            for tool_call in msg.tool_calls:
                if tool_call_count >= self._max_tool_calls:
                    errors.append(
                        f"Exceeded max tool calls ({self._max_tool_calls})"
                    )
                    break

                tool_call_count += 1
                tool_name = tool_call.function.name
                tool_args = self._parse_tool_args(tool_call.function.arguments)

                logger.info(
                    "Tool call #%d: %s(%s)",
                    tool_call_count, tool_name, tool_args,
                )

                tool_result = await self._tool_gateway.execute_tool(
                    tool_name, tool_args
                )
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
                        tool_name, elapsed,
                    )
                else:
                    log_entry["error"] = tool_result.error_message
                    logger.warning(
                        "Tool '%s' failed: %s",
                        tool_name, tool_result.error_message,
                    )

                execution_log.append(log_entry)

                result_content = (
                    json.dumps(tool_result.data, ensure_ascii=False)
                    if tool_result.success
                    else f"Error: {tool_result.error_message}"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content,
                })

                if tool_name == "write_file" and tool_result.success:
                    file_path = tool_args.get("file_path", "")
                    if file_path:
                        syntax_retries = await self._run_syntax_check(
                            file_path=file_path,
                            project_root=state.project_root,
                            messages=messages,
                            execution_log=execution_log,
                            retry_count=syntax_retries,
                            step_id=None,
                        )

                if tool_call_count >= self._max_tool_calls:
                    break

            if tool_call_count >= self._max_tool_calls:
                break

        total_duration = (time.monotonic() - start_time) * 1000
        logger.info(
            "ExecutionNode (direct) completed in %.1fms (%d tool calls)",
            total_duration, tool_call_count,
        )

        return {
            "execution_log": execution_log,
            "errors": errors,
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
        import os
        from pathlib import Path

        full_path = file_path
        if not os.path.isabs(file_path):
            full_path = str(Path(project_root) / file_path) if project_root else file_path

        validation_result = await self._validation_gateway.run_syntax_check(
            full_path
        )

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
                f"L{e.line}:{e.column} {e.message}"
                for e in validation_result.errors
            )
            logger.info(
                "Syntax check failed (attempt %d/%d): %s",
                retry_count, self._max_retries, error_summary,
            )

            messages.append({
                "role": "user",
                "content": (
                    f"The file '{file_path}' has syntax errors "
                    f"(attempt {retry_count}/{self._max_retries}):\n"
                    f"{error_summary}\n\n"
                    "Please fix the syntax errors and try again."
                ),
            })

        return retry_count

    # ── 辅助方法 ──────────────────────────────────────────────

    def _build_system_prompt(
        self,
        state: AgentState,
        tool_definitions: list[Any],
    ) -> str:
        """构建系统提示词。"""
        parts = [
            "You are a coding agent. Your task is to help the user with their coding request.",
            "",
            "You have access to the following tools. Use them to read and write files as needed.",
            "",
        ]

        if state.context:
            parts.append(f"## Context\n{state.context}\n")

        if tool_definitions:
            parts.append("## Available Tools")
            for td in tool_definitions:
                parts.append(f"- {td.name}: {td.description}")
            parts.append("")

        parts.extend([
            "## Guidelines",
            "- Use tools step by step to accomplish the task",
            "- After writing a file, the system will check syntax automatically",
            "- If syntax errors are reported, fix them promptly",
            "- You can make multiple tool calls in sequence",
            "- When you are done, respond with a summary of what you did",
        ])

        return "\n".join(parts)

    def _to_openai_tools(
        self, tool_definitions: list[Any]
    ) -> list[dict[str, Any]]:
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

    def _report_progress(self, progress: dict[str, Any]) -> None:
        """上报进度（如果设置了 callback）。"""
        if self._progress_callback:
            try:
                self._progress_callback(progress)
            except Exception as exc:
                logger.warning("Progress callback failed: %s", exc)
