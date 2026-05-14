"""ExecutionNode — 执行节点（Phase 3.5 升级版：偏离检测 + 目标追踪）。

核心功能：
1. Plan-aware: 按 PlanStep 逐步执行（state.plan 存在时）
2. Phase 1a 兼容: 直连 LLM 调用模式仍支持
3. 每次 write_file 后立即调用语法检查
4. 语法错误 → 反馈 LLM 重新生成修复代码（最多 3 次）
5. 单步骤超过 10 次 tool_calls → 强制结束
6. 进度上报：通过 callback 实时报告执行进度
7. TaskFocus: 偏离检测 + 目标追踪 + System Prompt 注入
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

# 每个 action 类型的合理工具列表
_ALLOWED_TOOLS_BY_ACTION: dict[str, set[str]] = {
    "create": {"write_file", "read_file"},
    "modify": {"read_file", "write_file", "search_code"},
    "delete": {"read_file", "delete_file"},
    "read": {"read_file", "search_code", "glob"},
    "command": {"run_terminal", "read_file"},
}

# 偏离检测阈值
_DEVIATION_WARN_LIMIT = 1   # 首次偏离 → 警告
_DEVIATION_UPGRADE_LIMIT = 2  # 连续 2 次 → 升级警告
_DEVIATION_HUMAN_LIMIT = 3    # 连续 3 次 → 请求人工审核


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
        execution_log: list[dict] = []
        errors: list[str] = list(state.errors)
        accumulated_changes: list[dict] = list(state.accumulated_changes)
        current_step_index = state.current_step_index
        deviation_count = state.deviation_count
        deviation_detected = state.deviation_detected
        human_review_required = state.human_review_required
        review_request = state.review_request

        tool_definitions = self._tool_gateway.list_tools()
        remaining_steps = state.plan[current_step_index:]

        # 构建完成/剩余步骤摘要
        completed_descriptions = [
            s.description for s in state.plan[:current_step_index]
        ]
        completed_steps_summary = "; ".join(completed_descriptions) if completed_descriptions else ""

        for i, step in enumerate(remaining_steps):
            # 如果偏离触发 Human Review，停止执行
            if human_review_required:
                break

            self._report_progress({
                "step_id": step.step_id,
                "status": "running",
                "file_path": step.target_file,
                "operation": step.action,
                "message": step.description,
            })

            # 计算剩余步骤描述（不含当前步骤）
            remaining_descriptions = [
                f"[{s.step_id}] {s.description}" for s in remaining_steps[i + 1:]
            ]

            # 构建步骤提示词
            step_prompt = self._build_step_prompt(state, step, tool_definitions)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": step_prompt},
            ]

            # 执行该步骤的工具调用循环（含偏离检测）
            step_result, step_deviation_count, step_human_review, step_review_request = (
                await self._execute_step_tool_loop(
                    step=step,
                    messages=messages,
                    tool_definitions=tool_definitions,
                    execution_log=execution_log,
                    errors=errors,
                    accumulated_changes=accumulated_changes,
                    deviation_count=deviation_count,
                    remaining_descriptions=remaining_descriptions,
                )
            )

            if step_result is not None:
                accumulated_changes = step_result

            deviation_count = step_deviation_count
            if step_human_review:
                human_review_required = True
                review_request = step_review_request

            current_step_index += 1

            # 更新完成摘要
            completed_descriptions.append(step.description)
            completed_steps_summary = "; ".join(completed_descriptions)

            self._report_progress({
                "step_id": step.step_id,
                "status": "completed",
                "file_path": step.target_file,
                "operation": step.action,
                "message": f"Step {step.step_id} completed",
            })

            # 如果偏离触发 Human Review，停止执行
            if human_review_required:
                break

        # 计算剩余任务描述
        remaining_descriptions = [
            f"[{s.step_id}] {s.description}"
            for s in (state.plan or [])[current_step_index:]
        ]

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
            "completed_steps_summary": completed_steps_summary,
            "tasks_remaining": remaining_descriptions,
            "deviation_detected": deviation_detected,
            "deviation_count": deviation_count,
            "human_review_required": human_review_required,
            "review_request": review_request,
        }

    def _build_step_prompt(
        self,
        state: AgentState,
        step: PlanStep,
        tool_definitions: list[Any],
    ) -> str:
        """为单个 PlanStep 构建提示词（含 TaskFocus 上下文注入）。"""
        total_steps = len(state.plan or [])
        parts: list[str] = [
            "You are a coding agent executing a specific step of a plan.",
            "",
            f"## Current Step ({step.step_id}/{total_steps})",
            f"Description: {step.description}",
            f"Action: {step.action}",
            f"Target: {step.target_file or 'N/A'}",
            "",
        ]

        # ── Phase 3.5 TaskFocus: 注入当前任务状态 ──────────────
        if state.original_goal_summary:
            parts.append(
                "## 当前任务状态\n"
                f"原始目标: {state.original_goal_summary}\n"
            )
        if state.completed_steps_summary:
            parts.append(f"进度: {state.completed_steps_summary}")
        if state.tasks_remaining:
            parts.append(
                "剩余步骤:\n" + "\n".join(
                    f"- {t}" for t in state.tasks_remaining
                )
            )
        if state.original_goal_summary or state.completed_steps_summary or state.tasks_remaining:
            parts.append(
                "请严格围绕上述目标执行，不要偏离到未规划的方向。\n"
            )

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
        deviation_count: int = 0,
        remaining_descriptions: list[str] | None = None,
    ) -> tuple[list[dict] | None, int, bool, dict | None]:
        """执行单个步骤的 tool calling 循环（含偏离检测）。

        Returns:
            tuple[accumulated_changes, deviation_count, human_review_required, review_request]
        """
        tool_call_count = 0
        syntax_retries = 0
        pending_syntax_check: str | None = None
        openai_tools = self._to_openai_tools(tool_definitions)
        human_review_required = False
        review_request: dict | None = None

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
                    errors.append(
                        f"Exceeded max tool calls per step ({self._max_tool_calls})"
                    )
                    break

                tool_call_count += 1
                tool_name = tool_call.function.name
                tool_args = self._parse_tool_args(tool_call.function.arguments)

                # ── Phase 3.5 偏离检测 ──────────────────────────
                is_deviation = self._check_deviation(step, tool_name, tool_args)
                if is_deviation:
                    deviation_count += 1
                    deviation_detected = True

                    execution_log.append({
                        "type": "deviation_detected",
                        "step_id": step.step_id,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "deviation_count": deviation_count,
                        "timestamp": time.time(),
                    })

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
                        execution_log.append({
                            "type": "human_review_required",
                            "reason": "repeated_deviation",
                            "deviation_count": deviation_count,
                            "timestamp": time.time(),
                        })
                        # DeepSeek 要求所有 tool_call_id 都有对应结果
                        for tc in msg.tool_calls:
                            if tc.id not in executed_tool_ids:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": "Tool execution skipped due to deviation limit requiring human review.",
                                })

                        # 先 flush 已收集的偏离警告，再添加最终警告
                        for w in pending_deviation_warnings:
                            messages.append(w)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"警告：检测到连续 {deviation_count} 次偏离计划的行为。"
                                f"执行已暂停，需要人工审核。"
                            ),
                        })
                        return accumulated_changes, deviation_count, human_review_required, review_request
                    elif deviation_count >= _DEVIATION_UPGRADE_LIMIT:
                        # 连续 2 次 → 升级警告（收集，在 tool 结果后追加）
                        pending_deviation_warnings.append({
                            "role": "user",
                            "content": (
                                f"警告（{deviation_count}/{_DEVIATION_HUMAN_LIMIT}）："
                                f"你正在偏离当前步骤的规划范围。"
                                f"当前步骤目标: {step.description}。"
                                f"请严格围绕当前步骤执行。"
                            ),
                        })
                        continue
                    else:
                        # 首次偏离 → 警告（收集，在 tool 结果后追加）
                        pending_deviation_warnings.append({
                            "role": "user",
                            "content": (
                                f"注意：工具 '{tool_name}' 的调用似乎偏离了当前步骤的规划。"
                                f"当前步骤目标: {step.description}。"
                                f"请确保工具调用在步骤范围内。"
                            ),
                        })
                        continue
                else:
                    deviation_detected = False

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
                executed_tool_ids.append(tool_call.id)

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

            # DeepSeek 要求所有 tool_call_id 都有对应结果（必须紧跟在 assistant msg 之后）
            for tc in msg.tool_calls:
                if tc.id not in executed_tool_ids:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Tool execution skipped (deviation or limit reached).",
                    })

            # 在 tool 结果之后追加偏离警告（确保 assistant→tool→user 的顺序）
            for w in pending_deviation_warnings:
                messages.append(w)

            if tool_call_count >= self._max_tool_calls:
                break

        return accumulated_changes, deviation_count, human_review_required, review_request

    # ── Phase 3.5 偏离检测 ────────────────────────────────────

    def _check_deviation(
        self, step: PlanStep, tool_name: str, tool_args: dict
    ) -> bool:
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
        if tool_name == "write_file":
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
        if allowed and tool_name not in allowed:
            return True

        return False

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

            executed_tool_ids: list[str] = []

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
                executed_tool_ids.append(tool_call.id)

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

            # DeepSeek 要求所有 tool_call_id 都有对应结果
            for tc in msg.tool_calls:
                if tc.id not in executed_tool_ids:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Tool execution skipped (limit reached).",
                    })

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
