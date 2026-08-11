"""Planning Node — 规划节点。

分析用户需求和项目上下文，生成结构化多步骤执行计划。
支持 JSON 解析重试、Schema 校验、冲突检测。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from codeagent import config as codeagent_config
from codeagent.context_engine.evolution.strategy_applier import StrategyApplier
from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.orchestration.budget import preflight_model_call
from codeagent.orchestration.state import AgentState, PlanStep
from codeagent.tracing import trace_node

logger = logging.getLogger(__name__)

# 最大重试次数（JSON 解析失败时的重试）
_MAX_RETRIES = 2

# ── 系统提示词模板 ─────────────────────────────────────────

_SYSTEM_PROMPT = """You are a planning agent for a coding assistant. Analyze the user's request and the project context, then generate a structured step-by-step execution plan.

Each step should be a JSON object. Respond with ONLY a JSON array of steps, no other text.

All target_file values MUST be relative to the workspace root. Never prefix a target with
the workspace directory name, an absolute path, or "./". For example use "src/app.py".

Keep the plan to the smallest set of steps that can implement and prove the request,
normally 2-5 steps. Combine closely related inspection and modification in one modify
step, followed by targeted validation. Do not add changelog or documentation work unless
the user or repository instructions explicitly require it. A regression test may be part
of the same modification step when it targets the same behavior.

## JSON Schema for each step

{{
  "step_id": int (1-based sequential),
  "description": str (what to do in this step),
  "action": "create" | "modify" | "delete" | "read" | "command",
  "target_file": str | null (file path, null for "command" actions),
  "risk": "low" | "high" (high = modifies/deletes files or runs commands),
  "dependencies": [int] (step_ids that must be completed before this one),
  "acceptance_criteria": str (observable tool or command evidence required to finish)
}}

## Examples

### Example 1: Create a Flask app
User request: "Create a hello world Flask app"
Context: empty project

Output:
{{
  "plan": [
    {{
      "step_id": 1,
      "description": "Create requirements.txt with Flask dependency",
      "action": "create",
      "target_file": "requirements.txt",
      "risk": "low",
      "dependencies": []
    }},
    {{
      "step_id": 2,
      "description": "Create app.py with Flask hello world",
      "action": "create",
      "target_file": "app.py",
      "risk": "low",
      "dependencies": []
    }}
  ],
  "original_goal_summary": "Create a hello world Flask application"
}}

### Example 2: Refactor a module
User request: "Rename main.py to app.py and update imports"
Context: existing project with main.py and utils.py

Output:
{{
  "plan": [
    {{
      "step_id": 1,
      "description": "Read main.py to understand current exports",
      "action": "read",
      "target_file": "main.py",
      "risk": "low",
      "dependencies": []
    }},
    {{
      "step_id": 2,
      "description": "Create app.py with main.py content",
      "action": "create",
      "target_file": "app.py",
      "risk": "low",
      "dependencies": [1]
    }},
    {{
      "step_id": 3,
      "description": "Update utils.py imports to reference app instead of main",
      "action": "modify",
      "target_file": "utils.py",
      "risk": "low",
      "dependencies": [2]
    }},
    {{
      "step_id": 4,
      "description": "Delete old main.py",
      "action": "delete",
      "target_file": "main.py",
      "risk": "high",
      "dependencies": [2]
    }}
  ],
  "original_goal_summary": "Rename main.py to app.py and update all related imports"
}}

## Current Request
User: {user_request}

{context_section}
Respond with a JSON object containing two fields:
1. "plan": a JSON array of steps (following the schema above)
2. "original_goal_summary": a brief 1-2 sentence summary of the user's original goal, used to keep the Agent focused during execution

Respond with ONLY the JSON object, no markdown or explanation."""


class PlanningNode:
    """规划节点 — 生成多步骤执行计划。

    向 LLM 发送规划提示词（含 JSON Schema + few-shot 示例），
    解析 LLM 输出的 JSON 为 PlanStep 列表，执行冲突检测。
    JSON 格式错误时自动重试（最多 2 次）。
    """

    def __init__(
        self,
        llm: Callable[..., Any],
        model_name: str = "deepseek/deepseek-v4-flash",
        max_retries: int = _MAX_RETRIES,
        memory_gateway: Optional[IMemoryGateway] = None,
        strategy_applier: Optional[StrategyApplier] = None,
        progress_callback: Any | None = None,
    ) -> None:
        """初始化 PlanningNode。

        Args:
            llm: LLM 调用函数
            model_name: 模型名称
            max_retries: JSON 解析失败重试次数
            memory_gateway: 记忆系统 Gateway，None 时跳过记忆检索
            strategy_applier: StrategyApplier 实例，None 时不注入策略
        """
        self._llm = llm
        self._model_name = model_name
        self._max_retries = max_retries
        self._memory_gateway = memory_gateway
        self._strategy_applier = strategy_applier
        self._current_strategy_ids: list[str] = []
        self._progress_callback = progress_callback

    @trace_node("planning")
    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """执行规划。

        Args:
            state: 当前 AgentState

        Returns:
            dict: 包含 plan 或 errors 的状态更新
        """
        if state.direct_execution:
            return {
                "plan": [],
                "original_goal_summary": state.user_request,
                "execution_log": [
                    *state.execution_log,
                    {"type": "planning_skipped", "reason": "follow_up_direct_execution"},
                ],
                "llm_call_count": state.llm_call_count,
                "estimated_tokens": state.estimated_tokens,
            }

        if state.benchmark_instance_id:
            # The benchmark workflow is intentionally fixed and does not benefit from
            # spending a model call restating it as JSON. File discovery remains inside
            # the modify step, where the coding model has repository tools and context.
            steps = [
                PlanStep(
                    step_id=1,
                    description=(
                        "Inspect the relevant production code and implement the smallest "
                        "fix for the benchmark issue without modifying tests"
                    ),
                    action="modify",
                    target_file=None,
                    risk="low",
                    dependencies=[],
                    acceptance_criteria=(
                        "A successful write_file or apply_patch call changes production code"
                    ),
                ),
                PlanStep(
                    step_id=2,
                    description="Run the supplied benchmark target tests",
                    action="command",
                    target_file=None,
                    risk="low",
                    dependencies=[1],
                    acceptance_criteria="A successful run_terminal call records test evidence",
                ),
            ]
            return {
                "plan": steps,
                "original_goal_summary": state.user_request,
                "applied_strategy_ids": [],
                "llm_call_count": state.llm_call_count,
                "estimated_tokens": state.estimated_tokens,
                "memory_hits": list(state.memory_hits),
                "memory_recalled": state.memory_recalled,
                "memory_context": state.memory_context,
                "execution_log": [{
                    "type": "plan_generated",
                    "source": "deterministic_benchmark_policy",
                    "steps": len(steps),
                    "conflicts": None,
                    "original_goal_summary": state.user_request,
                    "strategy_ids": None,
                }],
            }

        prompt = self._build_prompt(state)

        # Phase 6.5: 注入相关记忆到 System Prompt
        memory_section = await self._assemble_memory_section(state)
        if memory_section:
            prompt = f"{prompt}\n\n{memory_section}"

        # Phase 7.4: 注入相关策略到 System Prompt（在 Memory 层之后）
        strategy_section = await self._assemble_strategy_section(state)
        if strategy_section:
            prompt = f"{prompt}\n\n{strategy_section}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
        ]

        last_error: str | None = None
        llm_call_count = state.llm_call_count
        estimated_tokens = state.estimated_tokens

        max_attempts = min(self._max_retries + 1, 2) if state.benchmark_instance_id else (
            self._max_retries + 1
        )
        for attempt in range(max_attempts):
            try:
                try:
                    max_task_tokens = codeagent_config.get_max_tokens_per_task()
                    budget = preflight_model_call(
                        consumed_tokens=estimated_tokens,
                        max_task_tokens=max_task_tokens,
                        max_completion_tokens=(
                            codeagent_config.get_max_completion_tokens_per_call()
                        ),
                        messages=messages,
                    )
                    if (
                        llm_call_count >= codeagent_config.get_max_llm_calls_per_task()
                        or not budget.allowed
                    ):
                        return {
                            "errors": [
                                "Plan generation stopped by the task model budget "
                                f"({budget.reason or 'call_limit_reached'}; "
                                f"remaining={budget.remaining_tokens}, "
                                f"estimated_input={budget.estimated_input_tokens})"
                            ],
                            "plan": None,
                            "original_goal_summary": "",
                            "llm_call_count": llm_call_count,
                            "estimated_tokens": estimated_tokens,
                        }
                    response = await self._llm(
                        model=self._model_name,
                        messages=messages,
                        max_tokens=budget.max_completion_tokens,
                        model_role="planning",
                        routing_context={
                            "request": state.user_request,
                            "retry_count": state.retry_count,
                            "validation_failures": sum(
                                1 for item in state.validation_results
                                if not getattr(item, "success", False)
                            ),
                        },
                    )
                    llm_call_count += 1
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        estimated_tokens += int(
                            getattr(usage, "total_tokens", 0)
                            or (
                                getattr(usage, "prompt_tokens", 0)
                                + getattr(usage, "completion_tokens", 0)
                            )
                        )
                except Exception as exc:
                    logger.error("LLM call failed: %s", exc)
                    return {
                        "errors": [
                            f"Plan generation failed: LLM call error: {exc}"
                        ],
                        "plan": None,
                        "original_goal_summary": "",
                        "llm_call_count": llm_call_count,
                        "estimated_tokens": estimated_tokens,
                    }

                message = response.choices[0].message
                content = message.content or ""
                if not content.strip():
                    # Thinking-capable OpenAI-compatible providers sometimes put the
                    # entire structured answer in reasoning_content and leave content
                    # empty. Recover it before spending another paid planning call.
                    content = getattr(message, "reasoning_content", "") or ""

                # 清理可能存在的 markdown 包装
                plan_json = self._clean_json(content)
                parsed = json.loads(plan_json)

                # 支持两种格式：新格式 {"plan": [...], "original_goal_summary": "..."}
                # 和向后兼容的纯数组格式
                original_goal_summary = ""
                if isinstance(parsed, dict):
                    plan_data = parsed.get("plan", parsed)
                    original_goal_summary = parsed.get("original_goal_summary", "")
                    if isinstance(plan_data, dict):
                        plan_data = list(plan_data.values()) if not isinstance(plan_data, list) else []
                elif isinstance(parsed, list):
                    plan_data = parsed
                else:
                    raise ValueError(f"Expected JSON array or object, got {type(parsed).__name__}")

                steps = self._parse_steps(plan_data)

                # Schema 校验
                self._validate_steps(steps)

                # 冲突检测
                conflicts = self._detect_conflicts(steps)
                if conflicts:
                    logger.warning("Plan conflicts detected: %s", conflicts)

                logger.info(
                    "Plan generated: %d steps (attempt %d/%d)",
                    len(steps), attempt + 1, self._max_retries + 1,
                )

                return {
                    "plan": steps,
                    "original_goal_summary": original_goal_summary,
                    "applied_strategy_ids": list(self._current_strategy_ids),
                    "llm_call_count": llm_call_count,
                    "estimated_tokens": estimated_tokens,
                    "memory_hits": list(state.memory_hits),
                    "memory_recalled": state.memory_recalled,
                    "memory_context": state.memory_context,
                    "execution_log": [{
                        "type": "plan_generated",
                        "steps": len(steps),
                        "conflicts": conflicts or None,
                        "original_goal_summary": original_goal_summary or None,
                        "strategy_ids": list(self._current_strategy_ids) or None,
                    }],
                }

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Plan parse failed (attempt %d/%d): %s",
                    attempt + 1, max_attempts, last_error,
                )

                if attempt + 1 < max_attempts:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"JSON parse error: {last_error}\n\n"
                            "Please respond with ONLY a valid JSON object containing "
                            "a plan array and original_goal_summary. "
                            "No markdown, no explanation."
                        ),
                    })

        return {
            "errors": [
                f"Plan generation failed after {max_attempts} "
                f"attempts: {last_error}"
            ],
            "plan": None,
            "original_goal_summary": "",
            "llm_call_count": llm_call_count,
            "estimated_tokens": estimated_tokens,
        }

    def _build_prompt(self, state: AgentState) -> str:
        """构建规划提示词。"""
        context_lines: list[str] = []

        if state.file_tree:
            if isinstance(state.file_tree, dict):
                tree_text = self._format_tree_for_prompt(state.file_tree, is_root=True)
                context_lines.append(f"## Project Structure\n{tree_text}\n")

        if state.semantic_context:
            context_lines.append(
                f"## Context\n{state.semantic_context}\n"
            )

        if state.benchmark_instance_id:
            context_lines.append(
                "## Benchmark Planning Policy\n"
                f"Instance: {state.benchmark_instance_id}\n"
                "Prefer exactly two steps: (1) inspect as needed and modify the "
                "production target in one modify step; (2) run the supplied target "
                "tests in one command step. Do not add or modify tests unless the "
                "issue cannot be fixed without doing so. Keep exploration inside the "
                "modify step instead of creating a separate read step.\n"
            )

        context_section = "\n".join(context_lines) if context_lines else ""

        return _SYSTEM_PROMPT.format(
            user_request=state.user_request,
            context_section=context_section,
        )

    # ── Phase 6.5: 记忆检索 ─────────────────────────────────

    async def _assemble_memory_section(self, state: AgentState) -> str:
        """检索相关记忆并组装为 System Prompt 的 Memory 层。

        Args:
            state: Agent 状态（使用 user_request 作为查询）

        Returns:
            "## Relevant Memories\n{memory_xml}" 或 ""
        """
        if state.memory_mode == "off" or not self._memory_gateway:
            return ""
        if state.memory_recalled:
            return state.memory_context

        try:
            from codeagent import config
            token_budget = config.get_memory_token_budget()
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
            state.memory_hits = hits
            state.memory_context = f"## Relevant Memories\n\n{memory_xml}"
            self._report_progress({
                "type": "memory_recalled",
                "visibility": "internal",
                "summary": f"Recalled {len(hits)} relevant memory item(s)",
                "data": {"hits": hits, "token_budget": token_budget},
            })
            return state.memory_context
        except Exception as exc:
            logger.warning("Memory recall failed in PlanningNode (non-blocking): %s", exc)
            return ""

    def _report_progress(self, event: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(event)
        except Exception as exc:
            logger.warning("Planning progress callback failed: %s", exc)

    # ── Phase 7.4: 策略注入 ─────────────────────────────────

    async def _assemble_strategy_section(self, state: AgentState) -> str:
        """检索相关策略并组装为 System Prompt 的 Strategy 层。

        在 Memory 层之后注入，格式为 XML 块。
        记录已应用的策略 ID 供后续效果追踪。

        Args:
            state: Agent 状态（使用 user_request 作为查询）

        Returns:
            "## Relevant Strategies\n\n{strategy_xml}" 或 ""
        """
        if not state.evolution_enabled or not self._strategy_applier:
            return ""

        try:
            strategies = await self._strategy_applier.get_relevant_strategies(
                task_description=state.user_request,
            )
            if not strategies:
                return ""

            strategy_xml = self._strategy_applier.format_for_prompt(strategies)
            self._current_strategy_ids = [s.strategy_id for s in strategies]
            self._report_progress({
                "type": "strategy_recalled",
                "visibility": "internal",
                "summary": f"Applied {len(strategies)} procedural strategy item(s)",
                "data": {"strategy_ids": list(self._current_strategy_ids)},
            })
            return f"## Relevant Strategies\n\n{strategy_xml}"
        except Exception as exc:
            logger.warning("Strategy assembly failed in PlanningNode (non-blocking): %s", exc)
            return ""

    def _format_tree_for_prompt(
        self, tree: dict[str, Any], prefix: str = "", is_root: bool = False,
    ) -> str:
        """将文件树字典格式化为文本。"""
        lines: list[str] = []
        name = tree.get("name", "")
        node_type = tree.get("type", "")

        if node_type == "directory":
            children = tree.get("children", [])
            if not is_root:
                lines.append(f"{prefix}{name}/")
            for child in children:
                child_text = self._format_tree_for_prompt(
                    child, prefix if is_root else prefix + "  "
                )
                if child_text:
                    lines.append(child_text)
        elif node_type == "file":
            lines.append(f"{prefix}{name}")

        return "\n".join(lines)

    def _clean_json(self, content: str) -> str:
        """清理 LLM 输出，提取 JSON 数组。

        移除 markdown 代码块标记（```json、```）和无关文本。
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

        # 查找第一个 [ 或 {
        for start_char in ("[", "{"):
            idx = content.find(start_char)
            if idx >= 0:
                end_char = "]" if start_char == "[" else "}"
                end_idx = content.rfind(end_char)
                if end_idx > idx:
                    return content[idx : end_idx + 1]

        return content

    def _parse_steps(self, data: Any) -> list[PlanStep]:
        """将解析后的 JSON 数据转换为 PlanStep 列表。"""
        if not isinstance(data, list):
            raise ValueError(
                f"Expected JSON array, got {type(data).__name__}"
            )

        steps: list[PlanStep] = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected JSON object for step, got {type(item).__name__}"
                )

            step_id = item.get("step_id")
            if step_id is None:
                raise ValueError("Each step must have a 'step_id' field")
            if not isinstance(step_id, int) or step_id < 1:
                raise ValueError(
                    f"step_id must be a positive integer, got {step_id}"
                )

            description = item.get("description", "")
            if not description:
                raise ValueError(
                    f"Step {step_id} is missing 'description'"
                )

            action = item.get("action", "")
            valid_actions = {"create", "modify", "delete", "read", "command"}
            if action not in valid_actions:
                raise ValueError(
                    f"Step {step_id}: invalid action '{action}'. "
                    f"Must be one of: {', '.join(sorted(valid_actions))}"
                )

            target_file = item.get("target_file")
            if target_file is not None and not isinstance(target_file, str):
                raise ValueError(
                    f"Step {step_id}: 'target_file' must be a string or null"
                )

            risk = item.get("risk", "low")
            if risk not in ("low", "high"):
                raise ValueError(
                    f"Step {step_id}: invalid risk '{risk}'. "
                    "Must be 'low' or 'high'"
                )

            dependencies = item.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError(
                    f"Step {step_id}: 'dependencies' must be a list"
                )
            if not all(isinstance(d, int) for d in dependencies):
                raise ValueError(
                    f"Step {step_id}: 'dependencies' must be integers"
                )

            acceptance_criteria = item.get("acceptance_criteria", "")
            if not isinstance(acceptance_criteria, str):
                raise ValueError(
                    f"Step {step_id}: 'acceptance_criteria' must be a string"
                )
            if not acceptance_criteria.strip():
                acceptance_criteria = (
                    f"A successful {action} tool call records evidence for this step"
                )

            steps.append(
                PlanStep(
                    step_id=step_id,
                    description=description,
                    action=action,  # type: ignore[arg-type]
                    target_file=target_file,
                    risk=risk,  # type: ignore[arg-type]
                    dependencies=dependencies,
                    acceptance_criteria=acceptance_criteria,
                )
            )

        return steps

    def _validate_steps(self, steps: list[PlanStep]) -> None:
        """校验步骤列表的完整性。"""
        if not steps:
            raise ValueError("Plan must have at least one step")

        step_ids = {s.step_id for s in steps}

        # 校验 step_id 唯一性
        if len(step_ids) != len(steps):
            raise ValueError("Duplicate step_id found in plan")

        # 校验依赖存在
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_ids and dep >= step.step_id:
                    raise ValueError(
                        f"Step {step.step_id} depends on non-existent "
                        f"step {dep}"
                    )

    def _detect_conflicts(self, steps: list[PlanStep]) -> list[dict[str, Any]]:
        """检测计划中的操作冲突。

        Returns:
            冲突列表，每个冲突包含冲突描述和相关步骤 ID
        """
        conflicts: list[dict[str, Any]] = []

        # 按 target_file 分组
        file_ops: dict[str, list[PlanStep]] = {}
        for step in steps:
            if step.target_file:
                file_ops.setdefault(step.target_file, []).append(step)

        # 检测同一文件上的冲突操作
        for file_path, ops in file_ops.items():
            actions = {op.action for op in ops}
            if "delete" in actions and "modify" in actions:
                conflicts.append({
                    "type": "delete_and_modify",
                    "file": file_path,
                    "step_ids": [op.step_id for op in ops],
                    "message": (
                        f"File '{file_path}' is both modified and deleted"
                    ),
                })
            if "delete" in actions and "create" in actions:
                conflicts.append({
                    "type": "delete_and_create_same_file",
                    "file": file_path,
                    "step_ids": [op.step_id for op in ops],
                    "message": (
                        f"File '{file_path}' is both deleted and recreated"
                    ),
                })

        return conflicts
