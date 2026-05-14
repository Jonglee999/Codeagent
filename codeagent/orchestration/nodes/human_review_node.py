"""HumanReviewNode — 人工审核节点。

在关键决策点暂停执行，等待用户审核并做出决定。
支持以下场景：
- high_risk_plan: 高风险 planning 步骤（文件删除、批量修改）
- deviation_detected: 偏离检测触发后需要确认
- validation_failure: 连续验证失败后需要人工判断
- user_interrupt: 用户中途干预
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

from codeagent.orchestration.state import AgentState

logger = logging.getLogger(__name__)

# 审核类型
_REVIEW_TYPES = {
    "high_risk_plan": "高风险操作审核",
    "deviation_detected": "执行偏离检测",
    "validation_failure": "验证失败",
    "user_interrupt": "用户干预",
}

# 可用决策选项
_DECISION_OPTIONS = ["approve", "reject", "modify", "abort"]


class HumanReviewNode:
    """Human Review Node — 人工审核节点。

    在关键决策点暂停执行，等待用户审核并做出决定。
    支持外部注入 review_callback 实现自定义审核交互（如 Web UI、CLI 等）。
    """

    def __init__(
        self,
        review_callback: Callable[[dict], Awaitable[str]] | None = None,
    ) -> None:
        """初始化 Human Review 节点。

        Args:
            review_callback: 外部注入的审核回调函数。
                接收 review_request dict，返回 human_decision str。
                为 None 时使用默认控制台交互（input()）。
        """
        self._review_callback = review_callback

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        """执行 Human Review。

        如果 state.human_decision 已设置（通过 orchestrator.resume 注入），
        直接记录并返回。否则构建审核请求，通过 callback 或控制台收集决策。

        Args:
            state: 当前 Agent 状态

        Returns:
            dict: 更新后的状态字段
        """
        # 如果 decision 已注入（resume 路径），直接处理
        if state.human_decision:
            return self._handle_resume(state)

        # 确定审核类型并构建请求
        review_type = self._determine_review_type(state)
        review_request = self._build_review_request(state, review_type)

        # 收集决策
        try:
            if self._review_callback:
                decision = await self._review_callback(review_request)
            else:
                decision = await self._console_review(review_request)
        except Exception as exc:
            logger.error("Human review callback failed: %s", exc)
            decision = "abort"

        if decision not in _DECISION_OPTIONS:
            logger.warning("Invalid human decision '%s', defaulting to abort", decision)
            decision = "abort"

        return self._make_result(state, decision, review_request)

    def _handle_resume(self, state: AgentState) -> dict[str, Any]:
        """处理 resume 路径（decision 已通过 aupdate_state 注入）。"""
        decision = state.human_decision or "abort"
        logger.info("Human review resumed with decision: %s", decision)

        return {
            "human_decision": decision,
            "human_review_required": False,
            "deviation_detected": False,
            "deviation_count": 0,
            "review_request": None,
            "review_type": None,
            "execution_log": [
                *state.execution_log,
                {
                    "type": "human_review",
                    "timestamp": time.time(),
                    "human_decision": decision,
                    "resume": True,
                    "review_request": state.review_request,
                },
            ],
        }

    def _determine_review_type(self, state: AgentState) -> str:
        """根据 state 确定审核触发类型。"""
        if state.review_type in _REVIEW_TYPES:
            return state.review_type

        if state.human_review_required or state.deviation_detected:
            if state.review_request and state.review_request.get("review_type") == "deviation_detected":
                return "deviation_detected"
            if state.deviation_count >= 3:
                return "deviation_detected"

        # 检查 plan 中是否有高风险步骤（可能未触发 deviation）
        if state.plan and state.current_step_index == 0:
            high_risk = [s for s in state.plan if s.risk == "high"]
            if high_risk:
                return "high_risk_plan"

        # 检查验证失败
        if state.validation_results:
            failed = [r for r in state.validation_results if not r.passed]
            if failed and state.retry_count >= 3:
                return "validation_failure"

        return "user_interrupt"

    def _build_review_request(self, state: AgentState, review_type: str) -> dict:
        """根据审核类型构建审核请求内容。"""
        title = _REVIEW_TYPES.get(review_type, "人工审核")
        details: dict[str, Any] = {}
        context: dict[str, Any] = {}

        if review_type == "high_risk_plan":
            high_risk_steps = [
                {"step_id": s.step_id, "description": s.description,
                 "action": s.action, "target_file": s.target_file}
                for s in (state.plan or []) if s.risk == "high"
            ]
            details["high_risk_steps"] = high_risk_steps
            details["total_steps"] = len(state.plan or [])
            context["user_request"] = state.user_request

        elif review_type == "deviation_detected":
            if state.review_request and "details" in state.review_request:
                details.update(state.review_request["details"])
            details["deviation_count"] = state.deviation_count
            context["original_goal_summary"] = state.original_goal_summary

        elif review_type == "validation_failure":
            failed = [
                {
                    "file_path": (r.errors[0].file_path if r.errors else "unknown"),
                    "errors": [
                        {"line": e.line, "column": e.column, "message": e.message}
                        for e in r.errors[:3]
                    ],
                }
                for r in (state.validation_results or []) if not r.passed
            ]
            details["failed_validations"] = failed[:5]
            details["retry_count"] = state.retry_count
            context["user_request"] = state.user_request

        else:  # user_interrupt
            context["user_request"] = state.user_request
            context["current_step"] = state.current_step_index
            if state.plan:
                details["total_steps"] = len(state.plan)
                details["completed_steps"] = state.current_step_index

        return {
            "review_type": review_type,
            "title": title,
            "details": details,
            "options": _DECISION_OPTIONS.copy(),
            "context": context,
        }

    async def _console_review(self, review_request: dict) -> str:
        """默认控制台交互：打印审核请求并读取用户输入。"""
        import sys

        print("\n" + "=" * 60, file=sys.stderr)
        print(f"🔍 {review_request['title']}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        details = review_request.get("details", {})
        if review_request["review_type"] == "high_risk_plan":
            print("\n高风险步骤:", file=sys.stderr)
            for step in details.get("high_risk_steps", []):
                print(f"  [{step['step_id']}] {step['action']} {step['target_file']}"
                      f" — {step['description']}", file=sys.stderr)
        elif review_request["review_type"] == "deviation_detected":
            print(f"\n偏离计数: {details.get('deviation_count', '?')}", file=sys.stderr)
            if "tool_name" in details:
                print(f"工具: {details['tool_name']}", file=sys.stderr)
            if "step_description" in details:
                print(f"步骤: {details['step_description']}", file=sys.stderr)
        elif review_request["review_type"] == "validation_failure":
            print(f"\n验证失败 ({details.get('retry_count', '?')} 次重试):", file=sys.stderr)
            for v in details.get("failed_validations", []):
                print(f"  {v.get('file_path', '?')}: {v.get('errors', [])}", file=sys.stderr)

        print(f"\n选项: {', '.join(review_request['options'])}", file=sys.stderr)
        print("\n请输入决策:", file=sys.stderr)

        try:
            # 使用 asyncio 的事件循环在线程池中运行 input()
            import asyncio
            loop = asyncio.get_event_loop()
            decision = await loop.run_in_executor(None, input, "")
            decision = decision.strip().lower()
            if decision not in _DECISION_OPTIONS:
                print(f"无效选项，默认: abort", file=sys.stderr)
                return "abort"
            return decision
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return "abort"

    def _make_result(
        self, state: AgentState, decision: str, review_request: dict
    ) -> dict[str, Any]:
        """构建返回值。"""
        return {
            "human_decision": decision,
            "human_review_required": False,
            "deviation_detected": False,
            "deviation_count": 0,
            "review_request": None,
            "review_type": None,
            "execution_log": [
                *state.execution_log,
                {
                    "type": "human_review",
                    "timestamp": time.time(),
                    "human_decision": decision,
                    "review_type": review_request["review_type"],
                    "review_request": review_request,
                },
            ],
        }
