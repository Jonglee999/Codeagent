"""Deterministic Observe/Reflect/Revise decision after validation."""

from __future__ import annotations

import time
from typing import Any

from codeagent.orchestration.routing import route_after_validation
from codeagent.orchestration.state import AgentState


class ReflectionNode:
    """Turn validation evidence into an explicit, inspectable next action."""

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        previous_failed = (state.reflection or {}).get("failed_validations", [])
        legacy_route = route_after_validation(state)
        action = {
            "end": "finish",
            "execution": "repair",
            "planning": "replan",
            "human_review": "review",
        }[legacy_route]
        failed = [
            {
                "layer": getattr(result, "layer", "unknown"),
                "errors": [
                    getattr(error, "message", str(error))
                    for error in (getattr(result, "errors", []) or [])
                ],
            }
            for result in state.validation_results
            if not getattr(result, "passed", False)
        ]
        reflection = {
            "next_action": action,
            "validation_count": len(state.validation_results),
            "failed_validations": failed,
            "retry_count": state.retry_count,
            "reason": (
                "all validations passed" if action == "finish"
                else "validation evidence requires revision"
            ),
            "avoid_repeat": bool(failed and failed == previous_failed),
            "revision_guidance": {
                "hypothesis": (
                    "The previous change did not address the observed validation evidence"
                    if failed else "No revision is required"
                ),
                "preferred_tools": (
                    ["read_file", "search_code", "get_diagnostics", "write_file", "apply_patch"]
                    if failed else []
                ),
                "instruction": (
                    "Inspect the failing location and change the hypothesis before editing again"
                    if failed and failed == previous_failed
                    else "Use the failed validation evidence to make the smallest targeted revision"
                    if failed else "Finish"
                ),
            },
        }
        return {
            "reflection": reflection,
            "reflection_count": state.reflection_count + 1,
            "execution_log": [
                *state.execution_log,
                {"type": "reflection", "timestamp": time.time(), **reflection},
            ],
        }
