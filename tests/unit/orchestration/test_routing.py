"""条件路由函数单元测试。

覆盖 route_after_planning、route_after_execution、route_after_validation
的所有分支条件。
"""

from __future__ import annotations


from codeagent.gateway.validation_gateway import ValidationError, ValidationResult
from codeagent.orchestration.routing import (
    route_after_execution,
    route_after_human_review,
    route_after_planning,
    route_after_validation,
    route_after_validation_evidence,
)
from codeagent.orchestration.state import AgentState, PlanStep


def make_step(
    step_id: int,
    risk: str = "low",
    action: str = "create",
    target_file: str = "f.py",
) -> PlanStep:
    return PlanStep(
        step_id=step_id, description=f"Step {step_id}", action=action,  # type: ignore[arg-type]
        target_file=target_file, risk=risk,  # type: ignore[arg-type]
    )


# ── route_after_planning ────────────────────────────────────


class TestRouteAfterPlanning:
    """规划完成后路由测试。"""

    def test_normal_low_risk_goes_to_execution(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            auto_mode=True,
        )
        assert route_after_planning(state) == "execution"

    def test_high_risk_non_auto_goes_to_human_review(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "high")],
            auto_mode=False,
        )
        assert route_after_planning(state) == "human_review"

    def test_high_risk_auto_goes_to_execution(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "high")],
            auto_mode=True,
        )
        assert route_after_planning(state) == "execution"

    def test_mixed_risk_non_auto_highlights_high(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low"), make_step(2, "high")],
            auto_mode=False,
        )
        assert route_after_planning(state) == "human_review"

    def test_context_insufficient_goes_to_context(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            semantic_context=None,
            file_tree={"name": "root", "type": "directory"},
        )
        result = route_after_planning(state)
        assert result in ("execution", "context")

    def test_empty_plan_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[],
        )
        assert route_after_planning(state) == "end"

    def test_none_plan_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=None,
        )
        assert route_after_planning(state) == "end"

    def test_plan_generation_error_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=None,
            errors=["Plan generation failed after 3 attempts: bad json"],
        )
        assert route_after_planning(state) == "end"

    def test_has_context_skips_context_route(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            semantic_context="some context",
            file_tree={"name": "root", "type": "directory"},
        )
        assert route_after_planning(state) == "execution"

    def test_no_file_tree_context_check_bypassed(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            semantic_context=None,
            file_tree=None,
        )
        assert route_after_planning(state) == "execution"


# ── route_after_execution ────────────────────────────────────


class TestRouteAfterExecution:
    """执行完成后路由测试。"""

    def test_plan_complete_goes_to_validation(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            current_step_index=1,
        )
        assert route_after_execution(state) == "validation"

    def test_plan_incomplete_continues_execution(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low"), make_step(2, "low")],
            current_step_index=1,  # 还有一步未执行
        )
        assert route_after_execution(state) == "execution"

    def test_direct_read_only_turn_skips_validation(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=None,
            direct_execution=True,
        )
        assert route_after_execution(state) == "end"

    def test_direct_mutation_goes_to_validation(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[],
            direct_execution=True,
            accumulated_changes=[{"file_path": "main.py"}],
        )
        assert route_after_execution(state) == "validation"

    def test_max_tool_calls_error_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            current_step_index=0,
            errors=["Exceeded max tool calls (10)"],
        )
        assert route_after_execution(state) == "end"

    def test_llm_failure_error_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            errors=["LLM call failed: connection error"],
        )
        assert route_after_execution(state) == "end"

    def test_human_abort_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision="abort",
        )
        assert route_after_execution(state) == "end"

    def test_human_approve_does_not_abort(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            current_step_index=1,
            human_decision="approve",
        )
        assert route_after_execution(state) == "validation"

    def test_unrelated_error_does_not_stop(self) -> None:
        """非 fatal 错误不应阻止流程。"""
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            current_step_index=1,
            errors=["Some warning message"],
        )
        assert route_after_execution(state) == "validation"

    def test_all_steps_completed_with_errors_goes_to_validation(self) -> None:
        """步骤完成但有非 fatal 错误仍可验证。"""
        state = AgentState(
            user_request="test", project_root="/root",
            plan=[make_step(1, "low")],
            current_step_index=1,
            errors=["Tool 'read_file' returned status 404"],
        )
        assert route_after_execution(state) == "validation"


# ── route_after_validation ──────────────────────────────────


class TestRouteAfterValidation:
    """验证完成后路由测试。"""

    def test_all_passed_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=True)],
        )
        assert route_after_validation(state) == "end"

    def test_success_skips_reflection_phase(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=True)],
        )
        assert route_after_validation_evidence(state) == "end"

    def test_failure_enters_reflection_phase(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
        )
        assert route_after_validation_evidence(state) == "reflection"

    def test_benchmark_missing_dependency_stops_local_repair_loop(self) -> None:
        runtime_output = """collected 0 items / 1 error
ERROR collecting tests/test_feature.py
E   ModuleNotFoundError: No module named 'django'
"""
        state = AgentState(
            user_request="test",
            project_root="/root",
            benchmark_instance_id="django__django-11179",
            validation_results=[
                ValidationResult(passed=True),
                ValidationResult(passed=True),
                ValidationResult(
                    passed=False,
                    output=runtime_output,
                    errors=[ValidationError(
                        file_path="tests/test_feature.py",
                        message="ModuleNotFoundError: No module named 'django'",
                    )],
                ),
            ],
        )

        assert route_after_validation(state) == "end"
        assert route_after_validation_evidence(state) == "end"

    def test_non_benchmark_missing_dependency_still_requests_repair(self) -> None:
        state = AgentState(
            user_request="test",
            project_root="/root",
            validation_results=[
                ValidationResult(passed=True),
                ValidationResult(passed=True),
                ValidationResult(
                    passed=False,
                    output=(
                        "collected 0 items / 1 error\n"
                        "ERROR collecting tests/test_feature.py\n"
                        "ModuleNotFoundError: No module named 'project'"
                    ),
                ),
            ],
        )

        assert route_after_validation(state) == "execution"
        assert route_after_validation_evidence(state) == "reflection"

    def test_benchmark_real_test_failure_still_requests_repair(self) -> None:
        state = AgentState(
            user_request="test",
            project_root="/root",
            benchmark_instance_id="django__django-11179",
            validation_results=[
                ValidationResult(passed=True),
                ValidationResult(passed=True),
                ValidationResult(
                    passed=False,
                    output="1 failed, 40 passed\nAssertionError: expected None",
                ),
            ],
        )

        assert route_after_validation(state) == "execution"
        assert route_after_validation_evidence(state) == "reflection"

    def test_failure_with_retry_returns_execution(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=1,
        )
        assert route_after_validation(state) == "execution"

    def test_failure_retry_exhausted_auto_returns_planning(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=3,
            auto_mode=True,
        )
        assert route_after_validation(state) == "planning"

    def test_failure_retry_exhausted_non_auto_returns_human_review(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=3,
            auto_mode=False,
        )
        assert route_after_validation(state) == "human_review"

    def test_mixed_results_some_failed(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[
                ValidationResult(passed=True),
                ValidationResult(passed=False),
            ],
            retry_count=0,
        )
        assert route_after_validation(state) == "execution"

    def test_empty_results_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[],
        )
        assert route_after_validation(state) == "end"

    def test_retry_count_2_still_allows_retry(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=2,
        )
        assert route_after_validation(state) == "execution"

    def test_retry_count_exactly_3_triggers_replan(self) -> None:
        """retry_count == 3 时触发重新规划。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=3,
            auto_mode=True,
        )
        assert route_after_validation(state) == "planning"

    def test_retry_count_9_triggers_human_review_even_in_auto(self) -> None:
        """retry_count >= 9 时即使 auto_mode 也应转到 human_review。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=9,
            auto_mode=True,
        )
        assert route_after_validation(state) == "human_review"

    def test_retry_count_9_triggers_human_review_non_auto(self) -> None:
        """retry_count >= 9 时非 auto_mode 也应转到 human_review。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=9,
            auto_mode=False,
        )
        assert route_after_validation(state) == "human_review"

    def test_retry_count_above_9_triggers_human_review(self) -> None:
        """retry_count > 9 时也应转到 human_review。"""
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[ValidationResult(passed=False)],
            retry_count=10,
            auto_mode=True,
        )
        assert route_after_validation(state) == "human_review"

    def test_retry_count_3_to_8_auto_goes_planning(self) -> None:
        """retry_count 在 3-8 之间且 auto_mode 时转到 planning。"""
        for rc in [3, 4, 5, 6, 7, 8]:
            state = AgentState(
                user_request="test", project_root="/root",
                validation_results=[ValidationResult(passed=False)],
                retry_count=rc,
                auto_mode=True,
            )
            assert route_after_validation(state) == "planning", (
                f"retry_count={rc} should route to planning"
            )

    def test_multiple_results_all_passed(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            validation_results=[
                ValidationResult(passed=True),
                ValidationResult(passed=True),
                ValidationResult(passed=True),
            ],
        )
        assert route_after_validation(state) == "end"


class TestRouteAfterHumanReview:
    """人工审核完成后路由测试。"""

    def test_approve_goes_to_execution(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision="approve",
        )
        assert route_after_human_review(state) == "execution"

    def test_abort_returns_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision="abort",
        )
        assert route_after_human_review(state) == "end"

    def test_modify_goes_to_planning(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision="modify",
        )
        assert route_after_human_review(state) == "planning"

    def test_none_decision_defaults_to_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision=None,
        )
        assert route_after_human_review(state) == "end"

    def test_unknown_decision_defaults_to_end(self) -> None:
        state = AgentState(
            user_request="test", project_root="/root",
            human_decision="unknown",
        )
        assert route_after_human_review(state) == "end"
