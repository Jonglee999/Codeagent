"""ValidationNode 单元测试。

覆盖三层验证编排的所有分支：
- 全部通过 / 全部失败
- 各层独立失败
- 无修改文件、异常处理
- Markdown 报告格式
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.orchestration.nodes.validation_node import ValidationNode
from codeagent.orchestration.state import AgentState
from codeagent.validation.error_analyzer import ErrorAnalyzer, FixSuggestion
from codeagent.validation.runtime_validator import RuntimeValidator
from codeagent.validation.static_analyzer import StaticAnalyzer


@pytest.fixture
def mock_gateway() -> AsyncMock:
    gw = AsyncMock(spec=IValidationGateway)
    gw.run_syntax_check.return_value = ValidationResult(passed=True)
    gw.run_lint.return_value = ValidationResult(passed=True)
    gw.run_runtime_check.return_value = ValidationResult(passed=True)
    return gw


@pytest.fixture
def node(mock_gateway: AsyncMock) -> ValidationNode:
    return ValidationNode(mock_gateway)


def make_state(
    accumulated_changes: list | None = None,
    execution_log: list | None = None,
    **overrides,
) -> AgentState:
    return AgentState(
        user_request="test",
        project_root="/root",
        accumulated_changes=accumulated_changes or [],
        execution_log=execution_log or [],
        **overrides,
    )


class TestValidationNode:
    """验证节点主流程测试。"""

    async def test_all_three_layers_pass(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert len(result["validation_results"]) == 3
        assert all(r.passed for r in result["validation_results"])
        mock_gateway.run_syntax_check.assert_awaited_once_with("main.py")
        mock_gateway.run_lint.assert_awaited_once()
        mock_gateway.run_runtime_check.assert_awaited_once_with("main.py")

    async def test_syntax_failure_continues_to_other_layers(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", line=5, message="Syntax error")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert not results[0].passed  # syntax
        assert results[1].passed  # lint still runs
        assert results[2].passed  # runtime still runs
        mock_gateway.run_lint.assert_awaited_once()
        mock_gateway.run_runtime_check.assert_awaited_once()

    async def test_lint_failure_still_runs_runtime(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_lint.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", line=10, message="Unused import")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert results[0].passed  # syntax
        assert not results[1].passed  # lint
        assert results[2].passed  # runtime
        mock_gateway.run_runtime_check.assert_awaited_once()

    async def test_all_layers_fail(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", message="Syntax error")],
        )
        mock_gateway.run_lint.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", message="Lint error")],
        )
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", message="Import error")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert not any(r.passed for r in result["validation_results"])

    async def test_benchmark_missing_conftest_dependency_is_degraded(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(
                file_path="conftest.py",
                message=(
                    "ImportError while loading conftest '/workspace/conftest.py'.\n"
                    "ModuleNotFoundError: No module named 'hypothesis'"
                ),
            )],
        )
        state = make_state(
            accumulated_changes=[{"file_path": "main.py"}],
            benchmark_instance_id="owner__repo-1",
        )

        result = await node(state)

        assert result["validation_state"] == "validation_degraded"
        assert any("official evaluation" in warning for warning in result["warnings"])

    async def test_non_benchmark_missing_dependency_remains_failed(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(
                file_path="conftest.py",
                message=(
                    "ImportError while loading conftest '/workspace/conftest.py'.\n"
                    "ModuleNotFoundError: No module named 'hypothesis'"
                ),
            )],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])

        result = await node(state)

        assert result["validation_state"] == "validation_failed"

    async def test_benchmark_ignores_static_error_on_unchanged_line(
        self, mock_gateway: AsyncMock, tmp_path
    ) -> None:
        source = tmp_path / "main.py"
        original = "danger = compile('x', '', 'exec')\n"
        source.write_text(original + "fixed = True\n", encoding="utf-8")
        mock_gateway.run_lint.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(
                file_path=str(source), line=1, code="SEC_COMPILE", message="compile",
            )],
        )
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(
                file_path="conftest.py",
                message="ImportError while loading conftest; No module named 'project'",
            )],
        )
        validation = ValidationNode(mock_gateway)
        state = AgentState(
            user_request="fix",
            project_root=str(tmp_path),
            benchmark_instance_id="owner__repo-1",
            accumulated_changes=[{
                "file_path": "main.py",
                "original_content": original,
            }],
        )

        result = await validation(state)

        assert result["validation_state"] == "validation_degraded"
        assert result["validation_results"][1].passed
        assert result["validation_results"][1].warnings[-1].code == "PRE_EXISTING_STATIC"

    async def test_benchmark_keeps_static_error_on_changed_line(
        self, mock_gateway: AsyncMock, tmp_path
    ) -> None:
        source = tmp_path / "main.py"
        source.write_text("safe = True\ndanger = compile('x', '', 'exec')\n", encoding="utf-8")
        mock_gateway.run_lint.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(
                file_path=str(source), line=2, code="SEC_COMPILE", message="compile",
            )],
        )
        validation = ValidationNode(mock_gateway)
        state = AgentState(
            user_request="fix",
            project_root=str(tmp_path),
            benchmark_instance_id="owner__repo-1",
            accumulated_changes=[{
                "file_path": "main.py",
                "original_content": "safe = True\n",
            }],
        )

        result = await validation(state)

        assert result["validation_state"] == "validation_failed"
        assert result["validation_results"][1].errors[0].line == 2

    async def test_no_changed_files_all_pass_trivially(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state()
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        assert result["validation_results"][0].duration_ms == 0.0
        mock_gateway.run_syntax_check.assert_not_called()

    async def test_multiple_changed_files(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[
            {"file_path": "main.py"},
            {"file_path": "utils.py"},
        ])
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        assert mock_gateway.run_syntax_check.await_count == 2
        mock_gateway.run_syntax_check.assert_any_call("main.py")
        mock_gateway.run_syntax_check.assert_any_call("utils.py")

    async def test_deduplicates_file_paths(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[
            {"file_path": "main.py"},
            {"file_path": "main.py"},
        ])
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        mock_gateway.run_syntax_check.assert_awaited_once_with("main.py")

    async def test_collects_from_execution_log_as_well(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(execution_log=[
            {
                "type": "tool_call",
                "tool_name": "write_file",
                "arguments": {"file_path": "log_file.py"},
            },
        ])
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        mock_gateway.run_syntax_check.assert_awaited_once_with("log_file.py")

    async def test_collects_apply_patch_from_execution_log(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        state = make_state(execution_log=[{
            "type": "tool_call",
            "tool_name": "apply_patch",
            "arguments": {"file_path": "patched.py"},
        }])

        await node(state)

        mock_gateway.run_syntax_check.assert_awaited_once_with("patched.py")

    async def test_prefers_accumulated_changes_over_log(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        """accumulated_changes 和 log 中出现同一文件应去重。"""
        state = make_state(
            accumulated_changes=[{"file_path": "shared.py"}],
            execution_log=[
                {
                    "type": "tool_call",
                    "tool_name": "write_file",
                    "arguments": {"file_path": "shared.py"},
                },
            ],
        )
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        mock_gateway.run_syntax_check.assert_awaited_once_with("shared.py")


class TestValidationNodeEdgeCases:
    """验证节点边缘情况测试。"""

    async def test_syntax_check_gateway_exception(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.side_effect = RuntimeError("parser crash")
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert not results[0].passed
        assert len(results[0].errors) == 1
        assert "parser crash" in results[0].errors[0].message

    async def test_lint_gateway_exception(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_lint.side_effect = RuntimeError("lint crash")
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert results[0].passed  # syntax
        assert not results[1].passed  # lint failed due to exception
        assert results[2].passed  # runtime

    async def test_runtime_gateway_exception(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_runtime_check.side_effect = RuntimeError("runtime crash")
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert results[0].passed  # syntax
        assert results[1].passed  # lint
        assert not results[2].passed  # runtime

    async def test_runtime_false_with_output_is_not_promoted_to_passed(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            output="1 failed during collection",
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])

        result = await node(state)

        runtime = result["validation_results"][2]
        assert runtime.passed is False
        assert runtime.output == "1 failed during collection"
        assert runtime.errors[0].message == "1 failed during collection"

    async def test_empty_file_path_in_changes_skipped(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[{"file_path": ""}])
        result = await node(state)

        assert all(r.passed for r in result["validation_results"])
        mock_gateway.run_syntax_check.assert_not_called()

    async def test_mixed_errors_and_warnings(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=True,
            warnings=[ValidationError(file_path="main.py", message="Unused variable", severity="warning")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        results = result["validation_results"]
        assert results[0].passed  # warnings don't fail
        assert len(results[0].warnings) == 1

    async def test_execution_log_appended_with_report(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert len(result["execution_log"]) == 1
        assert result["execution_log"][0]["type"] == "validation_report"
        assert "Validation Report" in result["execution_log"][0]["report"]

    async def test_preserves_existing_execution_log(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(
            accumulated_changes=[{"file_path": "main.py"}],
            execution_log=[{"type": "previous_entry"}],
        )
        result = await node(state)

        assert len(result["execution_log"]) == 2
        assert result["execution_log"][0]["type"] == "previous_entry"
        assert result["execution_log"][1]["type"] == "validation_report"


class TestStaticAnalyzerIntegration:
    """StaticAnalyzer 集成测试 — Layer 2 增强。"""

    async def test_full_static_analysis_executed(
        self, mock_gateway: AsyncMock
    ) -> None:
        """StaticAnalyzer.run_all() 被调用，结果正确聚合。"""
        mock_sa = AsyncMock(spec=StaticAnalyzer)
        mock_sa.run_all.return_value = [
            ValidationResult(passed=True),
            ValidationResult(passed=True),
            ValidationResult(passed=True),
        ]
        node = ValidationNode(mock_gateway, static_analyzer=mock_sa)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert len(result["validation_results"]) == 3
        assert all(r.passed for r in result["validation_results"])
        mock_sa.run_all.assert_awaited_once_with(["main.py"])
        # gateway.run_lint 不应再被调用
        mock_gateway.run_lint.assert_not_called()

    async def test_static_analysis_aggregates_errors(
        self, mock_gateway: AsyncMock
    ) -> None:
        """多个子分析的错误聚合为一个 ValidationResult。"""
        mock_sa = AsyncMock(spec=StaticAnalyzer)
        mock_sa.run_all.return_value = [
            ValidationResult(passed=True),  # lint 通过
            ValidationResult(                # typecheck 失败
                passed=False,
                errors=[ValidationError(file_path="main.py", line=5, message="Incompatible types", code="MYPY_ERROR")],
            ),
            ValidationResult(                # security 失败
                passed=False,
                errors=[ValidationError(file_path="main.py", line=10, message="eval detected", code="SEC_EVAL")],
            ),
        ]
        node = ValidationNode(mock_gateway, static_analyzer=mock_sa)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert not result["validation_results"][1].passed  # layer 2
        assert len(result["validation_results"][1].errors) == 2
        assert any("Incompatible" in e.message for e in result["validation_results"][1].errors)
        assert any("eval" in e.message for e in result["validation_results"][1].errors)

    async def test_static_analysis_breakdown_in_report(
        self, mock_gateway: AsyncMock
    ) -> None:
        """报告中包含子分析 breakdown warning。"""
        mock_sa = AsyncMock(spec=StaticAnalyzer)
        mock_sa.run_all.return_value = [
            ValidationResult(passed=True),
            ValidationResult(passed=False, errors=[ValidationError(file_path="x.py", message="Type error", code="MYPY")]),
            ValidationResult(
                passed=False,
                errors=[ValidationError(file_path="x.py", message="Security issue", code="SEC")],
            ),
        ]
        node = ValidationNode(mock_gateway, static_analyzer=mock_sa)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "Static Analysis: ❌ FAILED" in report
        assert "Type error" in report
        assert "Security issue" in report
        assert "warning(s)" in report  # breakdown warning

    async def test_static_analysis_fallback_to_gateway(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        """无 StaticAnalyzer 时回退到 gateway.run_lint()。"""
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert result["validation_results"][1].passed
        mock_gateway.run_lint.assert_awaited_once()


class TestRuntimeValidatorIntegration:
    """RuntimeValidator 集成测试 — Layer 3 增强。"""

    async def test_runtime_tests_executed(
        self, mock_gateway: AsyncMock
    ) -> None:
        """RuntimeValidator.run_tests() 被调用。"""
        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(passed=True)
        mock_rv.last_test_output = "1 passed in 0.01s"

        node = ValidationNode(mock_gateway, runtime_validator=mock_rv)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert result["validation_results"][2].passed
        assert result["validation_results"][2].output == "1 passed in 0.01s"
        mock_rv.run_tests.assert_awaited_once_with(test_targets=["main.py"])
        # gateway.run_runtime_check 不应被调用
        mock_gateway.run_runtime_check.assert_not_called()

    async def test_benchmark_contract_targets_runtime_tests(
        self, mock_gateway: AsyncMock
    ) -> None:
        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(passed=True)
        mock_rv.last_test_output = "2 passed"
        node = ValidationNode(mock_gateway, runtime_validator=mock_rv)
        state = make_state(
            accumulated_changes=[{"file_path": "src/main.py"}],
            benchmark_fail_to_pass=["tests/test_fix.py::test_regression"],
            benchmark_pass_to_pass=["tests/test_existing.py::test_still_passes"],
        )

        await node(state)

        mock_rv.run_tests.assert_awaited_once_with(
            test_targets=[
                "tests/test_fix.py::test_regression",
                "tests/test_existing.py::test_still_passes",
            ]
        )

    async def test_runtime_fallback_when_no_framework(
        self, mock_gateway: AsyncMock
    ) -> None:
        """无测试框架时使用降级检查。"""
        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=True,
            warnings=[
                ValidationError(
                    file_path="", message="No test framework detected", code="NO_TEST_FRAMEWORK", severity="warning",
                )
            ],
        )
        mock_rv.run_fallback_check.return_value = ValidationResult(passed=True)
        mock_rv.last_test_output = None

        node = ValidationNode(mock_gateway, runtime_validator=mock_rv)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert result["validation_results"][2].passed
        mock_rv.run_fallback_check.assert_awaited_once_with(["main.py"])

    async def test_runtime_tests_failure(
        self, mock_gateway: AsyncMock
    ) -> None:
        """测试失败时结果正确传播。"""
        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="test_main.py", message="AssertionError: assert 1 == 2", code="AssertionError")],
        )
        mock_rv.last_test_output = "FAILED test_main.py::test_fail - AssertionError: assert 1 == 2"

        node = ValidationNode(mock_gateway, runtime_validator=mock_rv)
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert not result["validation_results"][2].passed
        assert any("AssertionError" in e.message for e in result["validation_results"][2].errors)
        # 不应有降级检查
        mock_rv.run_fallback_check.assert_not_called()

    async def test_runtime_fallback_to_gateway(
        self, node: ValidationNode, mock_gateway: AsyncMock
    ) -> None:
        """无 RuntimeValidator 时回退到 gateway.run_runtime_check()。"""
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert result["validation_results"][2].passed
        mock_gateway.run_runtime_check.assert_awaited_once_with("main.py")


class TestFixSuggestionIntegration:
    """FixSuggestion 集成测试 — 错误分析和修复建议。"""

    async def test_fix_suggestions_in_result_dict(
        self, mock_gateway: AsyncMock
    ) -> None:
        """返回字典包含 fix_suggestions 键。"""
        mock_ea = MagicMock(spec=ErrorAnalyzer)
        mock_ea.analyze.return_value = [
            FixSuggestion(
                error_summary="AssertionError: assert 1 == 2",
                affected_files=["test_main.py"],
                likely_cause="Test assertion failed",
                suggested_fix="Review assertion logic",
                is_pre_existing=False,
            ),
        ]

        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="test_main.py", message="AssertionError", code="AssertionError")],
        )
        mock_rv.last_test_output = "FAILED test_main.py::test_fail - AssertionError: assert 1 == 2"

        node = ValidationNode(
            mock_gateway,
            runtime_validator=mock_rv,
            error_analyzer=mock_ea,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert "fix_suggestions" in result
        assert len(result["fix_suggestions"]) == 1
        assert result["fix_suggestions"][0].error_summary == "AssertionError: assert 1 == 2"
        mock_ea.analyze.assert_called_once()

    async def test_fix_suggestions_in_report(
        self, mock_gateway: AsyncMock
    ) -> None:
        """报告中包含 FixSuggestion 部分。"""
        mock_ea = MagicMock(spec=ErrorAnalyzer)
        mock_ea.analyze.return_value = [
            FixSuggestion(
                error_summary="TypeError: unsupported operand",
                affected_files=["utils.py"],
                likely_cause="Type mismatch",
                suggested_fix="Check function signature",
                is_pre_existing=True,
            ),
        ]

        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="utils.py", message="TypeError", code="TypeError")],
        )
        mock_rv.last_test_output = "FAILED test_utils.py::test_type - TypeError"

        node = ValidationNode(
            mock_gateway,
            runtime_validator=mock_rv,
            error_analyzer=mock_ea,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "### Fix Suggestions" in report
        assert "TypeError: unsupported operand" in report
        assert "Type mismatch" in report
        assert "Check function signature" in report
        assert "pre-existing" in report  # pre_existing 标记

    async def test_fix_suggestions_empty_when_runtime_passes(
        self, mock_gateway: AsyncMock
    ) -> None:
        """运行时通过时无 FixSuggestion。"""
        mock_ea = MagicMock(spec=ErrorAnalyzer)
        mock_ea.analyze.return_value = []

        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(passed=True)
        mock_rv.last_test_output = None

        node = ValidationNode(
            mock_gateway,
            runtime_validator=mock_rv,
            error_analyzer=mock_ea,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        assert len(result["fix_suggestions"]) == 0
        report = result["execution_log"][-1]["report"]
        assert "### Fix Suggestions" not in report
        # ErrorAnalyzer 不应被调用
        mock_ea.analyze.assert_not_called()

    async def test_fix_suggestions_with_pre_existing_markers(
        self, mock_gateway: AsyncMock
    ) -> None:
        """pre_existing 标记正确出现。"""
        mock_ea = MagicMock(spec=ErrorAnalyzer)
        mock_ea.analyze.return_value = [
            FixSuggestion(
                error_summary="ValueError: bad value",
                affected_files=["legacy.py"],
                likely_cause="Invalid input",
                suggested_fix="Add validation",
                is_pre_existing=True,  # 已有错误
            ),
            FixSuggestion(
                error_summary="NameError: undefined",
                affected_files=["new_code.py"],
                likely_cause="Missing definition",
                suggested_fix="Define before use",
                is_pre_existing=False,  # 新错误
            ),
        ]

        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="test_all.py", message="Failures", code="TEST_FAILURE")],
        )
        mock_rv.last_test_output = "FAILED test_all.py - Error"

        node = ValidationNode(
            mock_gateway,
            runtime_validator=mock_rv,
            error_analyzer=mock_ea,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "pre-existing" in report
        assert "new" in report
        assert len(result["fix_suggestions"]) == 2

    async def test_fix_suggestions_layer_failure_independent(
        self, mock_gateway: AsyncMock, tmp_path
    ) -> None:
        """某层验证失败不影响其他层和 FixSuggestion 收集。"""
        mock_sa = AsyncMock(spec=StaticAnalyzer)
        mock_sa.run_all.return_value = [
            ValidationResult(passed=True),
            ValidationResult(passed=True),
            ValidationResult(
                passed=False,
                errors=[ValidationError(file_path="main.py", line=3, message="Hardcoded password", code="SEC_HARDCODED_SECRET")],
            ),
        ]

        mock_ea = MagicMock(spec=ErrorAnalyzer)
        mock_ea.analyze.return_value = [
            FixSuggestion(
                error_summary="AssertionError: assert 1 == 2",
                affected_files=["test_main.py"],
                likely_cause="Assertion failed",
                suggested_fix="Fix assertion",
                is_pre_existing=False,
            ),
        ]

        mock_rv = AsyncMock(spec=RuntimeValidator)
        mock_rv.run_tests.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="test_main.py", message="AssertionError", code="AssertionError")],
        )
        mock_rv.last_test_output = "FAILED test_main.py::test_fail - AssertionError"

        node = ValidationNode(
            mock_gateway,
            static_analyzer=mock_sa,
            runtime_validator=mock_rv,
            error_analyzer=mock_ea,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)

        # Layer 1: syntax (via gateway) — 通过
        assert result["validation_results"][0].passed
        # Layer 2: static — 失败 (security scan)
        assert not result["validation_results"][1].passed
        # Layer 3: runtime — 失败
        assert not result["validation_results"][2].passed
        # FixSuggestions 正常收集
        assert len(result["fix_suggestions"]) == 1
        assert "Fix Suggestions" in result["execution_log"][-1]["report"]


class TestValidationReport:
    """验证报告格式测试。"""

    async def test_report_all_passed(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "### Validation Report" in report
        assert "Syntax: ✅ PASSED" in report
        assert "Static Analysis: ✅ PASSED" in report
        assert "Runtime Tests: ✅ PASSED" in report
        assert "**Summary:** 0/3 layer(s) failed" in report

    async def test_report_syntax_failure(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="main.py", line=5, message="Expected ':'")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "Syntax: ❌ FAILED (1 error(s))" in report
        assert "main.py:5: Expected ':'" in report
        assert "**Summary:** 1/3 layer(s) failed" in report

    async def test_report_all_failed(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="a.py", message="Syntax error")],
        )
        mock_gateway.run_lint.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="b.py", message="Lint error")],
        )
        mock_gateway.run_runtime_check.return_value = ValidationResult(
            passed=False,
            errors=[ValidationError(file_path="c.py", message="Import error")],
        )
        state = make_state(accumulated_changes=[{"file_path": "a.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "**Summary:** 3/3 layer(s) failed" in report

    async def test_report_shows_warnings(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=True,
            warnings=[ValidationError(file_path="main.py", message="Unused import", severity="warning")],
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "Syntax: ✅ PASSED (1 warning(s))" in report or "1 warning" in report

    async def test_report_truncates_many_errors(self, node: ValidationNode, mock_gateway: AsyncMock) -> None:
        many_errors = [
            ValidationError(file_path=f"file_{i}.py", message=f"Error {i}")
            for i in range(15)
        ]
        mock_gateway.run_syntax_check.return_value = ValidationResult(
            passed=False,
            errors=many_errors,
        )
        state = make_state(accumulated_changes=[{"file_path": "main.py"}])
        result = await node(state)
        report = result["execution_log"][-1]["report"]

        assert "15 error(s)" in report
        assert "... and 5 more" in report
