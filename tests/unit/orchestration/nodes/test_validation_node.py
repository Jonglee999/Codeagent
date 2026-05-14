"""ValidationNode 单元测试。

覆盖三层验证编排的所有分支：
- 全部通过 / 全部失败
- 各层独立失败
- 无修改文件、异常处理
- Markdown 报告格式
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationError,
    ValidationResult,
)
from codeagent.orchestration.nodes.validation_node import ValidationNode
from codeagent.orchestration.state import AgentState


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
) -> AgentState:
    return AgentState(
        user_request="test",
        project_root="/root",
        accumulated_changes=accumulated_changes or [],
        execution_log=execution_log or [],
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
