"""CLI 格式化工具单元测试。

覆盖 format_tool_call、format_validation_result、format_execution_summary、
format_final_report、format_json_output 等格式化函数。
"""

from __future__ import annotations

import json

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codeagent.gateway.tool_gateway import ToolDefinition, ToolResult
from codeagent.interaction.cli import formatters


def _render_panel(panel: Panel) -> str:
    """使用 rich Console 渲染 Panel 为纯文本。"""
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    console = Console(file=buf, width=120)
    console.print(panel)
    return buf.getvalue()


class TestFormatToolCall:
    """验证 format_tool_call 输出。"""

    def test_successful_write_file(self) -> None:
        result = ToolResult(
            success=True,
            data={
                "file_path": "test.py",
                "mode": "create",
                "diff": "--- a/test.py\n+++ b/test.py\n@@ -0,0 +1 @@\n+print('hello')\n",
                "lines_added": 1,
                "lines_removed": 0,
            },
            duration_ms=15.5,
        )
        panel = formatters.format_tool_call("write_file", {"file_path": "test.py", "content": "print('hello')\n", "mode": "create"}, result)
        assert isinstance(panel, Panel)
        assert "write_file" in panel.title or "Tool Call" in panel.title

    def test_failed_tool_call(self) -> None:
        result = ToolResult(
            success=False,
            error_message="File not found",
            error_code="FILE_NOT_FOUND",
            duration_ms=2.0,
        )
        panel = formatters.format_tool_call("read_file", {"file_path": "nonexistent.py"}, result)
        assert isinstance(panel, Panel)
        rendered = _render_panel(panel)
        assert "Error" in rendered

    def test_read_file_result(self) -> None:
        result = ToolResult(
            success=True,
            data={
                "content": "def hello():\n    return 'world'\n",
                "total_lines": 3,
                "read_range": {"start": 1, "end": 3},
                "file_path": "hello.py",
                "language": "python",
            },
            duration_ms=5.0,
        )
        panel = formatters.format_tool_call("read_file", {"file_path": "hello.py"}, result)
        assert isinstance(panel, Panel)
        assert "Tool Call" in panel.title

    def test_tool_without_data(self) -> None:
        result = ToolResult(success=True, data=None, duration_ms=0.0)
        panel = formatters.format_tool_call("noop", {}, result)
        assert isinstance(panel, Panel)


class TestFormatValidationResult:
    """验证 format_validation_result 输出。"""

    def test_passed(self) -> None:
        panel = formatters.format_validation_result("test.py", True, [], 5.0)
        assert isinstance(panel, Panel)
        rendered = _render_panel(panel).lower()
        assert "passed" in rendered

    def test_failed_with_errors(self) -> None:
        errors = [
            {"line": 1, "column": 5, "message": "invalid syntax"},
            {"line": 3, "column": 0, "message": "unexpected indent"},
        ]
        panel = formatters.format_validation_result("test.py", False, errors, 10.0)
        assert isinstance(panel, Panel)
        rendered = _render_panel(panel).lower()
        assert "failed" in rendered
        assert "invalid syntax" in rendered
        assert "unexpected indent" in rendered

    def test_empty_errors(self) -> None:
        panel = formatters.format_validation_result("test.py", False, [], 1.0)
        assert isinstance(panel, Panel)


class TestFormatDiff:
    """验证 format_diff 输出。"""

    def test_diff_syntax_object(self) -> None:
        diff = "--- a/test.py\n+++ b/test.py\n@@ -0,0 +1 @@\n+print('hello')\n"
        syntax = formatters.format_diff(diff)
        assert isinstance(syntax, Syntax)
        # 不同 pygments 版本可能返回 "diff" 或 "Diff"
        assert syntax.lexer.name.lower() == "diff"

    def test_empty_diff(self) -> None:
        syntax = formatters.format_diff("")
        assert isinstance(syntax, Syntax)

    def test_multiline_diff(self) -> None:
        diff = (
            "--- a/old.py\n+++ b/new.py\n"
            "@@ -1,3 +1,5 @@\n"
            " def foo():\n"
            "-    pass\n"
            "+    return 42\n"
        )
        syntax = formatters.format_diff(diff)
        assert isinstance(syntax, Syntax)
    """验证 format_llm_response 输出。"""

    def test_markdown_output(self) -> None:
        md = formatters.format_llm_response("# Hello\nThis is a test.")
        assert md.markup == "# Hello\nThis is a test."

    def test_empty_response(self) -> None:
        md = formatters.format_llm_response("")
        assert md.markup == ""

    def test_code_block(self) -> None:
        md = formatters.format_llm_response("```python\nx = 1\n```")
        assert "x = 1" in md.markup


class TestFormatExecutionSummary:
    """验证 format_execution_summary 输出。"""

    def test_empty_log(self) -> None:
        panel = formatters.format_execution_summary([], 0.0)
        assert isinstance(panel, Panel)
        assert "Summary" in panel.title

    def test_with_tool_calls(self) -> None:
        log = [
            {"type": "tool_call", "tool_name": "write_file", "success": True},
            {"type": "tool_call", "tool_name": "write_file", "success": True},
            {"type": "syntax_check", "passed": True},
        ]
        panel = formatters.format_execution_summary(log, 150.0)
        assert isinstance(panel, Panel)

    def test_with_failures(self) -> None:
        log = [
            {"type": "tool_call", "tool_name": "write_file", "success": False},
            {"type": "tool_call", "tool_name": "write_file", "success": True},
        ]
        panel = formatters.format_execution_summary(log, 100.0)
        assert isinstance(panel, Panel)

    def test_with_llm_response(self) -> None:
        log = [
            {"type": "tool_call", "tool_name": "read_file", "success": True},
            {"type": "llm_response", "content": "Done!"},
        ]
        panel = formatters.format_execution_summary(log, 50.0)
        assert isinstance(panel, Panel)


class TestFormatErrorReport:
    """验证 format_error_report 输出。"""

    def test_no_errors(self) -> None:
        assert formatters.format_error_report([]) is None

    def test_with_errors(self) -> None:
        panel = formatters.format_error_report(["Error 1", "Error 2"])
        assert isinstance(panel, Panel)
        rendered = _render_panel(panel)
        assert "Error 1" in rendered or "Error" in rendered

    def test_single_error(self) -> None:
        panel = formatters.format_error_report(["Only error"])
        assert isinstance(panel, Panel)


class TestFormatFinalReport:
    """验证 format_final_report 输出。"""

    def test_basic_report(self) -> None:
        panel = formatters.format_final_report(
            request="Create a function",
            summary="Done! Created fibonacci.py",
            log=[{"type": "tool_call", "tool_name": "write_file", "success": True}],
            errors=[],
            total_duration_ms=100.0,
        )
        assert isinstance(panel, Panel)

    def test_with_errors(self) -> None:
        panel = formatters.format_final_report(
            request="Test",
            summary=None,
            log=[],
            errors=["Something failed"],
            total_duration_ms=50.0,
        )
        assert isinstance(panel, Panel)

    def test_no_summary(self) -> None:
        panel = formatters.format_final_report(
            request="Test",
            summary=None,
            log=[],
            errors=[],
            total_duration_ms=0.0,
        )
        assert isinstance(panel, Panel)


class TestFormatJsonOutput:
    """验证 format_json_output 输出。"""

    def test_successful_output(self) -> None:
        output = formatters.format_json_output(
            success=True,
            request="Create fibonacci",
            summary="Created fibonacci.py",
            log=[{"type": "tool_call", "tool_name": "write_file", "success": True}],
            errors=[],
            total_duration_ms=100.0,
        )
        data = json.loads(output)
        assert data["success"] is True
        assert data["request"] == "Create fibonacci"
        assert data["summary"] == "Created fibonacci.py"
        assert len(data["execution_log"]) == 1
        assert data["errors"] == []

    def test_failed_output(self) -> None:
        output = formatters.format_json_output(
            success=False,
            request="Do something",
            summary=None,
            log=[],
            errors=["Error occurred"],
            total_duration_ms=50.0,
        )
        data = json.loads(output)
        assert data["success"] is False
        assert data["errors"] == ["Error occurred"]

    def test_empty_log(self) -> None:
        output = formatters.format_json_output(
            success=True,
            request="Test",
            summary=None,
            log=[],
            errors=[],
            total_duration_ms=0.0,
        )
        data = json.loads(output)
        assert data["success"] is True
        assert data["execution_log"] == []


class TestFormatWelcome:
    """验证 format_welcome 输出。"""

    def test_welcome_panel(self) -> None:
        panel = formatters.format_welcome()
        assert isinstance(panel, Panel)
        assert "CodeAgent" in str(panel.renderable)


class TestFormatToolList:
    """验证 format_tool_list 输出。"""

    def test_tool_list_table(self) -> None:
        tools = [
            ToolDefinition(name="read_file", description="Read files"),
            ToolDefinition(name="write_file", description="Write files"),
        ]
        table = formatters.format_tool_list(tools)
        assert isinstance(table, Table)
        assert table.row_count == 2

    def test_empty_tool_list(self) -> None:
        table = formatters.format_tool_list([])
        assert isinstance(table, Table)
        assert table.row_count == 0


class TestMakeProgress:
    """验证 make_progress 创建。"""

    def test_progress_instance(self) -> None:
        progress = formatters.make_progress()
        # 验证 progress 可正常使用
        task_id = progress.add_task("Testing", total=100)
        progress.update(task_id, advance=50)
        progress.remove_task(task_id)


class TestFormatStepHeader:
    """验证 format_step_header 输出。"""

    def test_step_header(self) -> None:
        panel = formatters.format_step_header(1, 3, "Analyzing")
        assert isinstance(panel, Panel)
        assert "Step 1/3" in str(panel.renderable)
