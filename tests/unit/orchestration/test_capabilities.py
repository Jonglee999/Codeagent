from typing import Any

from codeagent.gateway.tool_gateway import ToolDefinition
from codeagent.orchestration.capabilities import (
    select_mcp_server_names,
    select_tool_capabilities,
)
from codeagent.orchestration.policy import RunProfile


def _tool(name: str, **metadata: Any) -> ToolDefinition:
    return ToolDefinition(name=name, description=f"{name} tool", **metadata)


def test_focused_task_exposes_core_and_defers_privileged_tools() -> None:
    tools = [
        _tool("read_file", category="exploration", read_only=True),
        _tool("apply_patch", category="mutation", risk_level="medium"),
        _tool("delete_file", category="mutation", risk_level="high"),
        _tool("git", category="version_control", risk_level="medium"),
        _tool("mcp__github__search_issues", source="mcp:github", external=True),
    ]

    selection = select_tool_capabilities(
        "Fix the login button regression",
        tools,
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    )

    assert [tool.name for tool in selection.selected] == ["read_file", "apply_patch"]
    assert [tool.name for tool in selection.deferred] == [
        "delete_file",
        "git",
        "mcp__github__search_issues",
    ]
    assert selection.reasons["__policy__"].startswith("direct workflow")


def test_explicit_intent_activates_delete_git_and_matching_external_tool() -> None:
    tools = [
        _tool("delete_file", category="mutation", risk_level="high"),
        _tool("git", category="version_control", risk_level="medium"),
        _tool("mcp__github__search_issues", source="mcp:github", external=True),
        _tool("mcp__playwright__screenshot", source="mcp:playwright", external=True),
    ]

    selection = select_tool_capabilities(
        "Delete the obsolete file, inspect the git diff, then update the GitHub issue",
        tools,
        RunProfile(workflow="planned", context="full", memory="off", learning="off"),
    )

    assert [tool.name for tool in selection.selected] == [
        "delete_file",
        "git",
        "mcp__github__search_issues",
    ]
    assert [tool.name for tool in selection.deferred] == ["mcp__playwright__screenshot"]


def test_public_manifest_is_schema_free_but_keeps_risk_metadata() -> None:
    tool = _tool(
        "read_file",
        parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        category="exploration",
        read_only=True,
    )

    manifest = select_tool_capabilities(
        "Read the implementation",
        [tool],
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    ).public_metadata()

    assert manifest["selected"][0]["name"] == "read_file"
    assert manifest["selected"][0]["read_only"] is True
    assert "parameters_schema" not in manifest["selected"][0]


def test_mcp_server_selection_does_not_require_transport_startup() -> None:
    servers = {"github", "playwright", "company_docs"}

    assert select_mcp_server_names("Fix the local parser", servers) == set()
    assert select_mcp_server_names("Review the GitHub PR", servers) == {"github"}
    assert select_mcp_server_names("Use company_docs to find the policy", servers) == {
        "company_docs"
    }
    assert select_mcp_server_names("Use MCP tools for this task", servers) == servers


# ── P0-2: run_terminal is on-demand, not a default core tool ───────────────


def _run_terminal_tool() -> ToolDefinition:
    return _tool("run_terminal", category="execution", risk_level="medium")


def test_direct_simple_task_defers_run_terminal() -> None:
    """A focused read/edit request should not carry the run_terminal schema."""
    tools = [
        _tool("read_file", read_only=True),
        _tool("write_file"),
        _run_terminal_tool(),
    ]

    selection = select_tool_capabilities(
        "Add a docstring to util.py",
        tools,
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    )

    assert "run_terminal" not in {tool.name for tool in selection.selected}
    assert "run_terminal" in {tool.name for tool in selection.deferred}


def test_direct_task_with_command_intent_exposes_run_terminal() -> None:
    tools = [_tool("read_file", read_only=True), _run_terminal_tool()]

    selection = select_tool_capabilities(
        "Run the test suite and report failures",
        tools,
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    )

    assert "run_terminal" in {tool.name for tool in selection.selected}


def test_command_only_follow_up_exposes_only_terminal_schema() -> None:
    tools = [
        _tool("read_file", read_only=True),
        _tool("list_files", read_only=True),
        _tool("search_code", read_only=True),
        _tool("write_file"),
        _tool("apply_patch"),
        _tool("get_diagnostics", read_only=True),
        _run_terminal_tool(),
    ]

    selection = select_tool_capabilities(
        "再运行一下这个代码并返回结果",
        tools,
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    )

    assert [tool.name for tool in selection.selected] == ["run_terminal"]


def test_chinese_write_request_keeps_file_mutation_tools() -> None:
    tools = [
        _tool("read_file", read_only=True),
        _tool("write_file"),
        _tool("apply_patch"),
        _run_terminal_tool(),
    ]

    selection = select_tool_capabilities(
        "帮我写一个 Python 随机数生成器并运行",
        tools,
        RunProfile(workflow="direct", context="minimal", memory="off", learning="off"),
    )

    assert {tool.name for tool in selection.selected} == {
        "read_file", "write_file", "apply_patch", "run_terminal",
    }


def test_planned_workflow_keeps_run_terminal_for_command_steps() -> None:
    """Planned command steps require run_terminal; policy must not defer it."""
    tools = [_tool("read_file", read_only=True), _run_terminal_tool()]

    selection = select_tool_capabilities(
        "Refactor the module to match the new API",
        tools,
        RunProfile(workflow="planned", context="full", memory="off", learning="off"),
    )

    assert "run_terminal" in {tool.name for tool in selection.selected}


def test_planned_workflow_keeps_mutation_tools_when_issue_wording_is_indirect() -> None:
    tools = [
        _tool("read_file", read_only=True),
        _tool("write_file"),
        _tool("apply_patch"),
        _run_terminal_tool(),
    ]

    selection = select_tool_capabilities(
        "Correct expected format in invalid DurationField error message",
        tools,
        RunProfile(workflow="planned", context="minimal", memory="off", learning="off"),
    )

    assert {tool.name for tool in selection.selected} == {
        "read_file", "write_file", "apply_patch", "run_terminal",
    }


def test_direct_benchmark_keeps_mutation_and_test_tools_for_indirect_issue() -> None:
    tools = [
        _tool("read_file", read_only=True),
        _tool("list_files", read_only=True),
        _tool("search_code", read_only=True),
        _tool("write_file"),
        _tool("apply_patch"),
        _run_terminal_tool(),
    ]

    selection = select_tool_capabilities(
        "Unexpected format in a DurationField error message",
        tools,
        RunProfile(
            workflow="direct",
            context="minimal",
            memory="off",
            learning="off",
            benchmark=True,
        ),
    )

    assert {tool.name for tool in selection.selected} == {
        "read_file",
        "list_files",
        "search_code",
        "write_file",
        "apply_patch",
        "run_terminal",
    }


def test_benchmark_defers_external_tools_even_when_issue_contains_url() -> None:
    tools = [
        _tool("read_file", read_only=True),
        _tool("mcp__fetch__fetch", source="mcp:fetch", external=True),
    ]

    selection = select_tool_capabilities(
        "Fix local behavior described at https://example.invalid/spec",
        tools,
        RunProfile(
            workflow="direct",
            context="minimal",
            memory="off",
            learning="off",
            benchmark=True,
        ),
    )

    assert [tool.name for tool in selection.selected] == ["read_file"]
    assert [tool.name for tool in selection.deferred] == ["mcp__fetch__fetch"]
