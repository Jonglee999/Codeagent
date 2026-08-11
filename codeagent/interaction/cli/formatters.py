"""CLI 输出格式化 — 使用 rich 库实现彩色终端输出。

提供统一的格式化函数供 CLI 主模块使用，包括：
- Markdown 渲染（Agent 生成的报告）
- Diff 语法高亮显示
- 工具调用和验证结果面板
- 执行进度和最终报告
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from codeagent.gateway.tool_gateway import ToolDefinition, ToolResult

console = Console()


def make_progress() -> Progress:
    """创建一个标准的 Progress 实例。"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def format_step_header(step_num: int, total: int, title: str) -> Panel:
    """格式化步骤标题面板。"""
    return Panel(
        Text(f"Step {step_num}/{total}", style="bold cyan"),
        title=Text(title, style="bold white"),
        border_style="cyan",
        padding=(1, 2),
    )


def format_tool_call(name: str, arguments: dict[str, Any], result: ToolResult) -> Panel:
    """格式化工具调用信息。

    Args:
        name: 工具名称
        arguments: 调用参数
        result: 执行结果

    Returns:
        包含工具调用详情的 Panel
    """
    # 参数区域
    args_text = Text()
    args_text.append(f"[{name}]\n\n", style="bold yellow")
    args_text.append("Arguments:\n", style="underline")
    for k, v in arguments.items():
        val_str = str(v)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        args_text.append(f"  {k}: ", style="green")
        args_text.append(f"{val_str}\n", style="white")

    # 结果区域
    result_text = _format_result_preview(result)

    content = Group(args_text, result_text)
    border = "green" if result.success else "red"
    title_text = f"Tool Call: {name}"
    return Panel(content, title=title_text, border_style=border, padding=(1, 2))


def _format_result_preview(result: ToolResult) -> Text | str:
    """格式化工具结果预览。"""
    lines = Text()
    if result.success and result.data:
        lines.append("\nResult: ", style="underline")
        if isinstance(result.data, dict):
            # 针对 write_file 结果
            if "diff" in result.data:
                lines.append("File written\n", style="green")
                lines.append(f"  Path: {result.data.get('file_path', '')}\n", style="white")
                lines.append(f"  Mode: {result.data.get('mode', '')}\n", style="white")
                lines.append(f"  Lines: +{result.data.get('lines_added', 0)}"
                             f"/-{result.data.get('lines_removed', 0)}\n", style="white")
            # 针对 read_file 结果
            elif "content" in result.data:
                content = result.data["content"]
                preview = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"  Content ({len(content)} chars):\n", style="white")
                lines.append(f"  {preview}\n", style="dim")
            else:
                preview = json.dumps(result.data, ensure_ascii=False)
                if len(preview) > 300:
                    preview = preview[:300] + "..."
                lines.append(f"  {preview}\n", style="white")
        else:
            lines.append(f"  {result.data}\n", style="white")
        lines.append(f"\nDuration: {result.duration_ms:.1f}ms", style="dim")
    elif not result.success:
        lines.append(f"\nError: {result.error_message}", style="bold red")
        if result.error_code:
            lines.append(f" ({result.error_code})", style="red")
        lines.append("\n")
    return lines


def format_validation_result(
    file_path: str,
    passed: bool,
    errors: list[dict[str, Any]],
    duration_ms: float,
) -> Panel:
    """格式化语法检查结果。

    Args:
        file_path: 检查的文件路径
        passed: 是否通过
        errors: 错误列表
        duration_ms: 耗时

    Returns:
        包含验证结果的 Panel
    """
    if passed:
        content = Text(f"[OK] Syntax check passed ({duration_ms:.1f}ms)", style="green")
        return Panel(content, title=f"Validation: {file_path}", border_style="green", padding=(1, 2))

    lines: list[Text | str] = [
        Text(f"[FAIL] Syntax check failed ({duration_ms:.1f}ms)", style="bold red"),
    ]
    for err in errors:
        line_num = err.get("line", 0)
        col_num = err.get("column", 0)
        msg = err.get("message", "Unknown error")
        lines.append(Text(f"  L{line_num}:{col_num} {msg}", style="red"))

    content = Group(*lines)
    return Panel(content, title=f"Validation: {file_path}", border_style="red", padding=(1, 2))


def format_diff(diff_text: str) -> Syntax:
    """将 unified diff 文本格式化为语法高亮的 Syntax 对象。

    Args:
        diff_text: unified diff 文本

    Returns:
        Syntax 对象，可直接传递给 Console.print()
    """
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)


def format_llm_response(content: str) -> Markdown:
    """将 LLM 的文本响应格式化为 Markdown。

    Args:
        content: LLM 返回的文本

    Returns:
        Markdown 可渲染对象
    """
    return Markdown(content)


def format_execution_summary(
    log: list[dict[str, Any]],
    total_duration_ms: float,
) -> Panel:
    """格式化执行摘要面板。

    Args:
        log: 执行日志
        total_duration_ms: 总耗时

    Returns:
        执行摘要 Panel
    """
    tool_calls = [e for e in log if e.get("type") == "tool_call"]
    syntax_checks = [e for e in log if e.get("type") == "syntax_check"]
    llm_responses = [e for e in log if e.get("type") == "llm_response"]

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Tool calls:", str(len(tool_calls)))
    table.add_row("Syntax checks:", str(len(syntax_checks)))
    table.add_row("LLM responses:", str(len(llm_responses)))

    # 工具调用成功/失败统计
    if tool_calls:
        succeeded = sum(1 for e in tool_calls if e.get("success"))
        failed = sum(1 for e in tool_calls if not e.get("success"))
        table.add_row("Succeeded:", str(succeeded))
        if failed:
            table.add_row("Failed:", Text(str(failed), style="red"))

    table.add_row("Total time:", f"{total_duration_ms:.1f}ms")

    return Panel(table, title="Execution Summary", border_style="blue", padding=(1, 2))


def format_error_report(errors: list[str]) -> Panel | None:
    """格式化错误报告。

    Args:
        errors: 错误消息列表

    Returns:
        错误报告 Panel，无错误时返回 None
    """
    if not errors:
        return None

    content = Group(*[Text(f"  - {e}", style="red") for e in errors])
    return Panel(content, title="Errors", border_style="red", padding=(1, 2))


def format_final_report(
    request: str,
    summary: str | None,
    log: list[dict[str, Any]],
    errors: list[str],
    total_duration_ms: float,
) -> Panel:
    """格式化最终报告。

    Args:
        request: 用户请求
        summary: Agent 的摘要响应（可选）
        log: 执行日志
        errors: 错误列表
        total_duration_ms: 总耗时

    Returns:
        最终报告的 Panel
    """
    components: list[Any] = []

    # 请求回顾
    components.append(Text(f"Request: {request}", style="bold white"))

    # Agent 摘要
    if summary:
        components.append(Rule(style="dim"))
        components.append(Text("Agent Response:", style="bold cyan"))
        components.append(Markdown(summary))

    # 执行摘要
    components.append(Rule(style="dim"))
    components.append(format_execution_summary(log, total_duration_ms))

    # 错误
    err_panel = format_error_report(errors)
    if err_panel:
        components.append(err_panel)

    content = Group(*components)
    return Panel(content, title="Final Report", border_style="green", padding=(1, 2))


def format_json_output(
    success: bool,
    request: str,
    summary: str | None,
    log: list[dict[str, Any]],
    errors: list[str],
    total_duration_ms: float,
) -> str:
    """将结果格式化为 JSON 字符串（用于 --json 模式）。

    Args:
        success: 是否整体成功
        request: 用户请求
        summary: Agent 响应摘要
        log: 执行日志
        errors: 错误列表
        total_duration_ms: 总耗时

    Returns:
        JSON 格式的字符串
    """
    output = {
        "success": success,
        "request": request,
        "summary": summary,
        "execution_log": log,
        "errors": errors,
        "total_duration_ms": total_duration_ms,
    }
    return json.dumps(output, ensure_ascii=False, indent=2, default=str)


def format_welcome() -> Panel:
    """格式化欢迎信息。"""
    text = Text()
    text.append("CodeAgent - AI-powered coding assistant\n\n", style="bold cyan")
    text.append("Type ", style="white")
    text.append("codeagent ask \"your request\"", style="bold yellow")
    text.append(" to get started.\n", style="white")
    text.append("Use ", style="dim")
    text.append("--help", style="italic dim")
    text.append(" for more information.", style="dim")
    return Panel(text, border_style="cyan", padding=(1, 2))


def format_tool_list(tools: list[ToolDefinition]) -> Table:
    """格式化工具列表表格。

    Args:
        tools: 工具定义列表

    Returns:
        工具列表 Table
    """
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for t in tools:
        table.add_row(t.name, t.description)

    return table
