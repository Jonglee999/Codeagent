"""CLI 命令行入口 — CodeAgent 的交互终端。

提供以下命令：
  ask     执行 Agent 任务请求
  init    初始化项目 .codeagent/ 配置
  config  查看/修改配置
  history 查看会话历史

使用 click 构建命令行接口，使用 rich 实现彩色终端输出。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import asyncio

import click

from codeagent.config import load_env_file, get_env, get_model
from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway_impl import ValidationGateway

from codeagent.interaction.cli.formatters import (
    console,
    format_error_report,
    format_execution_summary,
    format_final_report,
    format_json_output,
    format_llm_response,
    format_step_header,
    format_tool_call,
    format_tool_list,
    format_validation_result,
    format_welcome,
    make_progress,
)
from codeagent.orchestration.orchestrator import Orchestrator
from codeagent.orchestration.rollback import RollbackManager
from codeagent.orchestration.state import AgentState
from codeagent.tools.file.read_file import ReadFileTool
from codeagent.tools.file.write_file import WriteFileTool
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.registry import ToolRegistry
from codeagent.tools.terminal.run_terminal import RunTerminalTool

logger = logging.getLogger(__name__)

# ── IValidationGateway 适配器 ─────────────────────────────────────────────


class _ValidationGateway(ValidationGateway):
    """将 ValidationGateway 适配为 CLI 使用的网关接口。

    Phase 4 升级：使用 SyntaxValidator + StaticAnalyzer + RuntimeValidator
    的三层完整验证实现。
    """

    def __init__(self) -> None:
        super().__init__()


# ── 环境变量与 .env 加载 ────────────────────────────────────────────────────


# 使用 codeagent.config 代替本地的 _load_env_file / _get_env


# ── LLM 工厂 ────────────────────────────────────────────────────────────────


def _build_llm(model_name: str) -> Any:
    """构建 LLM 调用函数。

    使用 litellm.acompletion，需设置 LLM_API_KEY 环境变量。
    支持 LLM_API_BASE 环境变量设置自定义 API 地址（OpenAI 兼容接口）。
    内置网络重试机制（最多 3 次，指数退避）。
    """
    api_key = get_env("LLM_API_KEY", "")
    api_base = get_env("LLM_API_BASE", "")
    timeout = int(get_env("LLM_TIMEOUT", "60"))

    try:
        import litellm
    except ImportError:
        raise click.ClickException(
            "litellm is required. Install it with: pip install litellm"
        )

    litellm.set_verbose = False

    async def llm_call(**kwargs: Any) -> Any:
        """包装 litellm.acompletion 的异步调用，含网络重试。"""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                call_kwargs: dict[str, Any] = {
                    **{k: v for k, v in kwargs.items() if v is not None},
                    "timeout": timeout,
                }
                if api_key:
                    call_kwargs["api_key"] = api_key
                if api_base:
                    call_kwargs["api_base"] = api_base
                return await litellm.acompletion(**call_kwargs)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    return llm_call


# ── 组件构建 ────────────────────────────────────────────────────────────────


def _build_tool_gateway(project_root: str) -> ToolGateway:
    """构建工具 Gateway，注册 ReadFileTool、WriteFileTool 和 RunTerminalTool。"""
    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root=project_root))
    registry.register(WriteFileTool(project_root=project_root))
    try:
        registry.register(RunTerminalTool(project_root=project_root))
    except Exception:
        logger.warning("RunTerminalTool not available (Docker may not be installed)")
    return ToolGateway(registry)


def _build_validation_gateway(project_root: str = "") -> _ValidationGateway:
    """构建验证 Gateway。"""
    return _ValidationGateway()


def _build_context_gateway(project_root: str) -> Any:
    """构建上下文 Gateway，封装 ContextEngine 实现 IContextGateway。

    Args:
        project_root: 项目根目录路径

    Returns:
        ContextEngine 实例（实现了 IContextGateway 接口）
    """
    from codeagent.context_engine.engine import ContextEngine, ContextConfig

    budget = int(get_env("CONTEXT_BUDGET_TOKENS", "8000"))
    config = ContextConfig(total_budget=budget)
    return ContextEngine(config=config)


# ── click 命令 ──────────────────────────────────────────────────────────────


@click.group(
    help="CodeAgent — AI-powered coding assistant.",
    epilog="Example: codeagent ask \"Create a Flask app with /health endpoint\"",
)
@click.version_option(version="0.1.0", prog_name="codeagent")
def cli() -> None:
    """CodeAgent — AI-powered coding assistant."""
    # 加载项目根目录的 .env 文件
    load_env_file(Path.cwd() / ".env")


@cli.command()
@click.argument("request_text", nargs=-1, required=True)
@click.option(
    "--project",
    default=".",
    help="Project root directory (default: current directory)",
    show_default=True,
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Auto mode (skip all confirmations)",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    help="Max retry count for syntax error recovery",
    show_default=True,
)
@click.option(
    "--max-llm-calls",
    type=int,
    default=None,
    help="Max LLM calls per task (from env MAX_LLM_CALLS_PER_TASK or 50)",
)
@click.option(
    "--model",
    default=None,
    help="LLM model name (default: from LLM_MODEL env or deepseek/deepseek-v4-flash)",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed execution process",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output in JSON format (for script consumption)",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable colored output",
)
def ask(
    request_text: tuple[str, ...],
    project: str,
    auto: bool,
    max_retries: int,
    max_llm_calls: int | None,
    model: str | None,
    verbose: bool,
    json_output: bool,
    no_color: bool,
) -> None:
    """Execute an Agent task.

    Send a natural language request to CodeAgent. The agent reads and writes files,
    validates syntax, and reports results.

    Example: codeagent ask "Create a Python fibonacci function with tests"
    """
    request = " ".join(request_text)
    project_root = str(Path(project).resolve())

    if max_llm_calls is not None:
        os.environ["MAX_LLM_CALLS_PER_TASK"] = str(max_llm_calls)

    if not Path(project_root).is_dir():
        raise click.ClickException(f"Project directory not found: {project_root}")

    if no_color:
        os.environ["NO_COLOR"] = "1"

    # 确定模型 — R3: 使用 config.get_model() 统一默认值
    resolved_model = model or get_env("LLM_MODEL", "") or get_model()

    # ── 构建组件 ──────────────────────────────────────────
    try:
        tool_gateway = _build_tool_gateway(project_root)
    except Exception as e:
        raise click.ClickException(f"Failed to build tool gateway: {e}")

    try:
        context_gateway = _build_context_gateway(project_root)
    except Exception as e:
        raise click.ClickException(f"Failed to build context gateway: {e}")

    try:
        validation_gateway = _build_validation_gateway(project_root)
    except Exception as e:
        raise click.ClickException(f"Failed to build validation gateway: {e}")

    try:
        llm = _build_llm(resolved_model)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to initialize LLM: {e}")

    # ── 创建 Orchestrator（替代直接 ExecutionNode 调用）─────
    orchestrator = Orchestrator(
        context_gateway=context_gateway,
        tool_gateway=tool_gateway,
        validation_gateway=validation_gateway,
        llm=llm,
        model_name=resolved_model,
    )

    # ── 异步执行（含 Human Review 循环）─────────────────────
    total_start = time.monotonic()
    summary: str | None = None

    if not json_output:
        console.print(format_welcome())
        console.print(format_step_header(1, 1, "Executing request"))

    try:
        result = asyncio.run(
            _run_orchestrator_with_review(
                orchestrator=orchestrator,
                request=request,
                project_root=project_root,
                auto=auto,
            )
        )
    except KeyboardInterrupt:
        if json_output:
            click.echo(
                format_json_output(False, request, None, [], ["Interrupted by user"], 0.0)
            )
        else:
            console.print(
                format_error_report(["Interrupted by user"])
                or "[yellow]Interrupted by user[/yellow]"
            )
        return
    except Exception as e:
        if json_output:
            click.echo(
                format_json_output(False, request, None, [], [str(e)], 0.0)
            )
            return
        console.print(
            format_error_report([f"Execution failed: {e}"])
            or "[red]Execution failed[/red]"
        )
        raise click.ClickException(str(e))

    total_duration = (time.monotonic() - total_start) * 1000

    execution_log: list[dict[str, Any]] = result.execution_log
    errors: list[str] = result.errors

    # 从 last llm_response 提取摘要
    for entry in reversed(execution_log):
        if entry.get("type") == "llm_response":
            summary = entry.get("content")
            break

    success = len(errors) == 0

    # ── 持久化历史记录 ──────────────────────────────────────
    _save_session_history(project_root, {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "request": request,
        "success": success,
        "duration_ms": total_duration,
        "execution_log_count": len(execution_log),
        "error_count": len(errors),
    })

    # ── 输出 ──────────────────────────────────────────────
    if json_output:
        click.echo(
            format_json_output(success, request, summary, execution_log, errors, total_duration)
        )
        return

    # 详细输出
    if verbose:
        for entry in execution_log:
            _display_entry(entry, project_root)

    # 最终报告
    console.print()
    console.print(format_final_report(request, summary, execution_log, errors, total_duration))
    if not json_output:
        llm_call_count = result.llm_call_count
        if llm_call_count > 0:
            console.print(f"  LLM 调用次数: {llm_call_count}")
    console.print()

    # ── Rollback: 执行完成后检查是否可回滚 ────────────────
    accumulated_changes: list[dict] = result.accumulated_changes
    if accumulated_changes and not json_output and not auto:
        try:
            _handle_rollback_interactive(accumulated_changes)
        except Exception as e:
            logger.warning("Rollback handler error: %s", e)


def _handle_rollback_interactive(accumulated_changes: list[dict]) -> None:
    """交互式回滚处理 — 在 ask 命令执行完成后提供回滚选项。

    Args:
        accumulated_changes: 累计修改记录列表
    """
    from rich.prompt import Confirm, Prompt

    prompt_msg = (
        f"[bold]任务执行完成，共 {len(accumulated_changes)} 个文件变更。"
        "输入 /rollback 查看回滚选项，或直接回车跳过。[/bold]"
    )
    console.print(prompt_msg)

    user_input = Prompt.ask("")
    if not user_input or not user_input.startswith("/rollback"):
        return

    parts = user_input.split()
    step_id: int | None = None
    if len(parts) > 1:
        try:
            step_id = int(parts[1])
        except ValueError:
            console.print("[yellow]无效的步骤编号，将回滚全部。[/yellow]")

    manager = RollbackManager()
    state = AgentState(
        user_request="",
        project_root="",
        accumulated_changes=accumulated_changes,
    )

    # 展示回滚预览
    preview = manager.get_rollback_preview(state, step_id)
    console.print(preview)

    # 确认后执行
    if Confirm.ask("确认执行回滚？"):
        if step_id is not None:
            result = manager.rollback_step(state, step_id)
        else:
            result = manager.rollback_all(state)

        if result.success:
            console.print(
                f"[green]回滚完成：恢复 {len(result.restored_files)} 个文件，"
                f"删除 {len(result.deleted_files)} 个文件[/green]"
            )
        else:
            console.print(
                f"[yellow]回滚部分完成：恢复 {len(result.restored_files)} 个文件，"
                f"删除 {len(result.deleted_files)} 个文件，"
                f"{len(result.errors)} 个错误[/yellow]"
            )
            for err in result.errors:
                console.print(f"  [red]- {err}[/red]")
    else:
        console.print("[yellow]回滚已取消。[/yellow]")


def _display_entry(entry: dict[str, Any], project_root: str) -> None:
    """打印单条执行日志条目（--verbose 模式）。"""
    entry_type = entry.get("type", "")

    if entry_type == "tool_call":
        tool_name = entry.get("tool_name", "")
        arguments = entry.get("arguments", {})
        # 构造一个模拟的 ToolResult 用于格式化
        from codeagent.gateway.tool_gateway import ToolResult

        result = ToolResult(
            success=entry.get("success", False),
            data=entry.get("result"),
            error_message=entry.get("error"),
            error_code="",
            duration_ms=entry.get("duration_ms", 0),
        )
        console.print(format_tool_call(tool_name, arguments, result))

    elif entry_type == "syntax_check":
        console.print(
            format_validation_result(
                file_path=entry.get("file_path", ""),
                passed=entry.get("passed", False),
                errors=entry.get("errors", []),
                duration_ms=entry.get("duration_ms", 0),
            )
        )

    elif entry_type == "llm_response":
        content = entry.get("content", "")
        if content:
            console.print(format_llm_response(content))


# ── Orchestrator 异步执行（含 Human Review 循环）────────────────────────────


async def _run_orchestrator_with_review(
    orchestrator: Orchestrator,
    request: str,
    project_root: str,
    auto: bool,
) -> AgentState:
    """运行 Orchestrator 工作流，并在需要时处理 Human Review 循环。

    第一次运行 orchestrator.run()，如果返回结果标记需要人工审核且
    非 auto 模式，则循环提示用户输入决策并调用 orchestrator.resume()。

    Args:
        orchestrator: 编排器实例
        request: 用户请求
        project_root: 项目根目录
        auto: 是否自动模式

    Returns:
        最终 AgentState
    """
    result = await orchestrator.run(request, project_root, auto_mode=auto)

    # Human Review 循环：仅在需要审查且非 auto 模式下进行
    while result.human_review_required and not auto:
        _display_review_request(result.review_request)

        from rich.prompt import Prompt

        decision = Prompt.ask(
            "请输入决策",
            choices=["approve", "abort", "modify"],
            default="approve",
        )

        checkpoints = orchestrator.get_checkpoints()
        if not checkpoints:
            logger.warning("No checkpoints found for human review resume")
            break

        thread_id = checkpoints[-1]["thread_id"]
        result = await orchestrator.resume(thread_id, decision)

    return result


def _display_review_request(review_request: dict | None) -> None:
    """显示 Human Review 请求内容。"""
    if not review_request:
        return

    title = review_request.get("title", "人工审核")
    review_type = review_request.get("review_type", "")
    details = review_request.get("details", {})

    from rich.panel import Panel

    panel_text = f"[bold]{title}[/bold]\n\n"

    if review_type == "high_risk_plan":
        panel_text += "高风险步骤:\n"
        for step in details.get("high_risk_steps", []):
            panel_text += (
                f"  [{step['step_id']}] {step['action']} "
                f"{step.get('target_file', '')} — {step['description']}\n"
            )
    elif review_type == "deviation_detected":
        panel_text += f"偏离计数: {details.get('deviation_count', '?')}\n"
        if "tool_name" in details:
            panel_text += f"工具: {details['tool_name']}\n"
    elif review_type == "validation_failure":
        panel_text += f"验证失败 ({details.get('retry_count', '?')} 次重试):\n"
        for v in details.get("failed_validations", []):
            panel_text += f"  {v.get('file_path', '?')}: {v.get('errors', [])}\n"

    panel_text += "\n选项: approve / abort / modify"
    console.print(Panel(panel_text, title="Human Review", border_style="yellow"))


# ── 历史记录持久化 ────────────────────────────────────────────────────────────


def _save_session_history(project_root: str, session: dict) -> None:
    """将会话记录追加到 .codeagent/history.json。

    Args:
        project_root: 项目根目录路径
        session: 会话记录字典（timestamp, request, success, duration_ms 等）
    """
    from pathlib import Path
    import json

    history_path = Path(project_root) / ".codeagent" / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        try:
            sessions: list[dict] = json.loads(
                history_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            sessions = []
    else:
        sessions = []

    sessions.append(session)

    # 只保留最近 100 条记录
    sessions = sessions[-100:]

    history_path.write_text(
        json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@cli.command()
def init() -> None:
    """Initialize .codeagent/ configuration directory."""
    project_root = Path.cwd()
    codeagent_dir = project_root / ".codeagent"

    if codeagent_dir.exists():
        click.echo(f"✓ .codeagent/ already exists at {codeagent_dir}")
        return

    codeagent_dir.mkdir(parents=True, exist_ok=True)
    (codeagent_dir / "backups").mkdir(exist_ok=True)

    # 写入默认配置文件
    config = {
        "version": "0.1.0",
        "project_root": str(project_root),
        "auto_mode": False,
        "max_retries": 3,
    }
    config_path = codeagent_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    click.echo(f"✓ Initialized .codeagent/ at {codeagent_dir}")
    click.echo(f"  Created {config_path}")


@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--set", "set_key", nargs=2, metavar="KEY VALUE", help="Set a config key")
def config(show: bool, set_key: tuple[str, str] | None) -> None:
    """View or modify configuration."""
    codeagent_dir = Path.cwd() / ".codeagent"
    config_path = codeagent_dir / "config.json"

    if not config_path.exists():
        click.echo("Config not found. Run 'codeagent init' first.")
        return

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        click.echo(f"Error reading config: {e}")
        return

    if set_key:
        key, value = set_key
        cfg[key] = value
        config_path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        click.echo(f"✓ Set {key} = {value}")
        return

    if show:
        click.echo(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    click.echo("Usage: codeagent config --show  or  codeagent config --set KEY VALUE")


@cli.command()
@click.option("--limit", type=int, default=10, help="Number of recent sessions")
def history(limit: int) -> None:
    """View recent session history."""
    codeagent_dir = Path.cwd() / ".codeagent"
    history_path = codeagent_dir / "history.json"

    if not history_path.exists():
        click.echo("No session history found.")
        return

    try:
        sessions = json.loads(history_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        click.echo("No session history found.")
        return

    if not sessions:
        click.echo("No session history found.")
        return

    recent = sessions[-limit:]
    for i, session in enumerate(recent, 1):
        ts = session.get("timestamp", "unknown")
        req = session.get("request", "")[:80]
        status = "✓" if session.get("success") else "✗"
        click.echo(f" {i}. [{status}] {ts} — {req}")


# ── 入口点 ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI 入口点。"""
    cli()


if __name__ == "__main__":
    main()
