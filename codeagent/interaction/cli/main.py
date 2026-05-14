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

from codeagent.config import load_env_file, get_env
from codeagent.gateway.tool_gateway import IToolGateway
from codeagent.gateway.validation_gateway import (
    IValidationGateway,
    ValidationResult,
)
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
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState
from codeagent.tools.file.read_file import ReadFileTool
from codeagent.tools.file.write_file import WriteFileTool
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.registry import ToolRegistry
from codeagent.tools.terminal.run_terminal import RunTerminalTool
from codeagent.validation.syntax_validator import SyntaxValidator

logger = logging.getLogger(__name__)

# ── IValidationGateway 适配器 ─────────────────────────────────────────────


class _ValidationGateway(IValidationGateway):
    """将 SyntaxValidator 适配为 IValidationGateway 接口。"""

    def __init__(self, validator: SyntaxValidator) -> None:
        self._validator = validator

    async def run_syntax_check(self, file_path: str) -> ValidationResult:
        return await self._validator.check_file(file_path)

    async def run_lint(self, files: list[str]) -> ValidationResult:
        # Phase 1a: lint 简化为语法检查
        results = await self._validator.check_files(files)
        passed = all(r.passed for r in results)
        all_errors: list = []
        all_warnings: list = []
        for r in results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)
        return ValidationResult(
            passed=passed,
            errors=all_errors,
            warnings=all_warnings,
        )

    async def run_tests(self, project_root: str) -> ValidationResult:
        # Phase 1a: 不支持
        return ValidationResult(passed=True)

    async def run_runtime_check(self, file_path: str) -> ValidationResult:
        # Phase 1a: 不支持
        return ValidationResult(passed=True)


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


def _build_validation_gateway() -> _ValidationGateway:
    """构建验证 Gateway。"""
    validator = SyntaxValidator()
    return _ValidationGateway(validator)


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

    if not Path(project_root).is_dir():
        raise click.ClickException(f"Project directory not found: {project_root}")

    if no_color:
        os.environ["NO_COLOR"] = "1"

    # 确定模型
    resolved_model = (
        model or get_env("LLM_MODEL", "") or "openai/deepseek-v4-flash"
    )

    # ── 构建组件 ──────────────────────────────────────────
    try:
        tool_gateway = _build_tool_gateway(project_root)
    except Exception as e:
        raise click.ClickException(f"Failed to build tool gateway: {e}")

    try:
        validation_gateway = _build_validation_gateway()
    except Exception as e:
        raise click.ClickException(f"Failed to build validation gateway: {e}")

    try:
        llm = _build_llm(resolved_model)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to initialize LLM: {e}")

    execution_node = ExecutionNode(
        llm=llm,
        tool_gateway=tool_gateway,
        validation_gateway=validation_gateway,
        model_name=resolved_model,
        max_retries=max_retries,
    )

    state = AgentState(
        user_request=request,
        project_root=project_root,
        auto_mode=auto,
    )

    # ── 异步执行 ──────────────────────────────────────────
    total_start = time.monotonic()
    summary: str | None = None

    if not json_output:
        console.print(format_welcome())
        console.print(format_step_header(1, 1, "Executing request"))

    try:
        result = asyncio.run(execution_node(state))
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

    execution_log: list[dict[str, Any]] = result.get("execution_log", [])
    errors: list[str] = result.get("errors", [])

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
    console.print()


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
