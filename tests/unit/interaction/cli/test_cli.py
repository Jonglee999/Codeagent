"""CLI 命令行单元测试 — 使用 click.testing.CliRunner 测试命令行为。

测试 ask / init / config / history 命令的解析、参数校验、输出格式。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from codeagent.interaction.cli.main import cli
from codeagent.orchestration.state import AgentState


def _make_result_state(**overrides: object) -> AgentState:
    """Create an AgentState with defaults for testing the ask command.

    Tests can override specific fields to simulate different execution outcomes.
    """
    defaults: dict[str, object] = {
        "user_request": "test",
        "project_root": "/tmp",
        "auto_mode": False,
        "execution_log": [],
        "errors": [],
        "accumulated_changes": [],
        "llm_call_count": 0,
        "human_review_required": False,
        "review_request": None,
    }
    defaults.update(overrides)
    return AgentState(**defaults)  # type: ignore[arg-type]


def _mock_run(return_value: AgentState) -> callable:
    """Create a side_effect for asyncio.run mock that properly closes the coroutine.

    Prevents RuntimeWarning about unawaited coroutines when mocking asyncio.run.
    """
    def _side_effect(coro: object, *args: object, **kwargs: object) -> AgentState:
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[union-attr]
        return return_value
    return _side_effect


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """创建临时项目目录，含 .env 文件。"""
    (tmp_path / "test_file.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


# ── Tests: ask command ─────────────────────────────────────────────────────


class TestAskCommand:
    """验证 ask 命令的参数解析和基本行为。"""

    def test_ask_without_request_shows_error(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask"])
        assert result.exit_code != 0

    def test_ask_with_request_requires_valid_project(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["ask", "hello", "--project", "/nonexistent/path"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_basic_execution(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """基本执行流 — 验证组件构建和调用。"""
        # Mock LLM
        mock_llm = AsyncMock()
        mock_build_llm.return_value = mock_llm

        # Mock tool gateway
        mock_tg = MagicMock()
        mock_tg.list_tools.return_value = []
        mock_build_tg.return_value = mock_tg

        # Mock validation gateway
        mock_vg = MagicMock()
        mock_build_vg.return_value = mock_vg

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "write a test function", "--project", str(temp_project)],
            )

        assert result.exit_code == 0
        mock_build_tg.assert_called_once()
        mock_build_vg.assert_called_once()
        mock_build_llm.assert_called_once()

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_with_auto_flag(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--auto 标志应正常传递。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--auto"],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_with_max_retries(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--max-retries 应被正常解析。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--max-retries", "5"],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_with_model(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--model 应被传递到 LLM 构建。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--model", "gpt-4"],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_verbose_output(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--verbose 应显示详细输出。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state(
                execution_log=[
                    {
                        "type": "llm_response",
                        "content": "I will create the file.",
                    },
                    {
                        "type": "tool_call",
                        "tool_name": "write_file",
                        "arguments": {"file_path": "test.py", "content": "print('ok')"},
                        "success": True,
                        "result": {"file_path": "test.py", "mode": "create"},
                        "duration_ms": 10.0,
                    },
                ],
                errors=[],
            ))
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--verbose"],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_json_output(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--json 应输出 JSON 格式。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--json"],
            )

        assert result.exit_code == 0
        # JSON output should be parseable
        data = json.loads(result.output)
        assert data["success"] is True
        assert data["request"] == "hello"

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_with_errors(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """执行中有错误时应显示错误信息。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state(
                errors=["Tool execution failed"],
            ))
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    def test_ask_llm_build_failure(
        self,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """LLM 构建失败应报告错误。"""
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch(
            "codeagent.interaction.cli.main._build_llm",
            side_effect=ValueError("No API key"),
        ):
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        assert result.exit_code != 0

    def test_ask_multiple_words(self, runner: CliRunner, temp_project: Path) -> None:
        """多词请求应被合并。"""
        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "create", "a", "test", "file", "--project", str(temp_project)],
            )

        assert result.exit_code == 0

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_keyboard_interrupt_handled(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """KeyboardInterrupt 应被优雅处理。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        assert result.exit_code in (0, 1)

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_no_color_flag(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--no-color 标志应被接受。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--no-color"],
            )

        assert result.exit_code == 0


# ── Tests: R1 — Orchestrator 集成 ──────────────────────────────────────────


class TestAskOrchestratorIntegration:
    """R1: 验证 ask 命令已接入 Orchestrator 而非直接调用 ExecutionNode。"""

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    @patch("codeagent.interaction.cli.main._build_context_gateway")
    def test_ask_creates_orchestrator_and_runs(
        self,
        mock_build_cg: MagicMock,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """验证 ask 命令创建 Orchestrator 并调用其 run 方法。"""
        mock_build_cg.return_value = MagicMock()
        mock_build_llm.return_value = AsyncMock()
        mock_build_vg.return_value = MagicMock()
        mock_build_tg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.Orchestrator") as mock_orc_cls:
            mock_orc = MagicMock()
            mock_orc.run = AsyncMock(return_value=_make_result_state())
            mock_orc_cls.return_value = mock_orc

            result = runner.invoke(
                cli,
                ["ask", "create hello.py", "--project", str(temp_project)],
            )

        assert result.exit_code == 0
        # Orchestrator 应被实例化
        mock_orc_cls.assert_called_once()
        _, kwargs = mock_orc_cls.call_args
        assert "context_gateway" in kwargs
        assert "tool_gateway" in kwargs
        assert "validation_gateway" in kwargs
        assert "llm" in kwargs
        # run 方法应被调用
        mock_orc.run.assert_awaited_once()
        run_args = mock_orc.run.call_args[0]
        run_kwargs = mock_orc.run.call_args.kwargs
        assert run_args[0] == "create hello.py"
        assert run_kwargs["auto_mode"] is False

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    @patch("codeagent.interaction.cli.main._build_context_gateway")
    def test_ask_auto_mode_passed_to_orchestrator(
        self,
        mock_build_cg: MagicMock,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """--auto 标志应传递到 orchestrator.run 的 auto_mode 参数。"""
        mock_build_cg.return_value = MagicMock()
        mock_build_llm.return_value = AsyncMock()
        mock_build_vg.return_value = MagicMock()
        mock_build_tg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.Orchestrator") as mock_orc_cls:
            mock_orc = MagicMock()
            mock_orc.run = AsyncMock(return_value=_make_result_state())
            mock_orc_cls.return_value = mock_orc

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project), "--auto"],
            )

        assert result.exit_code == 0
        mock_orc.run.assert_awaited_once()
        assert mock_orc.run.call_args.kwargs["auto_mode"] is True

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    @patch("codeagent.interaction.cli.main._build_context_gateway")
    def test_ask_context_gateway_built(
        self,
        mock_build_cg: MagicMock,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """验证 _build_context_gateway 被调用。"""
        mock_build_cg.return_value = MagicMock()
        mock_build_llm.return_value = AsyncMock()
        mock_build_vg.return_value = MagicMock()
        mock_build_tg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        mock_build_cg.assert_called_once()

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    @patch("codeagent.interaction.cli.main._build_context_gateway")
    def test_ask_context_gateway_build_failure(
        self,
        mock_build_cg: MagicMock,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """context_gateway 构建失败应报告错误。"""
        mock_build_cg.side_effect = RuntimeError("No context engine available")
        mock_build_llm.return_value = AsyncMock()
        mock_build_vg.return_value = MagicMock()
        mock_build_tg.return_value = MagicMock()

        result = runner.invoke(
            cli,
            ["ask", "hello", "--project", str(temp_project)],
        )

        assert result.exit_code != 0
        assert "context gateway" in result.output.lower()

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    @patch("codeagent.interaction.cli.main._build_context_gateway")
    def test_ask_orchestrator_run_failure(
        self,
        mock_build_cg: MagicMock,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """Orchestrator.run 抛出异常时应显示错误。"""
        mock_build_cg.return_value = MagicMock()
        mock_build_llm.return_value = AsyncMock()
        mock_build_vg.return_value = MagicMock()
        mock_build_tg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = RuntimeError("Orchestrator crashed")
            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        assert result.exit_code != 0

    @patch("codeagent.interaction.cli.main._display_review_request")
    def test_orchestrator_human_review_loop_resumes(
        self,
        mock_display: MagicMock,
    ) -> None:
        """human_review_required 时应调用 orchestrator.resume()。"""
        import asyncio
        from codeagent.interaction.cli.main import _run_orchestrator_with_review

        mock_orc = MagicMock()
        mock_orc.run = AsyncMock()
        mock_orc.resume = AsyncMock()
        mock_orc.get_checkpoints = MagicMock()

        # 第一次 run 返回需要审核的状态
        review_state = _make_result_state(
            human_review_required=True,
            review_request={"title": "高风险操作", "review_type": "high_risk_plan"},
            user_request="test",
            project_root="/tmp",
        )
        # resume 后返回最终完成状态
        final_state = _make_result_state(
            human_review_required=False,
            user_request="test",
            project_root="/tmp",
        )

        mock_orc.run.return_value = review_state
        mock_orc.resume.return_value = final_state
        mock_orc.get_checkpoints.return_value = [{"thread_id": "test-thread"}]

        with patch(
            "rich.prompt.Prompt.ask",
        ) as mock_prompt_ask:
            mock_prompt_ask.return_value = "approve"
            result = asyncio.run(_run_orchestrator_with_review(
                orchestrator=mock_orc,
                request="test",
                project_root="/tmp",
                auto=False,
            ))

        # resume 应被调用一次（审核被 approve）
        mock_orc.resume.assert_awaited_once_with("test-thread", "approve")
        assert result.human_review_required is False


# ── Tests: init command ────────────────────────────────────────────────────


class TestInitCommand:
    """验证 init 命令。"""

    def test_init_creates_codeagent_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path) as td:
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(td, ".codeagent").is_dir()
            assert Path(td, ".codeagent", "config.json").is_file()

    def test_init_idempotent(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result1 = runner.invoke(cli, ["init"])
            result2 = runner.invoke(cli, ["init"])
            assert result1.exit_code == 0
            assert result2.exit_code == 0
            assert "already exists" in result2.output.lower()


# ── Tests: config command ──────────────────────────────────────────────────


class TestConfigCommand:
    """验证 config 命令。"""

    def test_config_without_init(self, runner: CliRunner, tmp_path: Path) -> None:
        """未初始化时 config 应提示。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_config_show(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "version" in data

    def test_config_set(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["config", "--set", "auto_mode", "true"])
            assert result.exit_code == 0
            # Verify it was saved — use cwd() since isolated_filesystem chdirs
            config_path = Path.cwd() / ".codeagent" / "config.json"
            assert config_path.exists()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            assert config["auto_mode"] == "true"


# ── Tests: history command ─────────────────────────────────────────────────


class TestHistoryCommand:
    """验证 history 命令。"""

    def test_history_without_init(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["history"])
            assert result.exit_code == 0
            assert "No session history" in result.output

    def test_history_with_data(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            # Create history file
            history = [
                {"timestamp": "2026-05-13T10:00:00", "request": "Test request", "success": True},
            ]
            history_path = Path.cwd() / ".codeagent" / "history.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")

            result = runner.invoke(cli, ["history"])
            assert result.exit_code == 0
            assert "Test request" in result.output

    def test_history_with_limit(self, runner: CliRunner, tmp_path: Path) -> None:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            sessions = [
                {"timestamp": f"2026-05-{d:02d}T10:00:00", "request": f"Session {d}", "success": True}
                for d in range(1, 21)
            ]
            history_path = Path.cwd() / ".codeagent" / "history.json"
            history_path.write_text(json.dumps(sessions), encoding="utf-8")

            result = runner.invoke(cli, ["history", "--limit", "5"])
            assert result.exit_code == 0


# ── Tests: _load_env_file ──────────────────────────────────────────────────


class TestLoadEnvFile:
    """验证 load_env_file 辅助函数。"""

    def test_load_env_file(self, tmp_path: Path) -> None:
        from codeagent.config import load_env_file

        env_file = tmp_path / ".env"
        env_file.write_text(
            "TEST_LLM_KEY=test-key\n"
            "TEST_LLM_MODEL=test-model\n"
            "# This is a comment\n"
            "EMPTY=\n"
        )

        load_env_file(env_file)
        assert os.environ.get("TEST_LLM_KEY") == "test-key"
        assert os.environ.get("TEST_LLM_MODEL") == "test-model"

    def test_load_env_file_not_exists(self) -> None:
        from codeagent.config import load_env_file

        load_env_file(Path("/nonexistent/.env"))  # should not raise


# ── Tests: _build_tool_gateway ──────────────────────────────────────────────


class TestBuildToolGateway:
    """验证 _build_tool_gateway 辅助函数。"""

    def test_build_tool_gateway(self, tmp_path: Path) -> None:
        from codeagent.interaction.cli.main import _build_tool_gateway

        gateway = _build_tool_gateway(str(tmp_path))
        tools = gateway.list_tools()
        tool_names = {t.name for t in tools}
        assert "read_file" in tool_names
        assert "write_file" in tool_names


# ── Tests: _build_validation_gateway ────────────────────────────────────────


class TestBuildValidationGateway:
    """验证 _build_validation_gateway 辅助函数。"""

    def test_build_validation_gateway(self) -> None:
        from codeagent.interaction.cli.main import _build_validation_gateway

        gateway = _build_validation_gateway()
        # 验证它实现了 IValidationGateway
        from codeagent.gateway.validation_gateway import IValidationGateway

        assert isinstance(gateway, IValidationGateway)


# ── Tests: _build_context_gateway ────────────────────────────────────────────


class TestBuildContextGateway:
    """验证 _build_context_gateway 辅助函数（R1 新增）。"""

    def test_build_context_gateway(self, tmp_path: Path) -> None:
        from codeagent.interaction.cli.main import _build_context_gateway

        gateway = _build_context_gateway(str(tmp_path))
        # 验证它实现了 IContextGateway
        from codeagent.gateway.context_gateway import IContextGateway

        assert isinstance(gateway, IContextGateway)

    def test_build_context_gateway_respects_budget(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from codeagent.interaction.cli.main import _build_context_gateway

        monkeypatch.setenv("CONTEXT_BUDGET_TOKENS", "16000")
        gateway = _build_context_gateway(str(tmp_path))
        assert gateway.config.total_budget == 16000


# =============================================================================
# 测试：_save_session_history 持久化
# =============================================================================


class TestSaveSessionHistory:
    """验证 _save_session_history 函数的读写行为。"""

    def test_save_new_file(self, tmp_path: Path) -> None:
        """新文件写入 — 应创建 history.json。"""
        from codeagent.interaction.cli.main import _save_session_history

        session = {"request": "test", "success": True, "timestamp": "2026-05-14T12:00:00"}
        _save_session_history(str(tmp_path), session)

        history_path = tmp_path / ".codeagent" / "history.json"
        assert history_path.exists()
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["request"] == "test"
        assert data[0]["success"] is True

    def test_save_append_to_existing(self, tmp_path: Path) -> None:
        """追加到已有文件 — 应保留历史记录。"""
        from codeagent.interaction.cli.main import _save_session_history

        s1 = {"request": "first", "timestamp": "2026-05-14T12:00:00"}
        s2 = {"request": "second", "timestamp": "2026-05-14T12:01:00"}

        _save_session_history(str(tmp_path), s1)
        _save_session_history(str(tmp_path), s2)

        history_path = tmp_path / ".codeagent" / "history.json"
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["request"] == "first"
        assert data[1]["request"] == "second"

    def test_read_with_data(self, tmp_path: Path) -> None:
        """有历史数据时应能追加。"""
        from codeagent.interaction.cli.main import _save_session_history

        sessions = [{"request": f"session_{i}", "timestamp": "2026-05-14T12:00:00"} for i in range(3)]
        for s in sessions:
            _save_session_history(str(tmp_path), s)

        history_path = tmp_path / ".codeagent" / "history.json"
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 3

    def test_save_read_with_empty_data(self, tmp_path: Path) -> None:
        """损坏的文件应优雅处理视为空。"""
        from codeagent.interaction.cli.main import _save_session_history

        history_path = tmp_path / ".codeagent" / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("invalid json content", encoding="utf-8")

        session = {"request": "after corruption", "success": True, "timestamp": "2026-05-14T12:00:00"}
        _save_session_history(str(tmp_path), session)

        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["request"] == "after corruption"

    def test_truncate_at_100(self, tmp_path: Path) -> None:
        """超过 100 条时应自动裁剪。"""
        from codeagent.interaction.cli.main import _save_session_history

        for i in range(105):
            _save_session_history(str(tmp_path), {
                "request": f"session_{i}",
                "timestamp": f"2026-05-14T12:{i:02d}:00",
            })

        history_path = tmp_path / ".codeagent" / "history.json"
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 100
        assert data[0]["request"] == "session_5"
        assert data[99]["request"] == "session_104"

    def test_empty_sessions_read(self, tmp_path: Path) -> None:
        """空的 JSON 文件应正确读取。"""
        from codeagent.interaction.cli.main import _save_session_history

        history_path = tmp_path / ".codeagent" / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("[]", encoding="utf-8")

        session = {"request": "after empty", "success": True, "timestamp": "2026-05-14T12:00:00"}
        _save_session_history(str(tmp_path), session)

        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 1


# =============================================================================
# 测试：CLI 集成 — ask 写入历史 + history 读取
# =============================================================================


class TestAskHistoryIntegration:
    """验证 ask 命令执行后自动写入历史记录。"""

    @patch("codeagent.interaction.cli.main._save_session_history")
    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_calls_save_session_history(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        mock_save: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """ask 命令应调用 _save_session_history。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            result = runner.invoke(
                cli,
                ["ask", "write a test", "--project", str(temp_project)],
            )

        assert result.exit_code == 0
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == str(temp_project)
        session_data = call_args[1]
        assert session_data["request"] == "write a test"
        assert session_data["success"] is True
        assert "timestamp" in session_data
        assert "duration_ms" in session_data

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_saves_history_on_success(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """ask 成功后应保存 history.json。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state())
            runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        history_path = Path(temp_project) / ".codeagent" / "history.json"
        assert history_path.exists()
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["request"] == "hello"
        assert data[0]["success"] is True

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_ask_saves_history_on_failure(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """ask 失败（有 errors）也应保存 history.json。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
            mock_run.side_effect = _mock_run(_make_result_state(
                errors=["Something went wrong"],
            ))
            runner.invoke(
                cli,
                ["ask", "hello", "--project", str(temp_project)],
            )

        history_path = Path(temp_project) / ".codeagent" / "history.json"
        assert history_path.exists()
        data = json.loads(history_path.read_text(encoding="utf-8"))
        assert data[0]["success"] is False
        assert data[0]["error_count"] == 1

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_history_command_shows_saved_data(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """history 命令应显示 ask 保存的数据。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        # history 命令使用 Path.cwd()，因此用 isolated_filesystem
        with runner.isolated_filesystem(temp_dir=temp_project) as td:
            (Path(td) / "test_file.py").write_text("x = 1\n")
            with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
                mock_run.side_effect = _mock_run(_make_result_state())
                runner.invoke(
                    cli,
                    ["ask", "integration test", "--project", td],
                )

            result = runner.invoke(cli, ["history"])
            assert result.exit_code == 0
            assert "integration test" in result.output

    @patch("codeagent.interaction.cli.main._build_tool_gateway")
    @patch("codeagent.interaction.cli.main._build_validation_gateway")
    @patch("codeagent.interaction.cli.main._build_llm")
    def test_history_limit_filter(
        self,
        mock_build_llm: MagicMock,
        mock_build_vg: MagicMock,
        mock_build_tg: MagicMock,
        runner: CliRunner,
        temp_project: Path,
    ) -> None:
        """history --limit 应限制显示条数。"""
        mock_build_llm.return_value = AsyncMock()
        mock_build_tg.return_value = MagicMock()
        mock_build_vg.return_value = MagicMock()

        with runner.isolated_filesystem(temp_dir=temp_project) as td:
            (Path(td) / "test_file.py").write_text("x = 1\n")
            with patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:
                mock_run.side_effect = _mock_run(_make_result_state())
                for i in range(3):
                    runner.invoke(
                        cli,
                        ["ask", f"request {i}", "--project", td],
                    )

            result = runner.invoke(cli, ["history", "--limit", "1"])
            assert result.exit_code == 0
            lines = [l for l in result.output.split("\n") if "request 2" in l.strip()]
            assert len(lines) == 1
