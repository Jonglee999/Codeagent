"""全局错误处理测试 — CLI/系统级别的异常恢复场景。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from codeagent.interaction.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


class TestGlobalNetworkErrors:
    """网络相关全局错误处理。"""

    def test_llm_network_timeout(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """LLM 调用网络超时 — 应显示超时错误而非崩溃。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()

            # 模拟网络超时
            mock_run.side_effect = TimeoutError("LLM API timed out after 60s")

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code != 0
        assert "timeout" in result.output.lower() or "timed out" in result.output.lower()

    def test_llm_api_key_expired(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """API Key 过期 — 应显示认证错误。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()

            # 模拟 API key 认证失败
            mock_run.side_effect = Exception("Authentication failed: API key invalid or expired")

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "authentication" in output_lower or "api key" in output_lower or "auth" in output_lower

    def test_llm_rate_limit(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """API 限流 — 应显示限流错误。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()
            mock_run.side_effect = Exception("Rate limit exceeded")

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code != 0
        assert "rate limit" in result.output.lower()

    def test_ask_with_network_error_json_output(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """网络错误 + JSON 输出模式 — 应输出有效 JSON。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()
            mock_run.side_effect = Exception("Connection refused")

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path), "--json"],
            )

        # JSON 模式下错误信息编码在响应体内
        data = json.loads(result.output)
        assert data["success"] is False

    def test_build_llm_import_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """litellm 未安装时 _build_llm 抛出 ImportError。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()

            # _build_llm 内部的 ImportError 会被捕获并转为 ClickException
            with patch(
                "codeagent.interaction.cli.main._build_llm",
                side_effect=Exception("No module named 'litellm'"),
            ):
                result = runner.invoke(
                    cli,
                    ["ask", "hello", "--project", str(tmp_path)],
                )

        assert result.exit_code != 0


class TestGlobalToolErrors:
    """工具构建全局错误处理。"""

    def test_tool_gateway_build_failure(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """工具 Gateway 构建失败 — 应显示构建错误。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch(
            "codeagent.interaction.cli.main._build_tool_gateway",
            side_effect=OSError("Permission denied"),
        ), \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm:

            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code != 0

    def test_validation_gateway_build_failure(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """验证 Gateway 构建失败 — 应显示构建错误。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch(
                 "codeagent.interaction.cli.main._build_validation_gateway",
                 side_effect=Exception("Validator init failed"),
             ), \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm:

            mock_btg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code != 0


class TestGlobalConfigErrors:
    """配置相关全局错误处理。"""

    def test_config_file_corrupted_json(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """配置文件损坏（无效 JSON）— 不应崩溃。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # 初始化
            runner.invoke(cli, ["init"])

            # 写入损坏的配置
            config_path = Path.cwd() / ".codeagent" / "config.json"
            config_path.write_text("{invalid json content", encoding="utf-8")

            # config --show 应处理 JSON 解码错误
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            assert "error" in result.output.lower() or "Error" in result.output

    def test_config_file_unreadable(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """配置文件不可读 — 应安全降级。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            # 删除配置文件
            config_path = Path.cwd() / ".codeagent" / "config.json"
            config_path.unlink()

            # config --show 应在文件缺失时提示
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_history_file_corrupted(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """历史记录文件损坏 — 不应崩溃。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])

            history_path = Path.cwd() / ".codeagent" / "history.json"
            history_path.write_text("[invalid json", encoding="utf-8")

            result = runner.invoke(cli, ["history"])
            assert result.exit_code == 0
            assert "No session history" in result.output

    def test_codeagent_dir_missing_for_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """.codeagent 目录完全缺失 — config 应安全提示。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # 不执行 init
            result = runner.invoke(cli, ["config", "--show"])
            assert result.exit_code == 0
            assert "not found" in result.output.lower() or "init" in result.output.lower()

    def test_codeagent_dir_missing_for_history(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """.codeagent 目录完全缺失 — history 应安全提示。"""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["history"])
            assert result.exit_code == 0
            assert "No session history" in result.output


class TestGlobalEdgeCases:
    """其他全局边缘情况。"""

    def test_project_root_not_exist(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """不存在的项目根目录 — 应显示明确错误。"""
        result = runner.invoke(
            cli,
            ["ask", "hello", "--project", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_empty_request(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """空请求 — 应显示使用错误。"""
        result = runner.invoke(
            cli,
            ["ask", "--project", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_unexpected_keyboard_interrupt(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """执行中突发 KeyboardInterrupt — 不会留下挂起的协程。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path)],
            )

        assert result.exit_code in (0, 1)

    def test_keyboard_interrupt_with_json_output(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """KeyboardInterrupt + JSON 输出 — 应输出有效 JSON。"""
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        with patch("codeagent.interaction.cli.main._build_tool_gateway") as mock_btg, \
             patch("codeagent.interaction.cli.main._build_validation_gateway") as mock_bvg, \
             patch("codeagent.interaction.cli.main._build_llm") as mock_blm, \
             patch("codeagent.interaction.cli.main.asyncio.run") as mock_run:

            mock_btg.return_value = MagicMock()
            mock_bvg.return_value = MagicMock()
            mock_blm.return_value = AsyncMock()
            mock_run.side_effect = KeyboardInterrupt()

            result = runner.invoke(
                cli,
                ["ask", "hello", "--project", str(tmp_path), "--json"],
            )

        assert result.exit_code in (0, 1)
        output = result.output.strip()
        if output:
            data = json.loads(output)
            assert data["success"] is False
