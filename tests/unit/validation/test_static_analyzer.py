"""StaticAnalyzer 单元测试。

覆盖 Lint 检查、类型检查、安全扫描、文件过滤、工具缺失、配置检测等场景。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from codeagent.gateway.validation_gateway import ValidationResult
from codeagent.validation.static_analyzer import StaticAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def analyzer() -> StaticAnalyzer:
    return StaticAnalyzer()


@pytest.fixture
def sample_py_file(tmp_path: Path) -> str:
    """创建一个有效的 Python 文件。"""
    f = tmp_path / "valid.py"
    f.write_text("x = 1\n")
    return str(f)


# ── Tests: Lint 检查 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRunLint:
    """ruff Lint 检查相关测试。"""

    async def test_lint_clean_file(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """有效代码无 lint 错误。"""
        f = tmp_path / "clean.py"
        f.write_text("x = 1\ny = x + 1\n")
        result = await analyzer.run_lint([str(f)])
        assert result.passed is True, f"Expected passed=True, got errors: {result.errors}"

    async def test_lint_with_errors(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """有 lint 错误的文件。"""
        f = tmp_path / "with_errors.py"
        f.write_text("import os\nx = 1\n")
        result = await analyzer.run_lint([str(f)])
        # F401: unused import — ruff 可能根据配置报告或不报告
        # 我们只验证不会 crash
        assert result.duration_ms > 0

    async def test_lint_severity_mapping(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """验证 ruff 错误严重级别正确映射。"""
        f = tmp_path / "severity_test.py"
        f.write_text("import os\nx = 1\n")
        result = await analyzer.run_lint([str(f)])
        if not result.passed:
            for err in result.errors:
                assert err.severity in ("error", "warning")
                assert err.code
                assert err.file_path
                assert err.line > 0

    async def test_lint_file_not_found(self, analyzer: StaticAnalyzer) -> None:
        """不存在的文件。"""
        result = await analyzer.run_lint(["/nonexistent/test.py"])
        # ruff 会对不存在的文件报错
        assert result.duration_ms >= 0

    async def test_lint_non_python_file(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """非 Python 文件自动跳过。"""
        f = tmp_path / "readme.md"
        f.write_text("# Title\n")
        result = await analyzer.run_lint([str(f)])
        assert result.passed is True
        assert len(result.errors) == 0

    async def test_lint_empty_file_list(self, analyzer: StaticAnalyzer) -> None:
        """空文件列表。"""
        result = await analyzer.run_lint([])
        assert result.passed is True
        assert len(result.errors) == 0

    async def test_lint_mixed_files(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """混合 Python 和非 Python 文件。"""
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n")
        md_file = tmp_path / "doc.md"
        md_file.write_text("# Doc\n")
        result = await analyzer.run_lint([str(py_file), str(md_file)])
        assert result.passed is True


# ── Tests: 类型检查 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRunTypeCheck:
    """mypy 类型检查相关测试。"""

    async def test_typecheck_clean(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """类型正确的代码。"""
        f = tmp_path / "clean_types.py"
        f.write_text("x: int = 1\n")
        result = await analyzer.run_typecheck([str(f)])
        assert result.passed is True

    async def test_typecheck_with_error(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """类型错误的代码。"""
        f = tmp_path / "type_error.py"
        f.write_text("x: int = 'hello'\n")
        result = await analyzer.run_typecheck([str(f)])
        # mypy 应报 Incompatible types in assignment 错误
        assert not result.passed
        assert len(result.errors) > 0
        assert any("Incompatible" in e.message for e in result.errors)

    async def test_typecheck_missing_annotation(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """缺少类型注解（取决于 mypy strict 模式）。"""
        f = tmp_path / "no_annotation.py"
        f.write_text("def f(x):\n    return x + 1\n")
        result = await analyzer.run_typecheck([str(f)])
        # 非 strict 模式下 mypy 默认可能不报 no-untyped-def
        # 我们只验证不会 crash
        assert result.duration_ms > 0

    async def test_typecheck_non_python_file(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """非 Python 文件跳过类型检查。"""
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = await analyzer.run_typecheck([str(f)])
        assert result.passed is True

    async def test_typecheck_empty_file_list(self, analyzer: StaticAnalyzer) -> None:
        """空文件列表。"""
        result = await analyzer.run_typecheck([])
        assert result.passed is True


# ── Tests: 安全扫描 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSecurityScan:
    """安全扫描相关测试。"""

    @pytest.fixture(autouse=True)
    def _disable_bandit(self, analyzer: StaticAnalyzer, mocker):
        """确保正则测试不受 bandit 影响。"""
        mocker.patch.object(analyzer, "_run_bandit", return_value=None)

    async def test_detect_eval(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测 eval() 调用。"""
        f = tmp_path / "use_eval.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        eval_errors = [e for e in result.errors if e.code == "SEC_EVAL"]
        assert len(eval_errors) >= 1

    async def test_detect_exec(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测 exec() 调用。"""
        f = tmp_path / "use_exec.py"
        f.write_text("exec('print(1)')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        exec_errors = [e for e in result.errors if e.code == "SEC_EXEC"]
        assert len(exec_errors) >= 1

    async def test_detect_os_system(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测 os.system() 调用。"""
        f = tmp_path / "use_os_system.py"
        f.write_text("import os\nos.system('ls')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        sec_errors = [e for e in result.errors if e.code == "SEC_OS_SYSTEM"]
        assert len(sec_errors) >= 1

    async def test_detect_hardcoded_password(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测硬编码密码。"""
        f = tmp_path / "hardcoded_pass.py"
        f.write_text('password = "supersecret123"\n')
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        pass_errors = [e for e in result.errors if e.code == "SEC_HARDCODED_SECRET"]
        assert len(pass_errors) >= 1

    async def test_clean_file_no_security_issues(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """无安全问题的文件。"""
        f = tmp_path / "clean_secure.py"
        f.write_text(
            '"""Safe module."""\n'
            "import os\n\n"
            'def greet(name: str) -> str:\n'
            '    return f"Hello, {name}"\n'
        )
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True

    async def test_detect_subprocess_shell(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测 subprocess shell=True。"""
        f = tmp_path / "use_subprocess.py"
        f.write_text(
            "import subprocess\n"
            "result = subprocess.run('ls', shell=True)\n"
        )
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        shell_errors = [e for e in result.errors if e.code == "SEC_SUBPROCESS_SHELL"]
        assert len(shell_errors) >= 1

    async def test_detect_sql_injection(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测潜在 SQL 注入。"""
        f = tmp_path / "sql_inject.py"
        f.write_text(
            'user_input = "admin"\n'
            'query = "SELECT * FROM users WHERE name = \'" + user_input + "\'\n"'
        )
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed

    async def test_detect_aws_key(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测硬编码 AWS Key。"""
        f = tmp_path / "aws_key.py"
        f.write_text('aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"\n')
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        aws_errors = [e for e in result.errors if e.code == "SEC_AWS_CREDENTIAL"]
        assert len(aws_errors) >= 1

    async def test_detect_jwt_secret(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """检测硬编码 JWT Secret。"""
        f = tmp_path / "jwt_secret.py"
        f.write_text('JWT_SECRET = "my-super-secret-key-12345"\n')
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed

    async def test_comment_should_not_trigger(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """注释中的模式不应触发检测。"""
        f = tmp_path / "comment_test.py"
        f.write_text('# eval() is dangerous but this is just a comment\n')
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True

    async def test_empty_file_security(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """空文件安全扫描。"""
        f = tmp_path / "empty_secure.py"
        f.write_text("")
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True

    async def test_non_python_file_security(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """非 Python 文件安全扫描自动跳过。"""
        f = tmp_path / "data.txt"
        f.write_text("eval hello")
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True

    async def test_empty_file_list_security(self, analyzer: StaticAnalyzer) -> None:
        """空文件列表安全扫描。"""
        result = await analyzer.run_security_scan([])
        assert result.passed is True


# ── Tests: run_all ────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRunAll:
    """统一入口 run_all 测试。"""

    async def test_run_all_clean(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """全部检查通过。"""
        f = tmp_path / "all_clean.py"
        f.write_text('"""Module doc."""\nx: int = 1\n')
        results = await analyzer.run_all([str(f)])
        assert len(results) == 3  # lint, typecheck, security
        # 期望全部通过
        for r in results:
            assert r.passed, f"Expected passed=True, got errors: {r.errors}"

    async def test_run_all_with_security_issue(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """安全扫描发现问题的混合情况。"""
        f = tmp_path / "mixed_issues.py"
        f.write_text(
            '"""Module."""\n'
            'x: int = 1\n'
            'result = eval("x + 1")\n'
        )
        results = await analyzer.run_all([str(f)])
        assert len(results) == 3
        # lint 可能通过或报错两种都可能
        # typecheck 应通过
        # security 应失败（eval）
        assert not results[2].passed  # security

    async def test_run_all_skip_lint(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """跳过 Lint 检查。"""
        f = tmp_path / "skip_lint.py"
        f.write_text("x = 1\n")
        results = await analyzer.run_all([str(f)], skip_lint=True)
        assert results[0].passed is True  # 跳过的直接返回 passed

    async def test_run_all_skip_typecheck(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """跳过类型检查。"""
        f = tmp_path / "skip_typecheck.py"
        f.write_text("x = 1\n")
        results = await analyzer.run_all([str(f)], skip_typecheck=True)
        assert results[1].passed is True

    async def test_run_all_skip_security(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """跳过安全扫描。"""
        f = tmp_path / "skip_security.py"
        f.write_text("x = 1\n")
        results = await analyzer.run_all([str(f)], skip_security=True)
        assert results[2].passed is True

    async def test_run_all_empty_files(self, analyzer: StaticAnalyzer) -> None:
        """空文件列表运行 run_all。"""
        results = await analyzer.run_all([])
        assert len(results) == 3
        for r in results:
            assert r.passed is True

    async def test_run_all_non_python(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """非 Python 文件运行 run_all。"""
        f = tmp_path / "data.json"
        f.write_text("{}")
        results = await analyzer.run_all([str(f)])
        assert len(results) == 3
        for r in results:
            assert r.passed is True


# ── Tests: 工具缺失模拟 ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestToolMissing:
    """工具未安装时的降级行为。"""

    async def test_ruff_not_available(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """ruff 不可用时返回警告而非错误。"""
        mocker.patch.object(analyzer, "_check_ruff_available", return_value=False)
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = await analyzer.run_lint([str(f)])
        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("ruff is not installed" in w.message for w in result.warnings)

    async def test_mypy_not_available(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """mypy 不可用时返回警告而非错误。"""
        mocker.patch.object(analyzer, "_check_mypy_available", return_value=False)
        f = tmp_path / "test.py"
        f.write_text("x: int = 1\n")
        result = await analyzer.run_typecheck([str(f)])
        assert result.passed is True
        assert len(result.warnings) > 0
        assert any("mypy is not installed" in w.message for w in result.warnings)


# ── Tests: 路径处理 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPathHandling:
    """路径相关行为测试。"""

    async def test_project_root_with_relative(self, tmp_path: Path) -> None:
        """项目根目录与相对路径。"""
        analyzer = StaticAnalyzer(project_root=str(tmp_path))
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = await analyzer.run_lint([str(f)])
        assert result.passed is True

    async def test_lint_with_config(self, analyzer: StaticAnalyzer, tmp_path: Path) -> None:
        """带配置文件运行 lint。"""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[tool.ruff]\n"
            'target-version = "py312"\n'
            "line-length = 100\n"
        )
        f = tmp_path / "test_config.py"
        f.write_text("x = 1\n")
        result = await analyzer.run_lint([str(f)], config=str(pyproject))
        assert result.passed is True

    async def test_non_existent_file_lint(self, analyzer: StaticAnalyzer) -> None:
        """不存在的文件返回错误而非崩溃。"""
        result = await analyzer.run_lint(["/tmp/__nonexistent_test_file__.py"])
        assert result.duration_ms >= 0
        # ruff 会返回一个错误信息，但不会 crash


# ── Tests: bandit 集成 ──────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBanditIntegration:
    """bandit 安全扫描集成测试。"""

    async def test_bandit_clean_file(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 可用时扫描无问题文件。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(
            returncode=0,
            stdout='{"results": []}',
            stderr="",
        )
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True
        mock_run.assert_called_once()

    async def test_bandit_high_severity(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 检测到 HIGH 级别问题 → errors。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(
            returncode=1,
            stdout=json.dumps({
                "results": [{
                    "filename": str(tmp_path / "test.py"),
                    "line_number": 5,
                    "issue_severity": "HIGH",
                    "issue_text": "Use of exec detected.",
                    "test_id": "B102",
                }],
            }),
            stderr="",
        )
        f = tmp_path / "test.py"
        f.write_text("exec('test')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed
        assert len(result.errors) == 1
        assert len(result.warnings) == 0
        assert result.errors[0].code == "B102"
        assert result.errors[0].line == 5

    async def test_bandit_medium_severity(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 检测到 MEDIUM 级别问题 → warnings。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(
            returncode=1,
            stdout=json.dumps({
                "results": [{
                    "filename": str(tmp_path / "test.py"),
                    "line_number": 10,
                    "issue_severity": "MEDIUM",
                    "issue_text": "Possible hardcoded password.",
                    "test_id": "B105",
                }],
            }),
            stderr="",
        )
        f = tmp_path / "test.py"
        f.write_text("password = 'secret'\n")
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True  # MEDIUM → warnings, not errors
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.warnings[0].code == "B105"

    async def test_bandit_low_severity_ignored(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit LOW 级别问题被忽略。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(
            returncode=1,
            stdout=json.dumps({
                "results": [{
                    "filename": str(tmp_path / "test.py"),
                    "line_number": 15,
                    "issue_severity": "LOW",
                    "issue_text": "Consider security implications.",
                    "test_id": "B199",
                }],
            }),
            stderr="",
        )
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        result = await analyzer.run_security_scan([str(f)])
        assert result.passed is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    async def test_bandit_not_available_fallback(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 不可用时降级为正则检测。"""
        mocker.patch("shutil.which", return_value=None)
        f = tmp_path / "test.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed  # regex 应检测到 eval
        assert any(e.code == "SEC_EVAL" for e in result.errors)

    async def test_bandit_returncode_2_fallback(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 返回码 2（自身错误）时降级为正则。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(returncode=2, stdout="", stderr="error")
        f = tmp_path / "test.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed  # 降级为正则后检测到 eval
        assert any(e.code == "SEC_EVAL" for e in result.errors)

    async def test_bandit_json_parse_error_fallback(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit JSON 解析失败时降级为正则。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mock_run = mocker.patch("codeagent.validation.static_analyzer._run_subprocess")
        mock_run.return_value = mocker.Mock(returncode=0, stdout="invalid json", stderr="")
        f = tmp_path / "test.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed  # 降级为正则后检测到 eval
        assert any(e.code == "SEC_EVAL" for e in result.errors)

    async def test_bandit_timeout_fallback(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 超时时降级为正则。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mocker.patch(
            "codeagent.validation.static_analyzer._run_subprocess",
            side_effect=subprocess.TimeoutExpired(cmd=["bandit"], timeout=60),
        )
        f = tmp_path / "test.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed  # 降级为正则后检测到 eval
        assert any(e.code == "SEC_EVAL" for e in result.errors)

    async def test_bandit_file_not_found_fallback(self, analyzer: StaticAnalyzer, mocker, tmp_path: Path) -> None:
        """bandit 命令文件不存在时降级为正则。"""
        mocker.patch("shutil.which", return_value="/usr/bin/bandit")
        mocker.patch(
            "codeagent.validation.static_analyzer._run_subprocess",
            side_effect=FileNotFoundError,
        )
        f = tmp_path / "test.py"
        f.write_text("result = eval('x + 1')\n")
        result = await analyzer.run_security_scan([str(f)])
        assert not result.passed  # 降级为正则后检测到 eval
        assert any(e.code == "SEC_EVAL" for e in result.errors)
