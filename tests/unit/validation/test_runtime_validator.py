"""RuntimeValidator 单元测试。

覆盖测试执行、降级检查、失败分析、超时处理、边界情况。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from codeagent.gateway.validation_gateway import ValidationResult
from codeagent.validation.error_analyzer import FixSuggestion
from codeagent.validation.runtime_validator import RuntimeValidator
from codeagent.validation.test_detector import TestFrameworkInfo


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_test_detector(mocker) -> MagicMock:
    """Mock TestDetector with controlled detect() return value."""
    mock = mocker.patch(
        "codeagent.validation.runtime_validator.TestDetector",
        autospec=True,
    )
    instance = mock.return_value
    instance.detect = AsyncMock()
    return instance


def _make_framework_info(
    framework: str | None = "pytest",
    config_files: list[str] | None = None,
    test_dirs: list[str] | None = None,
    has_asyncio: bool = False,
    has_cov: bool = False,
    test_command: str = "pytest -x --tb=short",
) -> TestFrameworkInfo:
    """创建 TestFrameworkInfo 测试实例。"""
    return TestFrameworkInfo(
        framework=framework,
        config_files=config_files or [],
        test_dirs=test_dirs or [],
        has_asyncio_support=has_asyncio,
        has_cov_support=has_cov,
        test_command=test_command,
        installed_packages=["pytest"],
    )


# ── 测试执行 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRunTests:
    """测试执行能力。"""

    async def test_run_tests_passed(
        self, mock_test_detector, mocker
    ) -> None:
        """测试全部通过。"""
        mock_test_detector.detect.return_value = _make_framework_info()

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "=== 3 passed in 0.10s ==="
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is True
        assert len(result.errors) == 0
        assert result.duration_ms >= 0

    async def test_run_tests_failed(
        self, mock_test_detector, mocker
    ) -> None:
        """测试有失败。"""
        mock_test_detector.detect.return_value = _make_framework_info()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = (
            "================================= FAILURES ===================================\n"
            "_______________________________ test_fail _________________________________\n\n"
            "    def test_fail():\n"
            ">       assert 1 == 2\n"
            "E       assert 1 == 2\n\n"
            "tests/test_sample.py:3: AssertionError\n"
            "=========================== short test summary info ===========================\n"
            "FAILED tests/test_sample.py::test_fail - AssertionError: assert 1 == 2\n"
            "============================== 1 failed in 0.12s ==============================\n"
        )
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is False
        assert len(result.errors) >= 1
        assert any(
            "AssertionError" in e.message for e in result.errors
        )

    async def test_run_tests_no_framework(
        self, mock_test_detector,
    ) -> None:
        """无测试框架时跳过测试。"""
        mock_test_detector.detect.return_value = _make_framework_info(
            framework=None
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is True
        assert len(result.warnings) >= 1
        assert any(
            "NO_TEST_FRAMEWORK" in w.code for w in result.warnings
        )

    async def test_run_tests_timeout(
        self, mock_test_detector, mocker
    ) -> None:
        """测试执行超时。"""
        mock_test_detector.detect.return_value = _make_framework_info()

        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=5),
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is False
        assert any("TIMEOUT" in e.code for e in result.errors)

    async def test_run_tests_command_not_found(
        self, mock_test_detector, mocker
    ) -> None:
        """测试命令未找到。"""
        mock_test_detector.detect.return_value = _make_framework_info()

        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            side_effect=FileNotFoundError(),
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is False
        assert any(
            "COMMAND_NOT_FOUND" in e.code for e in result.errors
        )

    async def test_run_tests_empty_command(
        self, mock_test_detector,
    ) -> None:
        """空测试命令时跳过。"""
        mock_test_detector.detect.return_value = _make_framework_info(
            test_command=""
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.passed is True
        assert len(result.warnings) >= 1


# ── 降级检查 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestFallbackCheck:
    """降级检查测试。"""

    async def test_fallback_script_mode(self, tmp_path, mocker) -> None:
        """脚本模式：有 __name__ == "__main__"，运行成功。"""
        project = tmp_path / "project"
        project.mkdir()
        file_path = project / "script.py"
        file_path.write_text(
            'def main():\n    print("ok")\n\nif __name__ == "__main__":\n    main()\n'
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "ok\n"
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["script.py"]
        )

        assert result.passed is True

    async def test_fallback_script_fails(self, tmp_path, mocker) -> None:
        """脚本模式：运行失败。"""
        project = tmp_path / "project"
        project.mkdir()
        file_path = project / "failing_script.py"
        file_path.write_text(
            'def main():\n    raise RuntimeError("fail")\n\n'
            'if __name__ == "__main__":\n    main()\n'
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "RuntimeError: fail\n"
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["failing_script.py"]
        )

        assert result.passed is False
        assert any(
            "FALLBACK_SCRIPT_ERROR" in e.code
            for e in result.errors
        )

    async def test_fallback_import_mode(self, tmp_path, mocker) -> None:
        """导入模式：模块导入成功。"""
        project = tmp_path / "project"
        project.mkdir()
        (project / "mymodule.py").write_text("VERSION = '1.0'\n")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["mymodule.py"]
        )

        assert result.passed is True

    async def test_fallback_import_fails(self, tmp_path, mocker) -> None:
        """导入模式：模块导入失败。"""
        project = tmp_path / "project"
        project.mkdir()
        (project / "broken_module.py").write_text(
            "import nonexistent_package_xyz\n"
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "ModuleNotFoundError: No module named 'nonexistent_package_xyz'\n"
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["broken_module.py"]
        )

        assert result.passed is False
        assert any(
            "FALLBACK_IMPORT_ERROR" in e.code
            for e in result.errors
        )

    async def test_fallback_main_entry(self, tmp_path, mocker) -> None:
        """主入口模式：运行 --help 成功。"""
        project = tmp_path / "project"
        project.mkdir()
        file_path = project / "cli_app.py"
        file_path.write_text(
            'def main():\n    """CLI app."""\n    pass\n\n'
            'if __name__ == "__main__":\n    main()\n'
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "usage: cli_app.py [--help]\n"
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["cli_app.py"]
        )

        assert result.passed is True

    async def test_fallback_no_library(
        self, tmp_path, mocker
    ) -> None:
        """库文件：无 __name__ == "__main__"，运行导入检查。"""
        project = tmp_path / "project"
        project.mkdir()
        (project / "lib.py").write_text("def helper():\n    return 42\n")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["lib.py"]
        )

        assert result.passed is True

    async def test_fallback_not_applicable(
        self, tmp_path, mocker
    ) -> None:
        """无适用的检查时返回警告。"""
        project = tmp_path / "project"
        project.mkdir()
        (project / "data.json").write_text('{"key": "value"}\n')

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["data.json"]
        )

        assert result.passed is True
        assert any(
            "NO_RUNTIME_CHECK" in w.code for w in result.warnings
        )

    async def test_fallback_non_python_file(
        self, tmp_path, mocker
    ) -> None:
        """非 Python 文件跳过。"""
        project = tmp_path / "project"
        project.mkdir()
        (project / "readme.md").write_text("# Documentation\n")

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["readme.md"]
        )

        assert result.passed is True
        assert any(
            "NO_RUNTIME_CHECK" in w.code for w in result.warnings
        )


# ── 失败分析 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAnalyzeFailures:
    """失败分析测试。"""

    async def test_analyze_failures_delegates_to_error_analyzer(
        self, mocker
    ) -> None:
        """analyze_failures 委托给 ErrorAnalyzer。"""
        mock_analyze = mocker.patch(
            "codeagent.validation.runtime_validator.ErrorAnalyzer.analyze",
            return_value=[
                FixSuggestion(
                    error_summary="AssertionError: assert 1 == 2",
                    affected_files=["tests/test_sample.py"],
                    likely_cause="Test assertion failed",
                    suggested_fix="Review assertion logic",
                )
            ],
        )

        validator = RuntimeValidator("/tmp/project")
        suggestions = await validator.analyze_failures(
            test_output="some output",
            changed_files=["tests/test_sample.py"],
        )

        assert len(suggestions) == 1
        assert suggestions[0].error_summary == "AssertionError: assert 1 == 2"
        mock_analyze.assert_called_once_with(
            "some output", ["tests/test_sample.py"]
        )

    async def test_analyze_failures_no_failures(self, mocker) -> None:
        """无失败时返回空列表。"""
        mocker.patch(
            "codeagent.validation.runtime_validator.ErrorAnalyzer.analyze",
            return_value=[],
        )

        validator = RuntimeValidator("/tmp/project")
        suggestions = await validator.analyze_failures(
            test_output="3 passed in 0.10s",
            changed_files=[],
        )

        assert suggestions == []


# ── 边界情况 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    """边界情况测试。"""

    async def test_nonexistent_project_root(self, mocker) -> None:
        """项目根目录不存在。"""
        mock_detect = mocker.patch(
            "codeagent.validation.runtime_validator.TestDetector.detect",
            new_callable=AsyncMock,
        )
        mock_detect.return_value = _make_framework_info(framework=None)

        validator = RuntimeValidator("/nonexistent/path")
        result = await validator.run_tests()

        assert result.passed is True
        assert any(
            "NO_TEST_FRAMEWORK" in w.code for w in result.warnings
        )

    async def test_run_tests_duration_positive(
        self, mock_test_detector, mocker
    ) -> None:
        """验证 duration_ms 为正数。"""
        mock_test_detector.detect.return_value = _make_framework_info()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "=== 1 passed in 0.05s ==="
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator("/tmp/project")
        result = await validator.run_tests()

        assert result.duration_ms >= 0

    async def test_run_fallback_check_no_files(
        self, tmp_path
    ) -> None:
        """空文件列表。"""
        project = tmp_path / "project"
        project.mkdir()

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(changed_files=[])

        assert result.passed is True
        assert any(
            "NO_RUNTIME_CHECK" in w.code for w in result.warnings
        )

    async def test_fallback_with_nested_module(
        self, tmp_path, mocker
    ) -> None:
        """嵌套模块（src/tests/ 结构）。"""
        project = tmp_path / "project"
        (project / "src" / "tests").mkdir(parents=True)
        (project / "src" / "tests" / "test_helper.py").write_text(
            "def test_x(): pass\n"
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mocker.patch(
            "codeagent.validation.runtime_validator._run_subprocess",
            return_value=mock_proc,
        )

        validator = RuntimeValidator(str(project))
        result = await validator.run_fallback_check(
            changed_files=["src/tests/test_helper.py"]
        )

        assert result.passed is True
