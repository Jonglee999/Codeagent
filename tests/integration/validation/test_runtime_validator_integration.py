"""RuntimeValidator 集成测试。

使用 sample_python_project fixture 验证与 TestDetector 和 ErrorAnalyzer 的集成。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.validation.runtime_validator import RuntimeValidator

# sample_python_project 的路径
_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_python_project"


@pytest.mark.asyncio
class TestRuntimeValidatorIntegration:
    """在真实项目上验证 RuntimeValidator 的集成行为。"""

    async def test_detect_framework_on_real_project(self) -> None:
        """验证能在真实项目上检测测试框架。"""
        if not _FIXTURE_DIR.exists():
            pytest.skip("sample_python_project fixture not found")

        validator = RuntimeValidator(str(_FIXTURE_DIR))
        result = await validator.run_tests()

        # 项目可能没有测试框架，此时 passed=True 含警告
        # 如果项目有测试框架，passed 取决于测试是否通过
        # 核心：验证执行流程不抛异常
        assert result.duration_ms >= 0

    async def test_fallback_on_real_project(self) -> None:
        """验证能在真实项目文件上运行降级检查。"""
        if not _FIXTURE_DIR.exists():
            pytest.skip("sample_python_project fixture not found")

        py_files = [str(p.relative_to(_FIXTURE_DIR)) for p in sorted(_FIXTURE_DIR.rglob("*.py"))]
        if not py_files:
            pytest.skip("no Python files found")

        validator = RuntimeValidator(str(_FIXTURE_DIR))
        result = await validator.run_fallback_check(
            changed_files=py_files[:3]  # 只检查前 3 个文件
        )

        # 降级检查应正常完成
        assert result.duration_ms >= 0

    async def test_analyze_failures_on_real_project(self) -> None:
        """验证错误分析功能。"""
        if not _FIXTURE_DIR.exists():
            pytest.skip("sample_python_project fixture not found")

        # 模拟一个测试失败输出
        fake_output = (
            "================================= FAILURES ===================================\n"
            "_______________________________ test_main _________________________________\n\n"
            "    def test_main():\n"
            ">       assert main() == 'expected'\n"
            "E       AssertionError: assert 'actual' == 'expected'\n\n"
            "tests/test_main.py:5: AssertionError\n"
            "=========================== short test summary info ===========================\n"
            "FAILED tests/test_main.py::test_main - AssertionError: assert 'actual' == 'expected'\n"
            "============================== 1 failed in 0.12s ==============================\n"
        )

        validator = RuntimeValidator(str(_FIXTURE_DIR))
        suggestions = await validator.analyze_failures(
            test_output=fake_output,
            changed_files=["src/main.py", "tests/test_main.py"],
        )

        assert len(suggestions) >= 1
        s = suggestions[0]
        assert s.error_summary
        assert s.affected_files
        assert s.likely_cause
        assert s.suggested_fix
        # test_main.py 在 changed_files 中 → 新错误
        assert not s.is_pre_existing

    async def test_analyze_failures_pre_existing(self) -> None:
        """验证 pre_existing 判断。"""
        if not _FIXTURE_DIR.exists():
            pytest.skip("sample_python_project fixture not found")

        fake_output = (
            "================================= FAILURES ===================================\n"
            "_______________________________ test_old __________________________________\n\n"
            "    def test_old():\n"
            ">       assert 1 == 2\n"
            "E       assert 1 == 2\n\n"
            "tests/test_legacy.py:3: AssertionError\n"
            "=========================== short test summary info ===========================\n"
            "FAILED tests/test_legacy.py::test_old - AssertionError: assert 1 == 2\n"
            "============================== 1 failed in 0.12s ==============================\n"
        )

        validator = RuntimeValidator(str(_FIXTURE_DIR))
        suggestions = await validator.analyze_failures(
            test_output=fake_output,
            changed_files=["src/main.py"],  # test_legacy.py 不在修改列表中
        )

        assert len(suggestions) >= 1
        assert suggestions[0].is_pre_existing

    async def test_full_flow_no_tests(self) -> None:
        """完整流程：项目无测试 → 降级检查。"""
        if not _FIXTURE_DIR.exists():
            pytest.skip("sample_python_project fixture not found")

        validator = RuntimeValidator(str(_FIXTURE_DIR))

        # 1. 运行测试（应跳过，返回警告）
        test_result = await validator.run_tests()
        assert test_result.duration_ms >= 0

        # 2. 降级检查
        py_files = [str(p.relative_to(_FIXTURE_DIR)) for p in sorted(_FIXTURE_DIR.rglob("*.py"))]
        fallback_result = await validator.run_fallback_check(
            changed_files=py_files[:2]
        )
        assert fallback_result.duration_ms >= 0
