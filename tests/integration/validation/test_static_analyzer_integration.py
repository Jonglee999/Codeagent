"""StaticAnalyzer 集成测试。

使用 tests/fixtures/sample_python_project/ 在真实项目上运行
Lint、类型检查和安全扫描，验证与真实工具的交互正确性。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.validation.static_analyzer import StaticAnalyzer

# sample_python_project 的路径
_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "sample_python_project"


def _get_py_files() -> list[str]:
    """获取 sample_python_project 下所有 Python 文件。"""
    if not _FIXTURE_DIR.exists():
        return []
    return [str(p) for p in sorted(_FIXTURE_DIR.rglob("*.py"))]


@pytest.mark.asyncio
class TestStaticAnalyzerIntegration:
    """在真实项目上验证 StaticAnalyzer 的集成行为。"""

    async def test_lint_real_project(self) -> None:
        """在 sample_python_project 上运行 ruff Lint 检查。"""
        py_files = _get_py_files()
        if not py_files:
            pytest.skip("sample_python_project fixture not found")

        analyzer = StaticAnalyzer(project_root=str(_FIXTURE_DIR))
        result = await analyzer.run_lint(py_files)

        # 验证返回结构正确
        assert result.duration_ms > 0
        # 可以 passed 或 not（取决于项目代码质量）
        # 核心：验证解析逻辑正确
        for err in result.errors:
            assert err.code, f"Error missing code: {err}"
            assert err.message, f"Error missing message: {err}"
            assert err.file_path, f"Error missing file_path: {err}"

    async def test_typecheck_real_project(self) -> None:
        """在 sample_python_project 上运行 mypy 类型检查。"""
        py_files = _get_py_files()
        if not py_files:
            pytest.skip("sample_python_project fixture not found")

        analyzer = StaticAnalyzer(project_root=str(_FIXTURE_DIR))
        result = await analyzer.run_typecheck(py_files)

        # 验证返回结构正确
        assert result.duration_ms > 0
        for err in result.errors:
            assert err.code, f"Error missing code: {err}"
            assert err.file_path, f"Error missing file_path: {err}"

    async def test_security_scan_real_project(self) -> None:
        """在 sample_python_project 上运行安全扫描。"""
        py_files = _get_py_files()
        if not py_files:
            pytest.skip("sample_python_project fixture not found")

        analyzer = StaticAnalyzer(project_root=str(_FIXTURE_DIR))
        result = await analyzer.run_security_scan(py_files)

        # 验证返回结构正确
        # 注意：纯 Python 安全扫描可能 < 0.5ms，不强制要求 > 0
        # 正常的项目应该没有安全问题
        if not result.passed:
            for err in result.errors:
                assert err.message, f"Security issue missing message: {err}"
                assert err.code.startswith("SEC_"), f"Unexpected code format: {err.code}"

    async def test_run_all_real_project(self) -> None:
        """在 sample_python_project 上运行所有静态分析。"""
        py_files = _get_py_files()
        if not py_files:
            pytest.skip("sample_python_project fixture not found")

        analyzer = StaticAnalyzer(project_root=str(_FIXTURE_DIR))
        results = await analyzer.run_all(py_files)

        # 3 个结果: lint, typecheck, security
        assert len(results) == 3
        for i, result in enumerate(results):
            # Lint 和 typecheck 使用子进程，应 > 0；security 纯 Python 可能 < 0.5ms
            if i < 2:
                assert result.duration_ms > 0, f"Result {i} has zero duration"
            for err in result.errors:
                assert err.code, f"Result {i} error missing code: {err}"
            for warn in result.warnings:
                assert warn.code, f"Result {i} warning missing code: {warn}"

    async def test_lint_single_file(self) -> None:
        """对 sample_python_project 的单个文件运行 Lint。"""
        py_files = _get_py_files()
        if not py_files:
            pytest.skip("sample_python_project fixture not found")

        analyzer = StaticAnalyzer(project_root=str(_FIXTURE_DIR))
        # 只检查 main.py
        main_py = str(_FIXTURE_DIR / "main.py")
        result = await analyzer.run_lint([main_py])

        assert result.duration_ms > 0

    async def test_security_scan_with_detections(self, tmp_path: Path) -> None:
        """在有安全问题的文件上验证安全扫描能正确检测。"""
        # 创建一个有安全问题的临时 Python 项目
        f = tmp_path / "vulnerable.py"
        f.write_text(
            'password = "admin123"\n'
            'result = eval("1+1")\n'
        )

        analyzer = StaticAnalyzer(project_root=str(tmp_path))
        result = await analyzer.run_security_scan([str(f)])

        assert not result.passed
        codes = {e.code for e in result.errors}
        assert "SEC_HARDCODED_SECRET" in codes
        assert "SEC_EVAL" in codes
