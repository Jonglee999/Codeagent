"""TestDetector 单元测试。

覆盖 pytest/unittest 框架检测、配置文件发现、目录扫描、依赖检测、
测试命令生成、边界情况等。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codeagent.validation.test_detector import TestDetector


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def empty_project(tmp_path: Path) -> Path:
    """空项目（无测试配置、无测试目录）。"""
    return tmp_path


@pytest.fixture
def project_with_pytest_ini(tmp_path: Path) -> Path:
    """通过 pytest.ini 配置的项目。"""
    d = tmp_path / "proj_pytest_ini"
    d.mkdir()
    (d / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    return d


@pytest.fixture
def project_with_pyproject_toml(tmp_path: Path) -> Path:
    """通过 pyproject.toml 配置的项目。"""
    d = tmp_path / "proj_pyproject"
    d.mkdir()
    (d / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        'testpaths = ["tests"]\n'
        'python_files = ["test_*.py"]\n'
        'asyncio_mode = "auto"\n'
    )
    return d


@pytest.fixture
def project_with_setup_cfg(tmp_path: Path) -> Path:
    """通过 setup.cfg 配置的项目。"""
    d = tmp_path / "proj_setup_cfg"
    d.mkdir()
    (d / "setup.cfg").write_text(
        "[tool:pytest]\n"
        "testpaths = tests\n"
    )
    return d


@pytest.fixture
def project_with_tox_ini(tmp_path: Path) -> Path:
    """通过 tox.ini 配置的项目。"""
    d = tmp_path / "proj_tox"
    d.mkdir()
    (d / "tox.ini").write_text(
        "[pytest]\n"
        "testpaths = tests\n"
    )
    return d


@pytest.fixture
def project_with_test_dirs(tmp_path: Path) -> Path:
    """有 tests/ 和 test/ 目录的项目。"""
    d = tmp_path / "proj_dirs"
    d.mkdir()
    (d / "tests").mkdir()
    (d / "tests" / "test_main.py").write_text("def test_x(): pass\n")
    (d / "tests" / "__init__.py").write_text("")
    (d / "test").mkdir()
    (d / "test" / "test_utils.py").write_text("def test_y(): pass\n")
    return d


@pytest.fixture
def project_with_multiple_configs(tmp_path: Path) -> Path:
    """同时有 pytest.ini 和 pyproject.toml 的项目。"""
    d = tmp_path / "proj_multi"
    d.mkdir()
    (d / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (d / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (d / "src").mkdir()
    return d


# ── 配置文件发现 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestConfigFileDetection:
    """测试各类配置文件的发现能力。"""

    async def test_detect_pytest_ini(
        self, project_with_pytest_ini: Path
    ) -> None:
        """pytest.ini 配置的项目。"""
        detector = TestDetector(str(project_with_pytest_ini))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("pytest.ini" in str(p) for p in info.config_files)

    async def test_detect_pyproject_toml(
        self, project_with_pyproject_toml: Path
    ) -> None:
        """pyproject.toml 配置的项目。"""
        detector = TestDetector(str(project_with_pyproject_toml))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("pyproject.toml" in str(p) for p in info.config_files)

    async def test_detect_setup_cfg(
        self, project_with_setup_cfg: Path
    ) -> None:
        """setup.cfg 配置的项目。"""
        detector = TestDetector(str(project_with_setup_cfg))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("setup.cfg" in str(p) for p in info.config_files)

    async def test_detect_tox_ini(
        self, project_with_tox_ini: Path
    ) -> None:
        """tox.ini 配置的项目。"""
        detector = TestDetector(str(project_with_tox_ini))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("tox.ini" in str(p) for p in info.config_files)

    async def test_multiple_configs(
        self, project_with_multiple_configs: Path
    ) -> None:
        """多个配置文件共存。"""
        detector = TestDetector(str(project_with_multiple_configs))
        info = await detector.detect()
        assert info.framework == "pytest"
        # pytest.ini 优先
        assert len(info.config_files) >= 2

    async def test_empty_project(self, empty_project: Path) -> None:
        """空项目无任何配置。"""
        detector = TestDetector(str(empty_project))
        info = await detector.detect()
        assert info.framework is None
        assert not info.config_files


# ── 测试目录发现 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTestDirDetection:
    """测试目录扫描能力。"""

    async def test_detect_tests_dir(self, tmp_path: Path) -> None:
        """检测 tests/ 目录。"""
        d = tmp_path / "proj"
        d.mkdir()
        (d / "tests").mkdir()
        (d / "tests" / "test_x.py").write_text("def test_x(): pass\n")
        detector = TestDetector(str(d))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("tests" in str(p) for p in info.test_dirs)

    async def test_detect_test_dir(self, tmp_path: Path) -> None:
        """检测 test/ 目录。"""
        d = tmp_path / "proj"
        d.mkdir()
        (d / "test").mkdir()
        (d / "test" / "test_y.py").write_text("def test_y(): pass\n")
        detector = TestDetector(str(d))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert any("test" in str(p) for p in info.test_dirs)

    async def test_detect_all_dirs(
        self, project_with_test_dirs: Path
    ) -> None:
        """同时发现多个测试目录。"""
        detector = TestDetector(str(project_with_test_dirs))
        info = await detector.detect()
        assert info.framework == "pytest"
        assert len(info.test_dirs) >= 2

    async def test_no_test_dir(self, empty_project: Path) -> None:
        """无测试目录。"""
        detector = TestDetector(str(empty_project))
        info = await detector.detect()
        assert info.framework is None
        assert not info.test_dirs

    async def test_empty_test_dir(self, tmp_path: Path) -> None:
        """空的 tests/ 目录（无测试文件）。"""
        d = tmp_path / "proj"
        d.mkdir()
        (d / "tests").mkdir()
        detector = TestDetector(str(d))
        info = await detector.detect()
        # 有 tests/ 目录但没有 test_*.py，框架仍应检测为 pytest
        assert info.framework == "pytest"
        assert len(info.test_dirs) >= 1


# ── 测试命令生成 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTestCommand:
    """测试命令生成。"""

    async def test_pytest_command(self, project_with_pytest_ini: Path) -> None:
        """pytest 项目生成正确命令。"""
        detector = TestDetector(str(project_with_pytest_ini))
        info = await detector.detect()
        assert "pytest" in info.test_command
        assert "-x" in info.test_command

    async def test_pytest_command_with_asyncio(
        self, project_with_pytest_ini: Path
    ) -> None:
        """包含 asyncio 支持的测试命令。"""
        import pytest_asyncio  # noqa: F401

        detector = TestDetector(str(project_with_pytest_ini))
        info = await detector.detect()
        if info.has_asyncio_support:
            assert "--asyncio-mode=auto" in info.test_command

    async def test_unittest_project(self, tmp_path: Path) -> None:
        """unittest 项目生成正确命令。"""
        d = tmp_path / "unittest_proj"
        d.mkdir()
        (d / "tests").mkdir()
        (d / "tests" / "test_x.py").write_text(
            "import unittest\nclass TestX(unittest.TestCase):\n    pass\n"
        )
        detector = TestDetector(str(d))
        info = await detector.detect()
        # 默认检测为 pytest，因为有 tests/ 目录
        assert info.framework == "pytest"

    async def test_get_test_command_shortcut(
        self, project_with_pytest_ini: Path
    ) -> None:
        """快捷方法 get_test_command。"""
        detector = TestDetector(str(project_with_pytest_ini))
        cmd = await detector.get_test_command()
        assert "pytest" in cmd


# ── Asyncio 检测 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAsyncioDetection:
    """pytest-asyncio 支持检测。"""

    async def test_asyncio_support_detected(self, tmp_path: Path) -> None:
        """有 pytest-asyncio 安装时。"""
        d = tmp_path / "asyncio_proj"
        d.mkdir()
        (d / "pytest.ini").write_text("[pytest]\nasyncio_mode = auto\n")
        detector = TestDetector(str(d))
        info = await detector.detect()
        # pytest-asyncio 在环境中已安装
        assert info.has_asyncio_support is True

    async def test_cov_support_detected(self, tmp_path: Path) -> None:
        """有 pytest-cov 安装时。"""
        d = tmp_path / "cov_proj"
        d.mkdir()
        (d / "pytest.ini").write_text("[pytest]\n")
        detector = TestDetector(str(d))
        info = await detector.detect()
        assert info.has_cov_support is True


# ── has_tests 属性 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestHasTests:
    """has_tests 属性行为。"""

    async def test_has_tests_true(self, project_with_pytest_ini: Path) -> None:
        """有测试框架时 has_tests=True。"""
        detector = TestDetector(str(project_with_pytest_ini))
        info = await detector.detect()
        assert info.has_tests is True

    async def test_has_tests_false(self, empty_project: Path) -> None:
        """无测试框架时 has_tests=False。"""
        detector = TestDetector(str(empty_project))
        info = await detector.detect()
        assert info.has_tests is False


# ── 边界情况 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    """边界情况测试。"""

    async def test_non_python_project(self, tmp_path: Path) -> None:
        """非 Python 项目（无 .py 文件）。"""
        d = tmp_path / "js_proj"
        d.mkdir()
        (d / "tests").mkdir()
        (d / "tests" / "test_app.js").write_text("// js test")
        detector = TestDetector(str(d))
        info = await detector.detect()
        # 有 tests/ 目录但无 .py 文件，仍应检测为 pytest
        assert info.framework == "pytest"
        assert any("tests" in str(p) for p in info.test_dirs)

    async def test_deeply_nested_project(self, tmp_path: Path) -> None:
        """深层嵌套项目结构。"""
        d = tmp_path / "deep_proj"
        d.mkdir()
        (d / "src" / "tests").mkdir(parents=True)
        (d / "src" / "tests" / "test_deep.py").write_text("def test_x(): pass\n")
        detector = TestDetector(str(d))
        info = await detector.detect()
        # src/tests/ 也会被扫描到
        assert any("src" + os.sep + "tests" in str(p) or "src/tests" in str(p) for p in info.test_dirs)

    async def test_project_root_nonexistent(self, tmp_path: Path) -> None:
        """不存在的项目根目录。"""
        detector = TestDetector(str(tmp_path / "nonexistent"))
        info = await detector.detect()
        assert info.framework is None
        assert not info.config_files
        assert not info.test_dirs

    async def test_pyproject_without_pytest(self, tmp_path: Path) -> None:
        """pyproject.toml 存在但没有 pytest 配置。"""
        d = tmp_path / "proj"
        d.mkdir()
        (d / "pyproject.toml").write_text(
            "[tool.ruff]\nline-length = 100\n"
        )
        detector = TestDetector(str(d))
        info = await detector.detect()
        # 没有测试配置，但也没有测试目录
        assert info.framework is None
