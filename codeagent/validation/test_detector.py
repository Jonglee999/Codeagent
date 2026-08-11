"""TestDetector — 测试框架自动检测器。

参考 SRS §6.3.3。
自动检测项目使用的测试框架、测试目录和配置，生成可执行的测试命令。
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestFrameworkInfo:
    """检测到的测试框架信息。"""

    framework: str | None = None  # "pytest" | "unittest" | "none"
    config_files: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    test_file_pattern: str = "test_*.py"
    has_asyncio_support: bool = False
    has_cov_support: bool = False
    installed_packages: list[str] = field(default_factory=list)
    test_command: str = ""

    @property
    def has_tests(self) -> bool:
        """项目是否配置了测试框架。"""
        return self.framework is not None and self.framework != "none"


class TestDetector:
    """自动检测项目使用的测试框架和配置。

    检测顺序：
    1. 扫描配置文件：pytest.ini, pyproject.toml, tox.ini, setup.cfg
    2. 扫描目录约定：tests/, test/, __tests__/, spec/
    3. 检查已安装的测试依赖
    """

    __test__ = False  # 不是 pytest 测试类

    # 测试配置文件名（按优先级排序）
    CONFIG_FILES = [
        "pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
    ]

    # 测试目录约定
    TEST_DIR_NAMES = ["tests", "test", "__tests__", "spec"]

    # 测试依赖
    TEST_DEPENDENCIES = {
        "pytest": "pytest",
        "pytest-asyncio": "pytest_asyncio",
        "pytest-cov": "pytest_cov",
        "pytest-mock": "pytest_mock",
        "unittest": "unittest",
    }

    def __init__(self, project_root: str) -> None:
        """初始化 TestDetector。

        Args:
            project_root: 项目根目录路径
        """
        self._project_root = Path(project_root).resolve()

    # ── 主检测流程 ────────────────────────────────────────────

    async def detect(self) -> TestFrameworkInfo:
        """全量检测项目测试框架配置。

        Returns:
            TestFrameworkInfo: 检测到的测试框架信息
        """
        info = TestFrameworkInfo()

        # 1. 扫描配置文件
        config_files = self._find_config_files()
        info.config_files = [str(p) for p in config_files]

        # 2. 扫描测试目录
        test_dirs = self._find_test_dirs()
        info.test_dirs = [str(d) for d in test_dirs]

        # 4. 检查测试依赖
        installed = self._check_dependencies()
        info.installed_packages = installed
        info.has_asyncio_support = "pytest-asyncio" in installed
        info.has_cov_support = "pytest-cov" in installed

        # 5. 综合判定框架类型
        info.framework = self._detect_framework(
            config_files, test_dirs, installed
        )
        if info.framework is None and self._find_root_test_files():
            info.framework = "pytest"

        # 6. 生成测试命令
        if info.framework == "pytest":
            info.test_command = self._build_pytest_command(info)
        elif info.framework == "unittest":
            info.test_command = self._build_unittest_command(info)
        elif test_dirs:
            # 有测试目录但无明确配置，默认用 pytest
            info.framework = "pytest"
            info.test_command = self._build_pytest_command(info)

        return info

    def _find_root_test_files(self) -> list[Path]:
        """Discover pytest-style files placed directly in the project root."""
        found = set(self._project_root.glob("test_*.py"))
        found.update(self._project_root.glob("*_test.py"))
        return sorted(path for path in found if path.is_file())

    # ── 配置文件发现 ──────────────────────────────────────────

    def _find_config_files(self) -> list[Path]:
        """查找项目根目录下的测试配置文件。

        Returns:
            list[Path]: 找到的配置文件路径列表
        """
        found: list[Path] = []
        for name in self.CONFIG_FILES:
            path = self._project_root / name
            if path.is_file():
                found.append(path)
        return found

    def _parse_pyproject_toml(self, path: Path) -> dict[str, Any]:
        """解析 pyproject.toml 中的 pytest 配置。

        Args:
            path: pyproject.toml 路径

        Returns:
            dict: 解析出的配置信息
        """
        result: dict[str, Any] = {}

        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                return result

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            # [tool.pytest.ini_options]
            pytest_opts = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
            if pytest_opts:
                result["pytest_config"] = pytest_opts

            # 检查 optional-dependencies 和 dependency-groups 中的 pytest
            deps: dict[str, list[str]] = {}

            # pyproject.toml 标准格式
            for section in ("dependencies", "optional-dependencies"):
                section_data = data.get(section, {})
                if isinstance(section_data, dict):
                    for group_name, group_deps in section_data.items():
                        if isinstance(group_deps, list):
                            deps.setdefault(str(group_name), []).extend(
                                str(d) for d in group_deps
                            )

            # dependency-groups (PEP 735)
            dep_groups = data.get("dependency-groups", {})
            if isinstance(dep_groups, dict):
                for group_name, group_deps in dep_groups.items():
                    if isinstance(group_deps, list):
                        deps.setdefault(str(group_name), []).extend(
                            self._flatten_dep_group(d) for d in group_deps
                        )

            result["dependencies"] = deps
            result["raw"] = data

        except Exception:
            pass

        return result

    def _flatten_dep_group(self, dep: Any) -> str:
        """展平 PEP 735 dependency-group 条目。"""
        if isinstance(dep, str):
            return dep
        if isinstance(dep, dict):
            # {include = "dev"} -> 按 include 处理
            if "include" in dep:
                return f"include:{dep['include']}"
        return str(dep)

    def _parse_setup_cfg(self, path: Path) -> dict[str, Any]:
        """解析 setup.cfg 中的 pytest 配置。"""
        result: dict[str, Any] = {}
        try:
            parser = configparser.ConfigParser()
            parser.read(str(path), encoding="utf-8")
            if parser.has_section("tool:pytest"):
                result["pytest_config"] = dict(parser["tool:pytest"])
            if parser.has_section("options"):
                result["options"] = dict(parser["options"])
        except Exception:
            pass
        return result

    def _parse_tox_ini(self, path: Path) -> dict[str, Any]:
        """解析 tox.ini 中的 pytest 配置。"""
        result: dict[str, Any] = {}
        try:
            parser = configparser.ConfigParser()
            parser.read(str(path), encoding="utf-8")
            if parser.has_section("pytest"):
                result["pytest_config"] = dict(parser["pytest"])
        except Exception:
            pass
        return result

    # ── 框架检测 ──────────────────────────────────────────────

    def _detect_framework(
        self,
        config_files: list[Path],
        test_dirs: list[Path],
        installed_packages: list[str],
    ) -> str | None:
        """综合检测测试框架类型。

        策略（按优先级）：
        1. 有配置文件 → 根据配置文件类型判定
        2. 无配置文件，有测试目录 → 默认为 pytest
        3. 两者都没有 → 不检测框架（返回 None）

        Args:
            config_files: 发现的配置文件列表
            test_dirs: 发现的测试目录列表
            installed_packages: 已安装的测试包列表

        Returns:
            str | None: "pytest", "unittest", 或 None
        """
        # 1. 通过配置文件检测（最强信号）
        for path in config_files:
            if path.name == "pytest.ini":
                return "pytest"

            if path.name == "pyproject.toml":
                parsed = self._parse_pyproject_toml(path)
                pytest_config = parsed.get("pytest_config")
                if pytest_config:
                    return "pytest"

            if path.name == "setup.cfg":
                parsed = self._parse_setup_cfg(path)
                if parsed.get("pytest_config"):
                    return "pytest"

            if path.name == "tox.ini":
                parsed = self._parse_tox_ini(path)
                if parsed.get("pytest_config"):
                    return "pytest"

        # 2. 通过测试目录检测（次强信号）
        if test_dirs:
            # 检查是否有测试文件
            has_test_files = False
            for td in test_dirs:
                if list(td.rglob("test_*.py")):
                    has_test_files = True
                    break
            if has_test_files or installed_packages:
                return "pytest"
            # 有目录但无文件也无依赖 → 仍返回 pytest（目录结构已明确意图）
            return "pytest"

        # 3. 没有配置文件和目录，不检测
        return None

    # ── 测试目录发现 ──────────────────────────────────────────

    def _find_test_dirs(self) -> list[Path]:
        """扫描项目根目录下的测试目录。

        Returns:
            list[Path]: 找到的测试目录路径列表
        """
        dirs: list[Path] = []
        for name in self.TEST_DIR_NAMES:
            path = self._project_root / name
            if path.is_dir():
                dirs.append(path)

        # 也检查 src/ 下的 tests/ 目录
        src_tests = self._project_root / "src" / "tests"
        if src_tests.is_dir():
            dirs.append(src_tests)

        return sorted(dirs)

    # ── 依赖检查 ──────────────────────────────────────────────

    def _check_dependencies(self) -> list[str]:
        """检查已安装的测试依赖。

        尝试导入测试库来判断是否已安装。

        Returns:
            list[str]: 已安装的测试包名列表
        """
        installed: list[str] = []
        for pkg_name, module_name in self.TEST_DEPENDENCIES.items():
            try:
                __import__(module_name)
                installed.append(pkg_name)
            except ImportError:
                pass
        return installed

    # ── 测试命令生成 ──────────────────────────────────────────

    def _build_pytest_command(self, info: TestFrameworkInfo) -> str:
        """构建 pytest 测试命令。"""
        cmd_parts = ["pytest"]

        # 取第一个有效配置
        if info.config_files:
            cmd_parts.append("-c")
            config_path = Path(info.config_files[0])
            try:
                config_path = config_path.relative_to(self._project_root)
            except ValueError:
                pass
            cmd_parts.append(config_path.as_posix())

        # 测试目录
        if info.test_dirs:
            relative_dirs: list[str] = []
            for item in info.test_dirs:
                directory = Path(item)
                try:
                    directory = directory.relative_to(self._project_root)
                except ValueError:
                    pass
                relative_dirs.append(directory.as_posix())
            dirs_str = " ".join(relative_dirs)
            cmd_parts.append(dirs_str)

        # 选项
        cmd_parts.append("-x")  # 首个失败即停止
        cmd_parts.append("--tb=short")

        # asyncio 支持
        if info.has_asyncio_support:
            cmd_parts.append("--asyncio-mode=auto")

        return " ".join(cmd_parts)

    def _build_unittest_command(self, info: TestFrameworkInfo) -> str:
        """构建 unittest 测试命令。"""
        cmd_parts = ["python", "-m", "unittest", "discover"]

        if info.test_dirs:
            cmd_parts.append("-s")
            cmd_parts.append(str(info.test_dirs[0]))

        cmd_parts.append("-v")

        return " ".join(cmd_parts)

    # ── 快捷方法 ──────────────────────────────────────────────

    async def get_test_command(self) -> str:
        """直接获取测试命令（快捷方法）。"""
        info = await self.detect()
        return info.test_command
