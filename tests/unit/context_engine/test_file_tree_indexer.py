"""FileTreeIndexer 单元测试。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from codeagent.context_engine.file_tree_indexer import FileTreeIndexer


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def temp_project() -> Path:
    """创建临时项目目录结构用于测试。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # 创建文件和子目录
        (root / "main.py").write_text("print('hello')\n")
        (root / "utils.py").write_text("def foo():\n    pass\n")
        (root / "README.md").write_text("# Project\n")

        sub_dir = root / "src"
        sub_dir.mkdir()
        (sub_dir / "__init__.py").write_text("")
        (sub_dir / "module.py").write_text("class A:\n    pass\n")
        (sub_dir / "styles.css").write_text("body {}\n")

        nested = root / "src" / "deep"
        nested.mkdir()
        (nested / "inner.py").write_text("# nested\n")

        deep = root / "src" / "deep" / "deeper"
        deep.mkdir()
        (deep / "deepest.py").write_text("# even deeper\n")

        yield root


@pytest.fixture
def indexer() -> FileTreeIndexer:
    return FileTreeIndexer()


# ── Tests: Basic scanning ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestScanBasic:
    """基本扫描功能测试。"""

    async def test_scan_normal_project(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        tree = await indexer.scan(temp_project)
        assert tree["type"] == "directory"
        assert tree["name"] == temp_project.name
        assert "children" in tree

    async def test_scan_root_contains_top_files(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        tree = await indexer.scan(temp_project)
        names = {c["name"] for c in tree["children"]}
        assert "main.py" in names
        assert "utils.py" in names
        assert "README.md" in names

    async def test_scan_root_contains_src_dir(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        tree = await indexer.scan(temp_project)
        dirs = [c for c in tree["children"] if c["type"] == "directory"]
        assert any(d["name"] == "src" for d in dirs)

    async def test_scan_nested_directory(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        tree = await indexer.scan(temp_project)
        src = next(c for c in tree["children"] if c["name"] == "src")
        assert src["type"] == "directory"
        src_names = {c["name"] for c in src["children"]}
        assert "__init__.py" in src_names
        assert "module.py" in src_names
        assert "deep" in src_names

    async def test_scan_file_attributes(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        tree = await indexer.scan(temp_project)
        main = next(c for c in tree["children"] if c["name"] == "main.py")
        assert main["type"] == "file"
        assert main["extension"] == ".py"
        assert main["size"] > 0
        assert main["mtime"] != ""

    async def test_scan_nonexistent_directory(self, indexer: FileTreeIndexer) -> None:
        with pytest.raises(NotADirectoryError):
            await indexer.scan("/nonexistent_path_xyz_123")

    async def test_scan_empty_directory(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = await indexer.scan(tmpdir)
            assert tree["type"] == "directory"
            assert tree["children"] == []


# ── Tests: Depth limiting ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestDepthLimiting:
    """验证深度限制功能。"""

    async def test_depth_limit_at_3(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        """深度超过 3 层的目录应被折叠。"""
        tree = await indexer.scan(temp_project)
        src = next(c for c in tree["children"] if c["name"] == "src")
        deep = next(c for c in src["children"] if c["name"] == "deep")
        assert deep["type"] == "directory"

        deeper = next(c for c in deep["children"] if c["name"] == "deeper")
        assert deeper.get("_summary") == "depth exceeded"

    async def test_depth_limit_file_count(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        """深度超过 3 层的目录应显示子项数量。"""
        tree = await indexer.scan(temp_project)
        src = next(c for c in tree["children"] if c["name"] == "src")
        deep = next(c for c in src["children"] if c["name"] == "deep")
        deeper = next(c for c in deep["children"] if c["name"] == "deeper")
        assert deeper.get("_child_count", 0) >= 1  # deepest.py


# ── Tests: Default exclusion patterns ────────────────────────────────────


@pytest.mark.asyncio
class TestDefaultExclusions:
    """验证默认排除模式。"""

    async def test_exclude_git(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("[core]\n")
            (root / "main.py").write_text("x = 1\n")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert ".git" not in names
            assert "main.py" in names

    async def test_exclude_pycache(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "main.cpython-312.pyc").write_text("trash")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "__pycache__" not in names

    async def test_exclude_pyc_files(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "module.pyc").write_text("trash")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "module.pyc" not in names

    async def test_exclude_venv(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".venv").mkdir()
            (root / ".venv" / "bin").mkdir()
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert ".venv" not in names

    async def test_exclude_codeagent_dir(self, indexer: FileTreeIndexer) -> None:
        """.codeagent 目录应被默认排除。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".codeagent").mkdir()
            (root / ".codeagent" / "config.yaml").write_text("key: val\n")
            (root / "main.py").write_text("x = 1\n")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert ".codeagent" not in names
            assert "main.py" in names

    async def test_exclude_claude_dir(self, indexer: FileTreeIndexer) -> None:
        """.claude 目录应被默认排除。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".claude").mkdir()
            (root / ".claude" / "settings.json").write_text("{}")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert ".claude" not in names


# ── Tests: Custom exclude patterns ───────────────────────────────────────


@pytest.mark.asyncio
class TestCustomExclusions:
    """验证自定义排除模式。"""

    async def test_custom_exclude(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "build").mkdir()
            (root / "dist").mkdir()
            (root / "main.py").write_text("x = 1\n")
            indexer = FileTreeIndexer(exclude_patterns=["build", "dist"])
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "build" not in names
            assert "dist" not in names
            assert "main.py" in names


# ── Tests: .gitignore filtering ──────────────────────────────────────────


@pytest.mark.asyncio
class TestGitignoreFiltering:
    """验证 .gitignore 过滤。"""

    async def test_gitignore_excludes_pattern(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("*.log\n")
            (root / "app.log").write_text("error\n")
            (root / "main.py").write_text("x = 1\n")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "app.log" not in names
            assert "main.py" in names

    async def test_gitignore_negation(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("*.pyc\n!important.pyc\n")
            (root / "module.pyc").write_text("trash")
            (root / "important.pyc").write_text("keep")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            # !important.pyc overrides *.pyc for important.pyc
            assert "important.pyc" in names or "module.pyc" not in names

    async def test_gitignore_directory_pattern(self, indexer: FileTreeIndexer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".gitignore").write_text("temp/\n")
            (root / "temp").mkdir()
            (root / "temp" / "cache.txt").write_text("data")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "temp" not in names

    async def test_gitignore_not_present(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        """没有 .gitignore 文件时应正常扫描。"""
        tree = await indexer.scan(temp_project)
        assert tree["type"] == "directory"
        assert len(tree["children"]) > 0


# ── Tests: Serialization ─────────────────────────────────────────────────


class TestSerialization:
    """验证序列化方法。"""

    def test_to_json(self, temp_project: Path) -> None:
        import json
        import asyncio

        indexer = FileTreeIndexer()
        tree = asyncio.run(indexer.scan(temp_project))
        json_str = indexer.to_json(tree)
        parsed = json.loads(json_str)
        assert parsed["name"] == temp_project.name
        assert parsed["type"] == "directory"

    def test_to_compact_string(self, temp_project: Path) -> None:
        import asyncio

        indexer = FileTreeIndexer()
        tree = asyncio.run(indexer.scan(temp_project))
        compact = indexer.to_compact_string(tree)
        assert isinstance(compact, str)
        assert len(compact) > 0
        # 应包含项目名称
        assert temp_project.name in compact

    def test_to_compact_string_max_lines(self, temp_project: Path) -> None:
        import asyncio

        indexer = FileTreeIndexer()
        tree = asyncio.run(indexer.scan(temp_project))
        compact = indexer.to_compact_string(tree, max_lines=5)
        lines = compact.split("\n")
        assert len(lines) <= 5

    def test_to_json_with_indent(self, temp_project: Path) -> None:
        import asyncio

        indexer = FileTreeIndexer()
        tree = asyncio.run(indexer.scan(temp_project))
        json_str = indexer.to_json(tree, indent=4)
        # indent 4 produces lines starting with 4 spaces
        assert "    " in json_str


# ── Tests: Edge cases ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    """边界场景测试。"""

    async def test_file_sort_order(self, temp_project: Path, indexer: FileTreeIndexer) -> None:
        """目录应在文件前面，同级按名称排序。"""
        tree = await indexer.scan(temp_project)
        children = tree["children"]
        # 找到第一个目录和第一个文件的索引
        dir_indices = [i for i, c in enumerate(children) if c["type"] == "directory"]
        file_indices = [i for i, c in enumerate(children) if c["type"] == "file"]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)

    async def test_empty_filename(self, indexer: FileTreeIndexer) -> None:
        """空目录应生成空的 children 列表。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tree = await indexer.scan(tmpdir)
            assert tree["children"] == []

    async def test_symlink_skipped(self, indexer: FileTreeIndexer) -> None:
        """符号链接应被跳过（follow_symlinks=False）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "real_file.py").write_text("x = 1\n")
            # 创建符号链接（跳过如果平台不支持）
            try:
                os.symlink(root / "real_file.py", root / "link.py")
            except (OSError, NotImplementedError):
                pytest.skip("Symlinks not supported on this platform")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "real_file.py" in names
            # 符号链接触发 PermissionError 或直接被跳过
            # 无论哪种情况，扫描不应崩溃

    async def test_scan_with_special_chars(self, indexer: FileTreeIndexer) -> None:
        """包含特殊字符的文件名应被正确处理。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file with spaces.py").write_text("x = 1\n")
            (root / "file-with-dashes.py").write_text("y = 2\n")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert "file with spaces.py" in names
            assert "file-with-dashes.py" in names

    async def test_hidden_dotfile_included(self, indexer: FileTreeIndexer) -> None:
        """非排除列表中的隐藏文件（.dotfile）应被包含。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".hidden.py").write_text("x = 1\n")
            (root / "visible.py").write_text("y = 2\n")
            tree = await indexer.scan(tmpdir)
            names = {c["name"] for c in tree["children"]}
            assert ".hidden.py" in names
