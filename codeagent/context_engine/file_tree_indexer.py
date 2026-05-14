"""FileTreeIndexer — 项目目录扫描和文件树构建。

使用 os.scandir() 实现高性能目录扫描，支持 .gitignore 过滤和深度限制。
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 默认排除模式（与 .gitignore 风格一致）
_DEFAULT_EXCLUDE_PATTERNS = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "*.pyc",
    ".DS_Store",
    ".idea",
    ".vscode",
    ".codeagent",
    ".claude",
    "*.egg-info",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
]

# 扫描深度限制
_MAX_DEPTH = 3

# 时间格式
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _parse_gitignore(gitignore_path: str | os.PathLike) -> list[tuple[str, bool]]:
    """解析 .gitignore 文件，返回 (pattern, is_negation) 元组列表。

    处理注释、空行、尾部空格、前导 `/`、尾部 `/` 和 `!` 取反。
    """
    patterns: list[tuple[str, bool]] = []
    try:
        with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                is_negation = line.startswith("!")
                if is_negation:
                    pattern = line[1:].strip()
                else:
                    pattern = line

                if not pattern:
                    continue

                # 移除前导 `/`（表示从项目根目录匹配）
                if pattern.startswith("/"):
                    pattern = pattern[1:]

                patterns.append((pattern, is_negation))
    except (OSError, PermissionError):
        pass

    return patterns


def _path_matches_gitignore(
    rel_path: str,
    is_dir: bool,
    patterns: list[tuple[str, bool]],
) -> bool:
    """检查路径是否匹配 .gitignore 规则。

    规则：
    - 按顺序匹配，最后一个匹配的规则生效
    - 取反规则（!）覆盖之前的忽略规则
    - 尾部 `/` 仅匹配目录
    """
    ignored = False
    for pattern, is_negation in patterns:
        # 目录规则（尾部 /）
        match_dir = pattern.endswith("/")
        if match_dir:
            pattern_without_slash = pattern.rstrip("/")
        else:
            pattern_without_slash = pattern

        # 尝试匹配路径的任意部分（如 `__pycache__` 匹配 `a/b/__pycache__/c.py`）
        # 也尝试匹配完整路径
        if _pattern_matches(pattern_without_slash, rel_path, is_dir, match_dir):
            ignored = not is_negation
        elif not match_dir and _pattern_matches(
            pattern_without_slash, rel_path, is_dir, False
        ):
            ignored = not is_negation

    return ignored


def _pattern_matches(
    pattern: str, path: str, is_dir: bool, dir_only: bool
) -> bool:
    """检查单个模式是否匹配路径。"""
    if dir_only and not is_dir:
        return False

    # 直接 fnmatch（匹配文件名或完整路径）
    if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
        os.path.basename(path), pattern
    ):
        return True

    # 处理双星号模式
    if "**" in pattern:
        return _globstar_match(pattern, path)

    return False


def _globstar_match(pattern: str, path: str) -> bool:
    """简单的双星号（**）模式匹配。"""
    parts = pattern.split("**")
    if len(parts) < 2:
        return False

    # 简化实现：模式必须是 path 的前缀+后缀
    prefix = parts[0].rstrip("/")
    suffix = parts[-1].lstrip("/")

    if not suffix:
        # 模式以 ** 结尾，如 a/** → 匹配 a/ 下的所有内容
        return path.startswith(prefix) or prefix == path

    # a/**/b → 匹配 a/x/y/z/b
    return path.startswith(prefix) and path.endswith(suffix)


class FileTreeIndexer:
    """项目文件树索引器。

    扫描项目目录结构，生成可序列化的嵌套字典，支持 .gitignore 过滤和深度限制。

    Attributes:
        exclude_patterns: 排除模式列表（fnmatch 风格），合并了默认模式
    """

    def __init__(
        self, exclude_patterns: list[str] | None = None
    ) -> None:
        """初始化 FileTreeIndexer。

        Args:
            exclude_patterns: 额外的排除模式，将与默认模式合并
        """
        self.exclude_patterns = list(_DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            self.exclude_patterns.extend(exclude_patterns)

    async def scan(self, project_root: str | os.PathLike) -> dict[str, Any]:
        """扫描项目目录，构建文件树。

        Args:
            project_root: 项目根目录路径

        Returns:
            FileTree 嵌套字典结构
        """
        root = Path(project_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")

        # 读取 .gitignore
        gitignore_path = root / ".gitignore"
        gitignore_patterns = _parse_gitignore(gitignore_path)

        tree = self._build_node(root, root, 0, gitignore_patterns)
        return tree

    def _build_node(
        self,
        root: Path,
        current: Path,
        depth: int,
        gitignore_patterns: list[tuple[str, bool]],
    ) -> dict[str, Any]:
        """递归构建目录树节点。"""
        rel_path = str(current.relative_to(root))

        if depth >= _MAX_DEPTH:
            # 统计子项数量但不递归
            child_count = 0
            try:
                child_count = sum(1 for _ in os.scandir(current))
            except (PermissionError, OSError):
                pass
            return {
                "name": current.name,
                "type": "directory",
                "path": rel_path,
                "size": 0,
                "mtime": _format_mtime(current),
                "children": [],
                "_summary": "depth exceeded",
                "_child_count": child_count,
            }

        try:
            entries = list(os.scandir(current))
        except PermissionError:
            return {
                "name": current.name,
                "type": "directory",
                "path": rel_path,
                "size": 0,
                "mtime": _format_mtime(current),
                "children": [],
                "_summary": "permission denied",
            }
        except OSError:
            return {
                "name": current.name,
                "type": "directory",
                "path": rel_path,
                "size": 0,
                "mtime": _format_mtime(current),
                "children": [],
            }

        children: list[dict[str, Any]] = []
        for entry in entries:
            # 跳过隐藏文件/目录？仅跳过特定的排除模式，不跳过所有隐藏文件
            entry_name = entry.name
            entry_rel = str(
                (current / entry_name).relative_to(root)
            ).replace("\\", "/")

            # 检查默认排除模式
            if self._is_excluded(entry_rel, entry_name, entry.is_dir()):
                continue

            # 检查 .gitignore 模式
            if _path_matches_gitignore(
                entry_rel, entry.is_dir(), gitignore_patterns
            ):
                continue

            try:
                if entry.is_dir(follow_symlinks=False):
                    node = self._build_node(
                        root,
                        current / entry_name,
                        depth + 1,
                        gitignore_patterns,
                    )
                    children.append(node)
                elif entry.is_file():
                    children.append(self._build_file_node(root, entry))
            except PermissionError:
                children.append({
                    "name": entry_name,
                    "type": "file",
                    "path": entry_rel,
                    "size": 0,
                    "mtime": "",
                    "extension": "",
                    "_summary": "permission denied",
                })

        # 文件按名称排序
        children.sort(key=lambda n: (n["type"] != "directory", n["name"]))

        mtime_val = _format_mtime(current)

        return {
            "name": current.name,
            "type": "directory",
            "path": rel_path,
            "size": 0,
            "mtime": mtime_val,
            "children": children,
        }

    def _build_file_node(
        self, root: Path, entry: os.DirEntry
    ) -> dict[str, Any]:
        """构建文件节点。"""
        rel_path = str(
            (Path(entry.path)).relative_to(root)
        ).replace("\\", "/")
        ext = Path(entry.name).suffix.lower()
        stat_info = entry.stat()

        return {
            "name": entry.name,
            "type": "file",
            "path": rel_path,
            "size": stat_info.st_size,
            "mtime": _format_timestamp(stat_info.st_mtime),
            "extension": ext,
        }

    def _is_excluded(
        self, rel_path: str, name: str, is_dir: bool
    ) -> bool:
        """检查路径是否匹配排除模式。"""
        for pattern in self.exclude_patterns:
            dir_only = pattern.endswith("/")
            pat_clean = pattern.rstrip("/")

            if dir_only and not is_dir:
                continue

            # 匹配文件名或相对路径
            if fnmatch.fnmatch(name, pat_clean) or fnmatch.fnmatch(
                rel_path.replace("\\", "/"), pat_clean
            ):
                return True

        return False

    def to_json(self, tree: dict[str, Any], indent: int = 2) -> str:
        """序列化文件树为 JSON 字符串。

        Args:
            tree: scan() 返回的文件树
            indent: JSON 缩进空格数

        Returns:
            str: JSON 字符串
        """
        return json.dumps(tree, indent=indent, ensure_ascii=False)

    def to_compact_string(
        self, tree: dict[str, Any], max_lines: int = 100
    ) -> str:
        """生成紧凑的文件树文本表示（供 LLM 上下文使用）。

        Args:
            tree: scan() 返回的文件树
            max_lines: 最大行数限制

        Returns:
            str: 紧凑文本表示
        """
        lines: list[str] = []
        self._compact_lines(tree, "", lines, max_lines)
        return "\n".join(lines[:max_lines])

    def _compact_lines(
        self,
        node: dict[str, Any],
        prefix: str,
        lines: list[str],
        max_lines: int,
    ) -> None:
        """递归生成紧凑文本行。"""
        if len(lines) >= max_lines:
            return

        if node.get("type") == "directory":
            summary = node.get("_summary")
            if summary == "depth exceeded":
                children = node.get("children", [])
                lines.append(f"{prefix}{node['name']}/ ({len(children)} items, nested)")
                return
            if summary == "permission denied":
                lines.append(f"{prefix}{node['name']}/ [permission denied]")
                return

            children = node.get("children", [])
            flat_count = _count_files(children)
            lines.append(f"{prefix}{node['name']}/ ({len(children)} entries, ~{flat_count} files)")

            for child in children:
                self._compact_lines(child, prefix + "  ", lines, max_lines)
        else:
            size = node.get("size", 0)
            size_str = _format_size(size)
            lines.append(f"{prefix}{node['name']}  {size_str}")


def _count_files(nodes: list[dict[str, Any]]) -> int:
    """递归统计文件数量。"""
    count = 0
    for n in nodes:
        if n.get("type") == "file":
            count += 1
        elif n.get("type") == "directory":
            count += _count_files(n.get("children", []))
    return count


def _format_mtime(path: Path) -> str:
    """格式化文件的最后修改时间。"""
    try:
        stat = path.stat()
        return _format_timestamp(stat.st_mtime)
    except OSError:
        return ""


def _format_timestamp(timestamp: float) -> str:
    """将时间戳格式化为 ISO 字符串。"""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime(_ISO_FORMAT)
    except (OSError, ValueError, OverflowError):
        return ""


def _format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
