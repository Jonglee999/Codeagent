"""MemoryStore — 两级记忆存储 CRUD。

提供全局（~/.codeagent/memory/）和项目（.codeagent/memory/）两级存储，
支持 5 种记忆类型，使用 YAML Frontmatter + Markdown 格式，
自动维护 MEMORY.md 索引。
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

import frontmatter

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    CODE_PATTERN = "code_pattern"
    SESSION = "session"


# ── 截断优先级：索引超 200 行时按此顺序删除最旧条目 ──────────
_TRUNCATION_PRIORITY: list[MemoryType] = [
    MemoryType.SESSION,
    MemoryType.PROJECT,
    MemoryType.CODE_PATTERN,
]

_INDEX_HEADER = "# Memory Index\n"

# 类型目录映射
_TYPE_DIRS: dict[MemoryType, str] = {t: t.value for t in MemoryType}


@dataclass
class MemoryEntry:
    """一条记忆条目。

    Attributes:
        name: kebab-case slug，唯一标识
        memory_type: 记忆类型
        description: 一行摘要，用于相关性判断
        body: Markdown 正文
        created_at: 创建时间
        updated_at: 更新时间
        last_accessed_at: 最后访问时间（None 表示从未访问）
        confidence: 置信度 0.0~1.0
        tags: 标签列表
    """

    name: str
    memory_type: MemoryType
    description: str
    body: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: Optional[datetime] = None
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)

    def to_frontmatter_str(self) -> str:
        """序列化为 YAML Frontmatter + Markdown 格式。"""
        meta: dict = {
            "name": self.name,
            "description": self.description,
            "metadata": {
                "type": self.memory_type.value,
                "created_at": self._format_dt(self.created_at),
                "updated_at": self._format_dt(self.updated_at),
                "confidence": self.confidence,
                "tags": list(self.tags),
            },
        }
        if self.last_accessed_at is not None:
            meta["metadata"]["last_accessed_at"] = self._format_dt(
                self.last_accessed_at
            )

        post = frontmatter.Post(self.body, **meta)
        result = frontmatter.dumps(post)
        return result

    @classmethod
    def from_frontmatter_str(cls, content: str) -> MemoryEntry:
        """从 YAML Frontmatter + Markdown 字符串反序列化。"""
        post = frontmatter.loads(content)
        md = post.metadata
        meta = md.get("metadata", {})

        name = md.get("name", "")
        description = md.get("description", "")
        memory_type = MemoryType(meta.get("type", MemoryType.SESSION.value))
        body = post.content

        created_at = cls._parse_dt(meta.get("created_at")) or datetime.now()
        updated_at = cls._parse_dt(meta.get("updated_at")) or datetime.now()
        last_accessed_at = cls._parse_dt(meta.get("last_accessed_at"))
        confidence = float(meta.get("confidence", 1.0))
        tags = list(meta.get("tags", []))

        return cls(
            name=name,
            memory_type=memory_type,
            description=description,
            body=body,
            created_at=created_at,
            updated_at=updated_at,
            last_accessed_at=last_accessed_at,
            confidence=confidence,
            tags=tags,
        )

    @staticmethod
    def _format_dt(dt: datetime) -> str:
        """格式化为 ISO 8601 字符串。"""
        return dt.isoformat()

    @staticmethod
    def _parse_dt(value: object) -> Optional[datetime]:
        """从字符串解析 datetime，失败时返回 None。"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None
        return None


class MemoryStore:
    """两级记忆存储 CRUD。

    全局存储路径：~/.codeagent/memory/（user/feedback 类型）
    项目存储路径：.codeagent/memory/（project/code_pattern/session 类型）
    """

    def __init__(
        self,
        global_root: Path,
        project_root: Optional[Path] = None,
    ) -> None:
        """初始化 MemoryStore。

        Args:
            global_root: 全局存储根目录（~/.codeagent/memory/）
            project_root: 项目存储根目录（.codeagent/memory/），可选
        """
        self._global_root = global_root
        self._project_root = project_root

    # ── 公共接口 ──────────────────────────────────────────────

    def save(self, entry: MemoryEntry, scope: str = "auto") -> Path:
        """写入记忆文件并更新 MEMORY.md 索引。

        Args:
            entry: 记忆条目
            scope: "auto"（user/feedback→全局，其余→项目）、"global"、"project"

        Returns:
            Path: 写入的文件路径

        Raises:
            ValueError: scope="project" 但 project_root 未设置
        """
        root = self._resolve_root(entry.memory_type, scope)
        file_path = root / _TYPE_DIRS[entry.memory_type] / f"{entry.name}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        content = entry.to_frontmatter_str()
        file_path.write_text(content, encoding="utf-8")

        # 更新该层的 MEMORY.md 索引
        scope_label = "project" if root == self._project_root else "global"
        self.update_index(scope_label)

        logger.debug("Memory saved: %s (%s)", entry.name, scope_label)
        return file_path

    def load(self, name: str) -> Optional[MemoryEntry]:
        """按 name 查找记忆，项目层优先于全局层。

        找到时会更新 last_accessed_at 并写回文件。

        Args:
            name: 记忆条目的 name（kebab-case）

        Returns:
            MemoryEntry 或 None
        """
        # 项目层优先
        if self._project_root:
            entry = self._load_from_root(self._project_root, name)
            if entry is not None:
                self._touch_last_accessed(self._project_root, entry)
                return entry

        # 全局层
        entry = self._load_from_root(self._global_root, name)
        if entry is not None:
            self._touch_last_accessed(self._global_root, entry)
        return entry

    def delete(self, name: str) -> bool:
        """删除记忆文件并从 MEMORY.md 移除对应条目。

        Args:
            name: 记忆条目的 name

        Returns:
            bool: 是否删除了任何文件
        """
        deleted = False
        roots: list[Path] = [self._global_root]
        if self._project_root:
            roots.append(self._project_root)

        for root in roots:
            for type_dir in _TYPE_DIRS.values():
                file_path = root / type_dir / f"{name}.md"
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
                    logger.debug("Memory deleted: %s from %s", name, root)

        if deleted:
            self.update_index("global")
            if self._project_root:
                self.update_index("project")

        return deleted

    def list_all(
        self,
        memory_type: Optional[MemoryType] = None,
        scope: str = "both",
    ) -> list[MemoryEntry]:
        """列出所有记忆，项目层结果排在全局层之前。

        Args:
            memory_type: 筛选特定类型，None 表示全部
            scope: "global"、"project" 或 "both"

        Returns:
            list[MemoryEntry]: 记忆条目列表
        """
        result: list[MemoryEntry] = []

        # 项目层优先
        if scope in ("project", "both") and self._project_root:
            result.extend(self._list_from_root(self._project_root, memory_type))

        # 全局层
        if scope in ("global", "both"):
            result.extend(self._list_from_root(self._global_root, memory_type))

        return result

    def update_index(self, scope: str) -> None:
        """重建 MEMORY.md 索引。

        超过 200 行时，按 session > project > code_pattern 顺序截断最旧条目。

        Args:
            scope: "global" 或 "project"
        """
        root = self._resolve_root_by_scope(scope)
        index_path = root / "MEMORY.md"

        lines: list[str] = [_INDEX_HEADER]
        type_order = ["user", "feedback", "project", "code_pattern", "session"]

        for type_name in type_order:
            type_dir = root / type_name
            if not type_dir.is_dir():
                continue
            for fpath in sorted(type_dir.iterdir()):
                if fpath.suffix != ".md":
                    continue
                try:
                    entry = MemoryEntry.from_frontmatter_str(
                        fpath.read_text(encoding="utf-8")
                    )
                    lines.append(
                        f"- [{entry.name}]({type_name}/{entry.name}.md)"
                        f" — {entry.description}"
                    )
                except Exception as exc:
                    logger.warning("Skipping invalid memory file %s: %s", fpath, exc)

        # 超过 200 行时截断
        max_lines = 200
        if len(lines) > max_lines:
            lines = self._truncate_index(lines, max_lines)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def archive_expired_sessions(self, days: int = 7) -> int:
        """将超过 days 天未访问的 session 类型记忆移至 archived/ 子目录。

        Args:
            days: 过期天数

        Returns:
            int: 归档数量
        """
        count = 0
        roots: list[Path] = [self._global_root]
        if self._project_root:
            roots.append(self._project_root)

        cutoff = datetime.now() - timedelta(days=days)

        for root in roots:
            session_dir = root / "session"
            archived_dir = root / "archived" / "session"
            if not session_dir.is_dir():
                continue

            for fpath in list(session_dir.iterdir()):
                if fpath.suffix != ".md":
                    continue
                try:
                    entry = MemoryEntry.from_frontmatter_str(
                        fpath.read_text(encoding="utf-8")
                    )
                    # 判断是否过期：last_accessed_at 或 created_at
                    ref_time = entry.last_accessed_at or entry.created_at
                    if ref_time < cutoff:
                        archived_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(fpath), str(archived_dir / fpath.name))
                        count += 1
                        logger.debug(
                            "Session archived: %s (last accessed: %s)",
                            entry.name, ref_time.isoformat(),
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to archive session %s: %s", fpath, exc
                    )

        if count > 0:
            self.update_index("global")
            if self._project_root:
                self.update_index("project")

        return count

    # ── 内部方法 ──────────────────────────────────────────────

    def _resolve_root(self, memory_type: MemoryType, scope: str) -> Path:
        """根据类型和作用域决定存储根目录。"""
        if scope == "global":
            return self._global_root
        if scope == "project":
            if self._project_root is None:
                raise ValueError(
                    "scope='project' requires project_root to be set"
                )
            return self._project_root

        # scope == "auto"
        if memory_type in (MemoryType.USER, MemoryType.FEEDBACK):
            return self._global_root
        if self._project_root is not None:
            return self._project_root
        return self._global_root

    def _resolve_root_by_scope(self, scope: str) -> Path:
        """根据 scope 获取根目录用于索引操作。"""
        if scope == "project":
            if self._project_root is None:
                raise ValueError("scope='project' requires project_root to be set")
            return self._project_root
        return self._global_root

    def _load_from_root(self, root: Path, name: str) -> Optional[MemoryEntry]:
        """在指定的 root 下查找 name（所有类型目录）。"""
        for type_dir in _TYPE_DIRS.values():
            file_path = root / type_dir / f"{name}.md"
            if file_path.exists():
                try:
                    return MemoryEntry.from_frontmatter_str(
                        file_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:
                    logger.warning("Failed to load memory %s: %s", file_path, exc)
                    return None
        return None

    def _touch_last_accessed(self, root: Path, entry: MemoryEntry) -> None:
        """更新内存中条目的 last_accessed_at 并写回文件。"""
        entry.last_accessed_at = datetime.now()
        file_path = (
            root / _TYPE_DIRS[entry.memory_type] / f"{entry.name}.md"
        )
        try:
            file_path.write_text(entry.to_frontmatter_str(), encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "Failed to update last_accessed_at for %s: %s",
                entry.name, exc,
            )

    def _list_from_root(
        self,
        root: Path,
        memory_type: Optional[MemoryType] = None,
    ) -> list[MemoryEntry]:
        """列出指定 root 下的记忆条目。"""
        results: list[MemoryEntry] = []
        types_to_scan = (
            [memory_type] if memory_type else list(MemoryType)
        )

        for mt in types_to_scan:
            type_dir = root / mt.value
            if not type_dir.is_dir():
                continue
            for fpath in sorted(type_dir.iterdir()):
                if fpath.suffix != ".md":
                    continue
                try:
                    entry = MemoryEntry.from_frontmatter_str(
                        fpath.read_text(encoding="utf-8")
                    )
                    results.append(entry)
                except Exception as exc:
                    logger.warning(
                        "Skipping invalid file %s: %s", fpath, exc
                    )

        return results

    def _truncate_index(
        self, lines: list[str], max_lines: int
    ) -> list[str]:
        """截断索引至 max_lines 行。

        按 SESSION > PROJECT > CODE_PATTERN 顺序删除最旧条目，
        user 和 feedback 类型永不截断。
        """
        # 分离 header 和条目
        entries: list[tuple[str, str, str, str]] = []  # (full_line, name, type, description)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            # 解析行格式：- [name](type/name.md) — description
            if line.startswith("- [") and "](" in line:
                try:
                    name = line[3:].split("]")[0]
                    rest = line.split("](")[1]
                    link_path = rest.split(")")[0]
                    entry_type = link_path.split("/")[0]
                    desc = rest.split("—", 1)[1].strip() if "—" in rest else ""
                    entries.append((line, name, entry_type, desc))
                except (IndexError, ValueError):
                    entries.append((line, "", "", ""))

        header = lines[:1]

        # 按类型分组
        user_feedback: list[tuple[str, str, str, str]] = []
        code_pattern: list[tuple[str, str, str, str]] = []
        project: list[tuple[str, str, str, str]] = []
        session: list[tuple[str, str, str, str]] = []
        other: list[tuple[str, str, str, str]] = []

        for e in entries:
            t = e[2]
            if t in ("user", "feedback"):
                user_feedback.append(e)
            elif t == "code_pattern":
                code_pattern.append(e)
            elif t == "project":
                project.append(e)
            elif t == "session":
                session.append(e)
            else:
                other.append(e)

        # 按类型排序（保持原有顺序，但每个类型内部按文件系统顺序）
        # 从最旧类型开始截断
        truncatable = {
            MemoryType.SESSION: session,
            MemoryType.CODE_PATTERN: code_pattern,
            MemoryType.PROJECT: project,
        }

        total = len(entries)
        for mt in _TRUNCATION_PRIORITY:
            if total <= max_lines:
                break
            typed_entries = truncatable.get(mt, [])
            # 从末尾开始删除（旧文件先列出）
            while typed_entries and total > max_lines:
                typed_entries.pop()  # 移除最后一个（最旧）
                total -= 1

        # 重新组装
        result: list[str] = list(header)
        for e in user_feedback + other + project + code_pattern + session:
            if e in entries or e in user_feedback:
                result.append(e[0])

        # 确保 user_feedback 始终保留
        for group in [user_feedback]:
            for e in group:
                line_text = e[0]
                if line_text not in result:
                    result.append(line_text)

        # 再重建确保顺序正确
        ordered: list[str] = list(header)
        seen: set[str] = set()
        for e in user_feedback + other + project + code_pattern + session:
            if e[0] not in seen:
                # 检查是否在截断后保留
                if e in user_feedback or e in other or e in project or e in code_pattern or e in session:
                    ordered.append(e[0])
                    seen.add(e[0])

        return ordered[:max_lines]
