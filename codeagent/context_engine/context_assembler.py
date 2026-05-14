"""ContextAssembler — 上下文窗口预算分配和格式化。

将 FileTreeIndexer 和其他模块的输出组装为 LLM-ready 消息格式，
支持 token 预算分配和裁剪。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from codeagent.gateway.context_gateway import CodeSnippet, ContextPackage

logger = logging.getLogger(__name__)

# 默认预算分配比例
FILE_TREE_RATIO = 0.05
RELATED_CODE_RATIO = 0.40
CURRENT_FILE_RATIO = 0.30
DEPENDENCY_RATIO = 0.15
SYSTEM_RESERVED_RATIO = 0.10

# 默认编码
_ENCODING = "cl100k_base"

# 裁剪优先级顺序（列表末尾最优先保留）
_TRIM_ORDER = ["related_code", "dependency", "file_tree"]


@dataclass
class BudgetAllocation:
    """预算分配项。

    Attributes:
        section: 区块名称
        budget: 分配的 token 预算
        used: 实际使用的 token 数
        trimmed: 是否被裁剪
    """

    section: str
    budget: int
    used: int = 0
    trimmed: bool = False


@dataclass
class BudgetReport:
    """预算使用情况报告。

    Attributes:
        total_budget: 总预算（token）
        total_used: 总使用量
        allocations: 各区块分配详情
    """

    total_budget: int
    total_used: int
    allocations: list[BudgetAllocation] = field(default_factory=list)


class ContextAssembler:
    """上下文组装器。

    将 ContextPackage 组装为 LLM-ready 的 XML 格式消息，
    通过预算分配和裁剪策略控制上下文窗口大小。

    Attributes:
        total_budget: 总 token 预算
    """

    def __init__(self, total_budget: int = 8000) -> None:
        """初始化 ContextAssembler。

        Args:
            total_budget: 总 token 预算（默认 8000）
        """
        self.total_budget = total_budget
        self._tokenizer = tiktoken.get_encoding(_ENCODING)
        self._last_report: BudgetReport | None = None

    def assemble(self, package: ContextPackage) -> str:
        """组装上下文为 LLM-ready XML 格式。

        根据预算分配组装各个区块，在超出预算时按策略裁剪。

        Args:
            package: 上下文数据包（包含文件树、相关代码、依赖信息）

        Returns:
            str: XML 格式的上下文文本
        """
        # ── 计算各区块预算 ──────────────────────────────
        system_reserve = int(self.total_budget * SYSTEM_RESERVED_RATIO)
        usable = self.total_budget - system_reserve

        file_tree_budget = int(usable * FILE_TREE_RATIO)
        related_code_budget = int(usable * RELATED_CODE_RATIO)
        current_file_budget = int(usable * CURRENT_FILE_RATIO)
        dependency_budget = int(usable * DEPENDENCY_RATIO)

        allocations = [
            BudgetAllocation(section="file_tree", budget=file_tree_budget),
            BudgetAllocation(section="related_code", budget=related_code_budget),
            BudgetAllocation(section="current_file", budget=current_file_budget),
            BudgetAllocation(section="dependency", budget=dependency_budget),
        ]

        parts: dict[str, str] = {}

        # ── 组装文件树 ──────────────────────────────────
        file_tree_str = self._format_file_tree(package.file_tree)
        parts["file_tree"] = self._trim_to_budget(
            file_tree_str, file_tree_budget, allocations[0]
        )

        # ── 组装相关代码 ────────────────────────────────
        related_str = self._format_related_code(package.related_code)
        parts["related_code"] = self._trim_to_budget(
            related_str, related_code_budget, allocations[1]
        )

        # ── 组装当前文件 ────────────────────────────────
        current_file_str = self._format_current_file(package.file_tree)
        parts["current_file"] = self._trim_to_budget(
            current_file_str, current_file_budget, allocations[2]
        )

        # ── 组装依赖信息 ────────────────────────────────
        dep_str = self._format_dependency(package.dependency_info)
        parts["dependency"] = self._trim_to_budget(
            dep_str, dependency_budget, allocations[3]
        )

        # ── 裁剪：超出预算时按优先级裁剪 ────────────────
        self._crop_if_needed(parts, allocations)

        # ── 组装最终输出 ────────────────────────────────
        output_parts: list[str] = []

        if parts.get("file_tree"):
            output_parts.append(
                f"<project_tree>\n{parts['file_tree']}\n</project_tree>"
            )

        if parts.get("current_file"):
            output_parts.append(
                f"<current_file>\n{parts['current_file']}\n</current_file>"
            )

        if parts.get("related_code"):
            output_parts.append(
                f"<related_code>\n{parts['related_code']}\n</related_code>"
            )

        if parts.get("dependency"):
            output_parts.append(
                f"<dependency_info>\n{parts['dependency']}\n</dependency_info>"
            )

        total_used = sum(
            self._count_tokens(v or "") for v in parts.values()
        )
        self._last_report = BudgetReport(
            total_budget=self.total_budget,
            total_used=total_used,
            allocations=allocations,
        )

        return "\n\n".join(output_parts)

    def get_budget_report(self) -> BudgetReport | None:
        """获取最近一次组装的预算使用报告。

        Returns:
            BudgetReport: 预算报告，未组装时返回 None
        """
        return self._last_report

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数。"""
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _trim_to_budget(
        self, text: str, budget: int, allocation: BudgetAllocation
    ) -> str | None:
        """将文本裁剪到预算内（按行裁剪）。"""
        if not text:
            return None

        tokens = self._count_tokens(text)
        allocation.used = tokens

        if tokens <= budget:
            return text

        # 按行裁剪，保留前部
        lines = text.split("\n")
        trimmed: list[str] = []
        current_tokens = 0

        # 始终保留标题行（如果有）
        for line in lines:
            line_tokens = self._count_tokens(line + "\n")
            if current_tokens + line_tokens <= budget:
                trimmed.append(line)
                current_tokens += line_tokens
            else:
                break

        allocation.trimmed = True
        allocation.used = current_tokens

        if not trimmed:
            # 至少保留一行
            trimmed = lines[:1]

        result = "\n".join(trimmed)
        if len(lines) > len(trimmed):
            result += f"\n... ({len(lines) - len(trimmed)} more lines cropped)"

        return result

    def _crop_if_needed(
        self,
        parts: dict[str, str | None],
        allocations: list[BudgetAllocation],
    ) -> None:
        """当总预算超出时，按优先级裁剪区块。

        裁剪顺序：相关代码 → 依赖信息 → 文件树
        """
        total = sum(
            self._count_tokens(v or "") for v in parts.values()
        )
        if total <= self.total_budget:
            return

        alloc_map = {a.section: a for a in allocations}

        for section in _TRIM_ORDER:
            if total <= self.total_budget:
                break
            content = parts.get(section)
            if not content:
                continue

            budget = alloc_map[section].budget

            # 逐步减少预算直到总预算符合
            while total > self.total_budget and budget > 0:
                budget = max(budget // 2, 10)
                parts[section] = self._trim_to_budget(
                    content, budget, alloc_map[section]
                )
                total = sum(
                    self._count_tokens(v or "") for v in parts.values()
                )

    def _format_file_tree(self, file_tree: Any) -> str:
        """格式化文件树为文本。"""
        if file_tree is None:
            return ""
        if isinstance(file_tree, dict):
            return self._dict_tree_to_text(file_tree)
        return str(file_tree)

    def _dict_tree_to_text(self, tree: dict[str, Any], prefix: str = "") -> str:
        """将嵌套字典格式的文件树转为文本。"""
        lines: list[str] = []
        name = tree.get("name", "")
        node_type = tree.get("type", "")

        if node_type == "directory":
            summary = tree.get("_summary")
            children = tree.get("children", [])

            if summary == "depth exceeded":
                lines.append(f"{prefix}{name}/ ({len(children)} sub-items)")
                return "\n".join(lines)
            if summary == "permission denied":
                lines.append(f"{prefix}{name}/ [denied]")
                return "\n".join(lines)

            lines.append(f"{prefix}{name}/")
            for child in children:
                child_text = self._dict_tree_to_text(child, prefix + "  ")
                if child_text:
                    lines.append(child_text)
        elif node_type == "file":
            size = _format_size(tree.get("size", 0))
            lines.append(f"{prefix}{name}  {size}")

        return "\n".join(lines)

    def _format_related_code(
        self, related_code: list[CodeSnippet]
    ) -> str:
        """格式化相关代码片段。"""
        if not related_code:
            return ""

        parts: list[str] = []
        for i, snippet in enumerate(related_code, 1):
            header = (
                f"[{i}] {snippet.file_path}:{snippet.start_line}-"
                f"{snippet.end_line}"
            )
            if snippet.score > 0:
                header += f" (score: {snippet.score:.2f})"
            parts.append(header)
            parts.append("```")
            parts.append(snippet.code.rstrip("\n"))
            parts.append("```")

        return "\n".join(parts)

    def _format_current_file(self, file_tree: Any) -> str:
        """从文件树中提取当前文件信息（简化版：返回文件树中的 .py 文件列表）。"""
        if file_tree is None:
            return ""
        if isinstance(file_tree, dict):
            py_files = self._find_python_files(file_tree)
            if not py_files:
                return ""
            lines = ["Key Python files in project:"]
            for f in py_files:
                lines.append(f"  {f}")
            return "\n".join(lines)
        return str(file_tree)

    def _find_python_files(
        self, node: dict[str, Any], prefix: str = ""
    ) -> list[str]:
        """递归查找 Python 文件路径。"""
        files: list[str] = []
        name = node.get("name", "")
        path = node.get("path", "")

        if node.get("type") == "directory":
            for child in node.get("children", []):
                files.extend(self._find_python_files(child, prefix))
        elif node.get("type") == "file" and name.endswith(".py"):
            files.append(path)

        return files

    def _format_dependency(self, dep_info: dict[str, Any]) -> str:
        """格式化依赖信息。"""
        if not dep_info:
            return ""

        parts: list[str] = []
        for key, value in dep_info.items():
            if isinstance(value, list):
                items = "\n".join(f"  - {item}" for item in value)
                parts.append(f"{key}:\n{items}")
            elif isinstance(value, dict):
                items = "\n".join(
                    f"  - {k}: {v}" for k, v in value.items()
                )
                parts.append(f"{key}:\n{items}")
            else:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)


def _format_size(size_bytes: int) -> str:
    """格式化文件大小。"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"
