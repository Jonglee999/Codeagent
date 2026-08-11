"""ContextAssembler — 上下文窗口预算分配和格式化。

将 FileTreeIndexer 和其他模块的输出组装为 LLM-ready 消息格式，
支持 token 预算分配和裁剪。

Phase 2 增强：
- 支持符号表摘要的 XML 格式输出
- 相关代码按 score 从低到高裁剪
- 文件树折叠嵌套超过 3 层的目录
- 当前文件保留函数/类签名 + import 区域
- 预算报告增加符号表和依赖图的 token 统计
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from typing import Optional

from codeagent.gateway.context_gateway import CodeSnippet, ContextPackage
from codeagent.gateway.memory_gateway import IMemoryGateway

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

# 文件树折叠深度
_MAX_TREE_DEPTH = 3

# 当前文件签名保留行数限制
_MAX_SIGNATURE_LINES = 200


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
        symbol_count: 符号数量（Phase 2 新增）
        dependency_count: 依赖项数量（Phase 2 新增）
    """

    total_budget: int
    total_used: int
    allocations: list[BudgetAllocation] = field(default_factory=list)
    symbol_count: int = 0
    dependency_count: int = 0
    tokenizer: str = _ENCODING
    trim_reasons: list[str] = field(default_factory=list)


class ContextAssembler:
    """上下文组装器。

    将 ContextPackage 组装为 LLM-ready 的 XML 格式消息，
    通过预算分配和裁剪策略控制上下文窗口大小。
    """

    def __init__(
        self,
        total_budget: int = 8000,
        memory_gateway: Optional[IMemoryGateway] = None,
        memory_token_budget: int = 800,
        model_name: str = "",
    ) -> None:
        """初始化 ContextAssembler。

        Args:
            total_budget: 总 token 预算（默认 8000）
            memory_gateway: 记忆系统 Gateway，None 时跳过 Memory 层
            memory_token_budget: Memory 层 token 预算
        """
        self.total_budget = total_budget
        self._memory_gateway = memory_gateway
        self._memory_token_budget = memory_token_budget
        self._tokenizer, self._tokenizer_name = self._resolve_tokenizer(model_name)
        self._last_report: BudgetReport | None = None

    def assemble(self, package: ContextPackage) -> str:
        """组装上下文为 LLM-ready XML 格式。

        根据预算分配组装各个区块，在超出预算时按策略裁剪。

        Args:
            package: 上下文数据包（包含文件树、相关代码、依赖信息、符号表）

        Returns:
            str: XML 格式的上下文文本
        """
        # Preserve a structurally valid minimal envelope for callers that use
        # zero as a sentinel while reporting zero payload tokens.
        if self.total_budget <= 0:
            allocations = [
                BudgetAllocation(section=section, budget=0, trimmed=True)
                for section in ("file_tree", "related_code", "current_file", "dependency")
            ]
            self._last_report = BudgetReport(
                total_budget=self.total_budget,
                total_used=0,
                allocations=allocations,
                symbol_count=len(package.symbol_table),
                dependency_count=self._count_dep_items(package.dependency_info),
                tokenizer=self._tokenizer_name,
                trim_reasons=["zero_context_budget"],
            )
            return "<project_tree>\n\n</project_tree>" if package.file_tree else ""

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

        # ── 组装文件树（折叠深层目录） ──────────────────
        folded_tree = self._fold_file_tree(package.file_tree)
        file_tree_str = self._format_file_tree(folded_tree)
        parts["file_tree"] = self._trim_to_budget(
            file_tree_str, file_tree_budget, allocations[0]
        )

        # ── 组装相关代码（按 score 排序裁剪） ──────────
        parts["related_code"] = self._trim_related_code_by_score(
            package.related_code, related_code_budget, allocations[1]
        )

        # ── 组装当前文件（签名+import 保留） ────────────
        current_file_str = self._format_current_file(package.file_tree)
        parts["current_file"] = self._trim_current_file_to_signatures(
            current_file_str, current_file_budget, allocations[2]
        )

        # ── 组装依赖信息 + 符号表 ──────────────────────
        dep_str = self._format_dependency(package.dependency_info, package.symbol_table)
        parts["dependency"] = self._trim_to_budget(
            dep_str, dependency_budget, allocations[3]
        )

        # ── 裁剪：超出总预算时按优先级裁剪 ──────────────
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

        symbol_count = len(package.symbol_table)
        dep_count = self._count_dep_items(package.dependency_info)

        self._last_report = BudgetReport(
            total_budget=self.total_budget,
            total_used=total_used,
            allocations=allocations,
            symbol_count=symbol_count,
            dependency_count=dep_count,
            tokenizer=self._tokenizer_name,
            trim_reasons=[
                f"{allocation.section}_exceeded_allocation"
                for allocation in allocations
                if allocation.trimmed
            ],
        )

        return "\n\n".join(output_parts)

    def get_budget_report(self) -> BudgetReport | None:
        """获取最近一次组装的预算使用报告。"""
        return self._last_report

    # ── Phase 6.5 Memory 层注入 ──────────────────────────────────────

    async def assemble_memory_section(self, task_description: str) -> str:
        """检索相关记忆并组装为 System Prompt Memory 层。

        在 memory_gateway=None 或无可检索记忆时返回空字符串。

        Args:
            task_description: 任务描述，用作检索查询

        Returns:
            "## Relevant Memories\n{memory_xml}" 或 ""
        """
        if not self._memory_gateway:
            return ""

        try:
            memory_xml = await self._memory_gateway.recall(
                query=task_description,
                token_budget=self._memory_token_budget,
            )
            if not memory_xml:
                return ""
            return f"## Relevant Memories\n\n{memory_xml}"
        except Exception as exc:
            logger.warning("Memory layer injection failed (non-blocking): %s", exc)
            return ""

    # ── Token 计数 ─────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数。"""
        return len(self._tokenizer.encode(text, disallowed_special=()))

    @staticmethod
    def _resolve_tokenizer(model_name: str) -> tuple[Any, str]:
        """Use a provider/model tokenizer when tiktoken knows it, else a safe fallback."""
        candidate = model_name.split("/", 1)[-1] if model_name else ""
        if candidate:
            try:
                encoding = tiktoken.encoding_for_model(candidate)
                return encoding, encoding.name
            except KeyError:
                pass
        encoding = tiktoken.get_encoding(_ENCODING)
        return encoding, f"{_ENCODING}:fallback"

    # ── 通用裁剪 ───────────────────────────────────────────────────────────

    def _trim_to_budget(
        self, text: str, budget: int, allocation: BudgetAllocation
    ) -> str | None:
        """将文本裁剪到预算内（按行裁剪）。"""
        if not text or budget <= 0:
            allocation.trimmed = bool(text)
            allocation.used = 0
            return None

        tokens = self._count_tokens(text)
        allocation.used = tokens

        if tokens <= budget:
            return text

        lines = text.split("\n")
        trimmed: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self._count_tokens(line + "\n")
            if current_tokens + line_tokens <= budget:
                trimmed.append(line)
                current_tokens += line_tokens
            else:
                break

        allocation.trimmed = True
        allocation.used = current_tokens

        result = "\n".join(trimmed)
        if len(lines) > len(trimmed):
            result += f"\n... ({len(lines) - len(trimmed)} more lines cropped)"
        result = self._hard_trim(result, budget)
        allocation.used = self._count_tokens(result)
        return result or None

    def _hard_trim(self, text: str, budget: int) -> str:
        """Token-level final guard for very small budgets and long single lines."""
        if not text or budget <= 0:
            return ""
        encoded = self._tokenizer.encode(text, disallowed_special=())
        if len(encoded) <= budget:
            return text
        return self._tokenizer.decode(encoded[:budget])

    def _crop_if_needed(
        self,
        parts: dict[str, str | None],
        allocations: list[BudgetAllocation],
    ) -> None:
        """当总预算超出时，按优先级裁剪区块。"""
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

            budget = min(
                alloc_map[section].budget,
                self._count_tokens(content),
            )

            while total > self.total_budget and budget > 0:
                next_budget = budget // 2
                if next_budget >= budget:
                    next_budget = 0
                budget = next_budget
                parts[section] = self._trim_to_budget(
                    content, budget, alloc_map[section]
                )
                total = sum(
                    self._count_tokens(v or "") for v in parts.values()
                )

        # Defensive guard: custom formatters must never be able to exceed the
        # advertised budget, even when it is only a handful of tokens.
        if total > self.total_budget:
            for section in [*_TRIM_ORDER, "current_file"]:
                if total <= self.total_budget:
                    break
                parts[section] = None
                allocation = alloc_map[section]
                allocation.trimmed = True
                allocation.used = 0
                total = sum(self._count_tokens(v or "") for v in parts.values())

    # ── Phase 2: 文件树折叠 ───────────────────────────────────────────────

    def _fold_file_tree(self, file_tree: Any, depth: int = 0) -> Any:
        """折叠嵌套超过 _MAX_TREE_DEPTH 层的目录。

        超过深度时，将子目录折叠为摘要形式。
        """
        if file_tree is None:
            return None
        if not isinstance(file_tree, dict):
            return file_tree

        if file_tree.get("type") != "directory":
            return file_tree

        children = file_tree.get("children", [])
        if depth >= _MAX_TREE_DEPTH and children:
            # 折叠：仅保留子项的数量统计
            file_count = sum(1 for c in children if c.get("type") == "file")
            dir_count = sum(1 for c in children if c.get("type") == "directory")
            summary_parts = []
            if file_count:
                summary_parts.append(f"{file_count} files")
            if dir_count:
                summary_parts.append(f"{dir_count} subdirectories")
            file_tree = dict(file_tree)
            file_tree["_summary"] = "depth exceeded"
            file_tree["children"] = [
                {"name": f"[{', '.join(summary_parts)}]", "type": "summary"}
            ] if summary_parts else []
            return file_tree

        folded_children = []
        for child in children:
            folded = self._fold_file_tree(child, depth + 1)
            if folded is not None:
                folded_children.append(folded)

        result = dict(file_tree)
        result["children"] = folded_children
        return result

    # ── Phase 2: 相关代码按 score 裁剪 ────────────────────────────────────

    def _trim_related_code_by_score(
        self,
        related_code: list[CodeSnippet],
        budget: int,
        allocation: BudgetAllocation,
    ) -> str | None:
        """将相关代码按 score 从低到高移除，直到符合预算。"""
        if not related_code or budget <= 0:
            allocation.trimmed = bool(related_code)
            allocation.used = 0
            return None

        # 先格式化为文本
        full_text = self._format_related_code(related_code)
        tokens = self._count_tokens(full_text)
        allocation.used = tokens

        if tokens <= budget:
            return full_text

        # 按 score 升序排列（最低分最优先被移除）
        sorted_snippets = sorted(related_code, key=lambda s: s.score)
        remaining = list(sorted_snippets)

        while remaining:
            text = self._format_related_code(remaining)
            t = self._count_tokens(text)
            if t <= budget:
                allocation.trimmed = True
                allocation.used = t
                if len(remaining) < len(related_code):
                    removed = len(related_code) - len(remaining)
                    text += f"\n... ({removed} lower-score snippets removed)"
                text = self._hard_trim(text, budget)
                allocation.used = self._count_tokens(text)
                return text or None
            # 移除最低分的代码片段
            remaining = remaining[1:]

        # 全都不行时保留最高分的片段
        best = [max(related_code, key=lambda s: s.score)]
        text = self._format_related_code(best)
        allocation.trimmed = True
        removed = len(related_code) - 1
        text += f"\n... ({removed} lower-score snippets removed)"
        text = self._hard_trim(text, budget)
        allocation.used = self._count_tokens(text)
        return text or None

    # ── Phase 2: 当前文件裁剪为签名+import ────────────────────────────────

    def _trim_current_file_to_signatures(
        self, text: str, budget: int, allocation: BudgetAllocation
    ) -> str | None:
        """将当前文件内容裁剪为仅保留函数/类签名 + import 区域。"""
        if not text or budget <= 0:
            allocation.trimmed = bool(text)
            allocation.used = 0
            return None

        tokens = self._count_tokens(text)
        allocation.used = tokens

        if tokens <= budget:
            return text

        # 提取 import 行、函数/类签名行
        lines = text.split("\n")
        significant_lines: list[str] = []
        removed_count = 0
        current_tokens = 0

        for line in lines:
            stripped = line.strip()
            keep = False
            if stripped.startswith(("import ", "from ", "# ")):
                keep = True
            elif stripped.startswith(("def ", "class ", "async def ")):
                keep = True
            elif stripped.startswith(("@", "    def ", "    class ")):
                keep = True
            elif not stripped:
                keep = True  # 保留空行增强可读性

            if keep:
                line_tokens = self._count_tokens(line + "\n")
                if current_tokens + line_tokens <= budget:
                    significant_lines.append(line)
                    current_tokens += line_tokens
                else:
                    break
            else:
                removed_count += 1

        allocation.trimmed = True
        allocation.used = current_tokens

        result = "\n".join(significant_lines)
        if removed_count > 0:
            result += f"\n... ({removed_count} body lines cropped to signatures only)"
        elif len(lines) > len(significant_lines):
            result += f"\n... ({len(lines) - len(significant_lines)} lines cropped)"

        result = self._hard_trim(result, budget)
        allocation.used = self._count_tokens(result)
        return result or None

    # ── Phase 2: 符号表 XML 格式化 ─────────────────────────────────────────

    def _format_symbol_table_xml(self, symbols: list[dict]) -> str:
        """将符号表格式化为 XML。

        Returns:
            形如 <symbols><symbol name="..." kind="..." file="..." line="..."/></symbols>
        """
        if not symbols:
            return ""

        parts = ["<symbols>"]
        for sym in symbols:
            name = sym.get("name", "")
            kind = sym.get("kind", "")
            file_path = sym.get("file_path", "")
            line = sym.get("start_line", 0)
            parts.append(
                f'  <symbol name="{name}" kind="{kind}" '
                f'file="{file_path}" line="{line}"/>'
            )
        parts.append("</symbols>")
        return "\n".join(parts)

    def _count_dep_items(self, dep_info: dict) -> int:
        """统计依赖项数量。"""
        count = 0
        for value in dep_info.values():
            if isinstance(value, list):
                count += len(value)
            elif isinstance(value, dict):
                count += len(value)
            elif value:
                count += 1
        return count

    # ── 格式化方法 ─────────────────────────────────────────────────────────

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
                summary_text = ""
                for child in children:
                    child_name = child.get("name", "")
                    if child_name:
                        summary_text = child_name
                lines.append(f"{prefix}{name}/ ({summary_text})")
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
        elif node_type == "summary":
            lines.append(f"{prefix}{name}")

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
        """从文件树中提取当前文件信息。"""
        if file_tree is None:
            return ""
        if isinstance(file_tree, dict):
            py_files = self._find_python_files(file_tree)
            if not py_files:
                return ""
            lines = ["Key Python files in project:"]
            for f in py_files[:50]:  # 限制最多 50 个
                lines.append(f"  {f}")
            if len(py_files) > 50:
                lines.append(f"  ... ({len(py_files) - 50} more files)")
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

    def _format_dependency(
        self,
        dep_info: dict[str, Any],
        symbol_table: list[dict] | None = None,
    ) -> str:
        """格式化依赖信息和符号表。

        Phase 2 增强：将符号表以 XML 形式嵌入 dependency_info 区块。
        """
        parts: list[str] = []

        # 格式化符号表 XML
        if symbol_table:
            sym_xml = self._format_symbol_table_xml(symbol_table)
            if sym_xml:
                parts.append(sym_xml)

        # 格式化依赖信息
        if dep_info:
            dep_parts: list[str] = []
            for key, value in dep_info.items():
                if isinstance(value, list):
                    items = "\n".join(f"  - {item}" for item in value)
                    dep_parts.append(f"{key}:\n{items}")
                elif isinstance(value, dict):
                    items = "\n".join(
                        f"  - {k}: {v}" for k, v in value.items()
                    )
                    dep_parts.append(f"{key}:\n{items}")
                else:
                    dep_parts.append(f"{key}: {value}")

            if dep_parts:
                if parts:
                    parts.append("")  # 空行分隔
                parts.append("<dependencies>")
                for p in dep_parts:
                    parts.append(f"  {p}")
                parts.append("</dependencies>")

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
