"""CodeChunker — AST 感知的代码分块器，在函数/类边界处断开，保持语义完整性。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

from codeagent.context_engine.symbol_table import _detect_language, _init_parser

# ── Token 计数 ─────────────────────────────────────────────────────────────────

_ENCODER_CACHE: dict[str, Any] = {}


def _get_encoder(model: str = "cl100k_base") -> Any:
    """获取 tiktoken 编码器（缓存）。"""
    if model not in _ENCODER_CACHE:
        _ENCODER_CACHE[model] = tiktoken.get_encoding(model)
    return _ENCODER_CACHE[model]


def _count_tokens(text: str) -> int:
    """使用 tiktoken 计算 token 数。"""
    encoder = _get_encoder()
    return len(encoder.encode(text))


# ── CodeChunk 数据类 ───────────────────────────────────────────────────────────


@dataclass
class CodeChunk:
    """语义完整的代码块。"""

    file_path: str
    start_line: int
    end_line: int
    code: str
    symbol_name: str | None = None
    language: str | None = None
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        d: dict[str, Any] = {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
        }
        if self.symbol_name is not None:
            d["symbol_name"] = self.symbol_name
        if self.language is not None:
            d["language"] = self.language
        d["token_count"] = self.token_count
        return d


# ── AST 感知分块逻辑 ──────────────────────────────────────────────────────────


def _extract_ast_blocks(code: str, language: str) -> list[dict[str, Any]]:
    """使用 tree-sitter 解析代码，提取函数/类/方法定义块。

    返回按行排序的块列表，每个块包含 {name, kind, start_line, end_line}。
    """
    from tree_sitter import Node

    if language == "python":
        from tree_sitter_python import language as _lang

        lang_obj = _lang()
    elif language == "javascript":
        from tree_sitter_javascript import language as _lang

        lang_obj = _lang()
    elif language == "typescript":
        from tree_sitter_typescript import language_typescript as _lang

        lang_obj = _lang()
    elif language == "tsx":
        from tree_sitter_typescript import language_tsx as _lang

        lang_obj = _lang()
    else:
        raise ValueError(f"Unsupported language: {language}")

    from tree_sitter import Language, Parser

    parser = Parser()
    parser.language = Language(lang_obj)
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    blocks: list[dict[str, Any]] = []
    _walk_blocks(root, code, blocks, language)
    return blocks


def _walk_blocks(
    node: Any, code: str, blocks: list[dict[str, Any]], language: str
) -> None:
    """递归遍历 AST，收集函数/类/方法定义。"""
    chunk_types = {
        "python": {
            "function_definition",
            "class_definition",
            "async_function_definition",
        },
        "javascript": {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
        },
        "typescript": {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
            "interface_declaration",
            "type_alias_declaration",
        },
        "tsx": {
            "function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
            "interface_declaration",
            "type_alias_declaration",
        },
    }

    targets = chunk_types.get(language, set())

    if node.type in targets:
        name_node = node.child_by_field_name("name")
        name = (
            name_node.text.decode("utf-8", errors="replace")
            if name_node
            else ""
        )
        blocks.append({
            "name": name,
            "kind": node.type,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        })

        # 如果是类定义，也提取其中的方法
        if node.type == "class_definition":
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _walk_blocks(child, code, blocks, language)
            return

    # 继续遍历子节点
    for child in node.children:
        _walk_blocks(child, code, blocks, language)


def _merge_orphan_code(
    lines: list[str],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将未被 AST 块覆盖的代码（模块级代码、import 等）合并为块。"""
    covered_lines: set[int] = set()
    for b in blocks:
        for ln in range(b["start_line"], b["end_line"] + 1):
            covered_lines.add(ln)

    total_lines = len(lines)
    orphan_blocks: list[dict[str, Any]] = []
    orphan_start: int | None = None

    for ln in range(1, total_lines + 1):
        if ln not in covered_lines:
            if orphan_start is None:
                orphan_start = ln
        else:
            if orphan_start is not None:
                orphan_blocks.append({
                    "name": "",
                    "kind": "module_level",
                    "start_line": orphan_start,
                    "end_line": ln - 1,
                })
                orphan_start = None

    if orphan_start is not None:
        orphan_blocks.append({
            "name": "",
            "kind": "module_level",
            "start_line": orphan_start,
            "end_line": total_lines,
        })

    return orphan_blocks


def _split_large_block(
    lines: list[str],
    block: dict[str, Any],
    max_chunk_size: int,
    file_path: str,
    language: str,
) -> list[CodeChunk]:
    """将超过 max_chunk_size 的大块按逻辑边界拆分。"""
    block_lines = lines[block["start_line"] - 1: block["end_line"]]
    block_text = "".join(block_lines)
    tokens = _count_tokens(block_text)

    if tokens <= max_chunk_size:
        chunk = _make_chunk(lines, block, file_path, language)
        return [chunk]

    # 超过限制时按行数平均拆分
    chunks: list[CodeChunk] = []
    num_lines = block["end_line"] - block["start_line"] + 1
    target_lines_per_chunk = max(
        1, int(num_lines * max_chunk_size / max(tokens, 1))
    )

    for start in range(block["start_line"] - 1, block["end_line"], target_lines_per_chunk):
        end = min(start + target_lines_per_chunk, block["end_line"])
        chunk_lines = lines[start:end]
        chunk_text = "".join(chunk_lines)
        token_count = _count_tokens(chunk_text)

        chunks.append(CodeChunk(
            file_path=file_path,
            start_line=start + 1,
            end_line=end,
            code=chunk_text,
            symbol_name=block.get("name"),
            language=language,
            token_count=token_count,
        ))

    return chunks


def _make_chunk(
    lines: list[str],
    block: dict[str, Any],
    file_path: str,
    language: str,
) -> CodeChunk:
    """从块定义创建 CodeChunk。"""
    chunk_lines = lines[block["start_line"] - 1: block["end_line"]]
    chunk_text = "".join(chunk_lines)
    token_count = _count_tokens(chunk_text)

    return CodeChunk(
        file_path=file_path,
        start_line=block["start_line"],
        end_line=block["end_line"],
        code=chunk_text,
        symbol_name=block.get("name"),
        language=language,
        token_count=token_count,
    )


# ── CodeChunker 主类 ──────────────────────────────────────────────────────────


class CodeChunker:
    """AST 感知的代码分块器——在函数/类边界处断开，保持语义完整性。"""

    def chunk_file(self, file_path: str, max_chunk_size: int = 512) -> list[CodeChunk]:
        """将文件分割为语义完整的代码块。

        Args:
            file_path: 文件路径。
            max_chunk_size: 每个块的最大 token 数，默认 512。

        Returns:
            语义完整的代码块列表。
        """
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        language = _detect_language(file_path)
        if language is None:
            return []

        try:
            code = fp.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as e:
            raise OSError(f"Cannot read file {file_path}: {e}") from e

        return self.chunk_code(code, language, max_chunk_size, file_path)

    def chunk_code(
        self,
        code: str,
        language: str,
        max_chunk_size: int = 512,
        file_path: str = "",
    ) -> list[CodeChunk]:
        """直接对代码字符串分块。

        Args:
            code: 源代码字符串。
            language: 编程语言（python, javascript, typescript, tsx）。
            max_chunk_size: 每个块的最大 token 数，默认 512。
            file_path: 可选的文件路径，用于元数据。

        Returns:
            语义完整的代码块列表。
        """
        if not code.strip():
            return []

        lines = code.splitlines(keepends=True)
        chunks: list[CodeChunk] = []

        try:
            ast_blocks = _extract_ast_blocks(code, language)
        except (ImportError, ValueError, Exception):
            # AST 解析失败时退化为按行数分块
            return self._fallback_chunk(code, language, max_chunk_size, file_path, lines)

        # 合并孤儿代码（未覆盖的模块级代码）
        orphan_blocks = _merge_orphan_code(lines, ast_blocks)

        # 合并所有块并按行排序
        all_blocks = sorted(
            ast_blocks + orphan_blocks,
            key=lambda b: b["start_line"],
        )

        for block in all_blocks:
            split_chunks = _split_large_block(
                lines, block, max_chunk_size, file_path, language
            )
            chunks.extend(split_chunks)

        return chunks

    def _fallback_chunk(
        self,
        code: str,
        language: str,
        max_chunk_size: int,
        file_path: str,
        lines: list[str],
    ) -> list[CodeChunk]:
        """AST 解析失败时的降级分块策略：按空行分隔的逻辑块分块。"""
        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_start = 1
        total_lines = len(lines)

        for i, line in enumerate(lines):
            current_lines.append(line)
            next_line = lines[i + 1] if i + 1 < total_lines else ""

            # 遇到连续空行时分割
            if not line.strip() and not next_line.strip() and len(current_lines) > 1:
                chunk_text = "".join(current_lines)
                token_count = _count_tokens(chunk_text)

                # 如果块太大，按行数拆分
                if token_count > max_chunk_size:
                    sub_chunks = self._split_by_lines(
                        current_lines, current_start, max_chunk_size,
                        file_path, language,
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i + 1,
                        code=chunk_text,
                        language=language,
                        token_count=token_count,
                    ))

                current_lines = []
                current_start = i + 2

        # 处理剩余行
        if current_lines:
            chunk_text = "".join(current_lines)
            token_count = _count_tokens(chunk_text)
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=current_start,
                end_line=total_lines,
                code=chunk_text,
                language=language,
                token_count=token_count,
            ))

        return chunks

    def _split_by_lines(
        self,
        lines: list[str],
        start_line: int,
        max_chunk_size: int,
        file_path: str,
        language: str,
    ) -> list[CodeChunk]:
        """按行数简单拆分。"""
        chunks: list[CodeChunk] = []
        chunk_size = max(1, len(lines) // max(1, (sum(
            _count_tokens(l) for l in lines
        ) // max_chunk_size)))

        for i in range(0, len(lines), chunk_size):
            sub_lines = lines[i:i + chunk_size]
            sub_text = "".join(sub_lines)
            chunks.append(CodeChunk(
                file_path=file_path,
                start_line=start_line + i,
                end_line=start_line + i + len(sub_lines) - 1,
                code=sub_text,
                language=language,
                token_count=_count_tokens(sub_text),
            ))

        return chunks
