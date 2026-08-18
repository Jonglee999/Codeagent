"""Bounded semantic code navigation with indexed and lexical fallbacks."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from codeagent.gateway.tool_gateway import ToolResult
from codeagent.tools.base import BaseTool
from codeagent.tools.search.search_code import SearchCodeTool


class NavigateCodeTool(BaseTool):
    """Find symbol definitions or references without dumping whole files."""

    name = "navigate_code"
    category = "exploration"
    read_only = True
    latency_hint = "fast"
    description = (
        "按符号查找定义或引用；优先使用项目 AST 符号索引，并以有界的精确词搜索降级，"
        "适合 go-to-definition/find-references"
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "要定位的标识符，例如 ContextEngine 或 build_context",
            },
            "action": {
                "type": "string",
                "enum": ["definition", "references"],
                "default": "definition",
                "description": "查找定义或引用",
            },
            "file_pattern": {
                "type": "string",
                "description": "可选文件 glob，例如 *.py",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 30,
            },
        },
        "required": ["symbol"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        project_root: str | Path = ".",
        context_engine: Any | None = None,
    ) -> None:
        super().__init__()
        self._project_root = Path(project_root).resolve()
        self._context_engine = context_engine
        self._search = SearchCodeTool(project_root, context_engine=context_engine)

    async def execute(  # type: ignore[override]
        self,
        symbol: str,
        action: str = "definition",
        file_pattern: str | None = None,
        max_results: int = 30,
    ) -> ToolResult:
        started = time.monotonic()
        symbol = symbol.strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", symbol, flags=re.UNICODE):
            return ToolResult(
                success=False,
                error_message="symbol must be one identifier, not an expression",
                error_code="INVALID_SYMBOL",
                duration_ms=(time.monotonic() - started) * 1000,
            )
        max_results = max(1, min(int(max_results), 100))

        if action == "definition":
            indexed = self._indexed_definitions(symbol, file_pattern, max_results)
            if indexed:
                return ToolResult(
                    success=True,
                    data={
                        "symbol": symbol,
                        "action": action,
                        "results": indexed,
                        "total_results": len(indexed),
                        "engine": "ast-symbol-index",
                        "truncated": len(indexed) >= max_results,
                    },
                    duration_ms=(time.monotonic() - started) * 1000,
                )
            pattern = (
                rf"(?:class|def|async\s+def|function|interface|type|enum|"
                rf"const|let|var)\s+{re.escape(symbol)}\b"
            )
        else:
            pattern = rf"\b{re.escape(symbol)}\b"

        result = await self._search.execute(
            query=pattern,
            search_type="regex",
            file_pattern=file_pattern,
            max_results=max_results,
            context_lines=0,
        )
        if not result.success:
            return result
        data = dict(result.data or {})
        data.update({
            "symbol": symbol,
            "action": action,
            "engine": "ripgrep-word-fallback",
        })
        result.data = data
        result.duration_ms = (time.monotonic() - started) * 1000
        return result

    def _indexed_definitions(
        self, symbol: str, file_pattern: str | None, max_results: int,
    ) -> list[dict[str, Any]]:
        analyzer = getattr(self._context_engine, "code_analyzer", None)
        query = getattr(analyzer, "query_symbol", None)
        if not callable(query):
            return []
        results: list[dict[str, Any]] = []
        for item in query(symbol):
            raw_path = Path(str(item.file_path))
            try:
                path = (
                    raw_path.resolve().relative_to(self._project_root).as_posix()
                    if raw_path.is_absolute()
                    else raw_path.as_posix()
                )
            except (OSError, ValueError):
                continue
            if file_pattern and not Path(path).match(file_pattern):
                continue
            results.append({
                "file_path": path,
                "line": int(item.start_line),
                "end_line": int(item.end_line),
                "column": 1,
                "kind": str(item.kind),
                "signature": str(item.signature or "").splitlines()[0][:500],
            })
            if len(results) >= max_results:
                break
        return results
