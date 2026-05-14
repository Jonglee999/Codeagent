"""SearchCodeTool — 在代码库中进行正则搜索和语义搜索。

支持两种模式：
- regex 模式：调用 ripgrep (rg) 子进程，解析 JSON 输出
- semantic 模式：调用 ContextEngine 的语义搜索接口
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from codeagent.tools.base import BaseTool, ToolResult


class SearchCodeTool(BaseTool):
    """在代码库中搜索代码，支持正则和语义两种模式。"""

    name = "search_code"
    description = "在代码库中搜索代码，支持正则和语义两种模式"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或自然语言查询",
            },
            "search_type": {
                "type": "string",
                "enum": ["regex", "semantic"],
                "default": "regex",
                "description": "搜索类型：regex（正则匹配）或 semantic（语义搜索）",
            },
            "file_pattern": {
                "type": "string",
                "description": "文件 glob 过滤，如 *.py",
            },
            "max_results": {
                "type": "integer",
                "default": 20,
                "description": "最大返回结果数，默认 20",
            },
            "context_lines": {
                "type": "integer",
                "default": 3,
                "description": "搜索结果上下文行数，默认 3",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        project_root: str | Path = ".",
        context_engine: Any = None,
    ) -> None:
        """初始化 SearchCodeTool。

        Args:
            project_root: 项目根目录路径。
            context_engine: ContextEngine 实例（semantic 模式需要）。
        """
        super().__init__()
        self._project_root = Path(project_root).resolve()
        self._context_engine = context_engine

    async def execute(  # type: ignore[override]
        self,
        query: str,
        search_type: str = "regex",
        file_pattern: str | None = None,
        max_results: int = 20,
        context_lines: int = 3,
    ) -> ToolResult:
        """执行代码搜索。

        Args:
            query: 搜索关键词或自然语言查询。
            search_type: "regex" 或 "semantic"。
            file_pattern: 文件 glob 过滤（如 "*.py"）。
            max_results: 最大返回结果数。
            context_lines: 搜索结果上下文行数。

        Returns:
            ToolResult: 搜索结果。
        """
        start_time = time.monotonic()

        if not query or not query.strip():
            return ToolResult(
                success=False,
                error_message="Query cannot be empty",
                error_code="EMPTY_QUERY",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        query = query.strip()

        if search_type == "regex":
            return await self._execute_regex(
                query, file_pattern, max_results, context_lines, start_time,
            )
        elif search_type == "semantic":
            return await self._execute_semantic(query, max_results, start_time)
        else:
            return ToolResult(
                success=False,
                error_message=f"Invalid search_type '{search_type}'. Use 'regex' or 'semantic'.",
                error_code="INVALID_SEARCH_TYPE",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

    async def _execute_regex(
        self,
        query: str,
        file_pattern: str | None,
        max_results: int,
        context_lines: int,
        start_time: float,
    ) -> ToolResult:
        """执行正则搜索（ripgrep）。"""
        cmd = ["rg", "--json", "-n"]

        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        if file_pattern:
            cmd.extend(["--glob", file_pattern])

        cmd.append(query)
        cmd.append(str(self._project_root))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error_message=(
                    "ripgrep (rg) not found. "
                    "Install it first: https://github.com/BurntSushi/ripgrep"
                ),
                error_code="RG_NOT_FOUND",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # returncode 1 = no matches (not an error)
        if proc.returncode not in (0, 1):
            error_text = stderr.decode("utf-8", errors="replace").strip()
            return ToolResult(
                success=False,
                error_message=f"ripgrep error: {error_text}",
                error_code="RG_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # Parse NDJSON output from ripgrep
        results = []
        seen: set[tuple[str, int]] = set()

        output = stdout.decode("utf-8", errors="replace")
        for line in output.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "match":
                continue

            data = obj.get("data", {})
            path = data.get("path", {}).get("text", "")

            # Normalize path to relative
            try:
                rel_path = str(Path(path).relative_to(self._project_root))
            except ValueError:
                rel_path = path
            rel_path = rel_path.replace("\\", "/")

            line_num = data.get("line_number", 0)
            line_content = data.get("lines", {}).get("text", "")

            # Extract column from first submatch
            column = 1
            submatches = data.get("submatches", [])
            if submatches:
                column = submatches[0].get("start", 0) + 1

            # Deduplicate by (file, line)
            dedup_key = (rel_path, line_num)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            results.append({
                "file_path": rel_path,
                "line": line_num,
                "column": column,
                "line_content": line_content.rstrip("\n").rstrip("\r"),
            })

            if len(results) >= max_results:
                break

        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_results": len(results),
                "search_type": "regex",
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    async def _execute_semantic(
        self,
        query: str,
        max_results: int,
        start_time: float,
    ) -> ToolResult:
        """执行语义搜索。"""
        if self._context_engine is None:
            return ToolResult(
                success=False,
                error_message="Semantic search requires ContextEngine to be configured",
                error_code="CONTEXT_ENGINE_NOT_CONFIGURED",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        try:
            snippets = await self._context_engine.search_semantic(
                query, top_k=max_results,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Semantic search error: {e}",
                error_code="SEMANTIC_SEARCH_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        results = []
        for snippet in snippets:
            first_line = snippet.code.split("\n")[0] if snippet.code else ""
            results.append({
                "file_path": snippet.file_path.replace("\\", "/"),
                "line": snippet.start_line,
                "end_line": snippet.end_line,
                "column": 1,
                "line_content": first_line,
                "code_snippet": snippet.code,
                "score": round(snippet.score, 4),
            })

        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_results": len(results),
                "search_type": "semantic",
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )
