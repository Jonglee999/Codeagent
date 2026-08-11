"""SearchCodeTool — 在代码库中进行正则搜索和语义搜索。

支持两种模式：
- regex 模式：调用 ripgrep (rg) 子进程，解析 JSON 输出
- semantic 模式：调用 ContextEngine 的语义搜索接口
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import time
from pathlib import Path
from typing import Any

from codeagent.tools.base import BaseTool, ToolResult


class SearchCodeTool(BaseTool):
    """在代码库中搜索代码，支持正则和语义两种模式。"""

    name = "search_code"
    category = "exploration"
    read_only = True
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
                "enum": ["literal", "regex", "semantic", "hybrid"],
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
            "case_sensitive": {"type": "boolean", "default": True},
            "include_globs": {"type": "array", "items": {"type": "string"}},
            "exclude_globs": {"type": "array", "items": {"type": "string"}},
            "paths": {"type": "array", "items": {"type": "string"}},
            "max_output_bytes": {"type": "integer", "default": 262144},
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
        case_sensitive: bool = True,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        paths: list[str] | None = None,
        max_output_bytes: int = 262144,
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
        max_results = max(1, min(int(max_results), 200))
        context_lines = max(0, min(int(context_lines), 20))
        max_output_bytes = max(4096, min(int(max_output_bytes), 2 * 1024 * 1024))
        if len(query) > 4096:
            return ToolResult(
                success=False,
                error_message="Search query exceeds the 4096 character limit",
                error_code="QUERY_TOO_LONG",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if search_type in {"literal", "regex"}:
            return await self._execute_regex(
                query, file_pattern, max_results, context_lines, start_time,
                literal=search_type == "literal",
                case_sensitive=case_sensitive,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                paths=paths,
                max_output_bytes=max_output_bytes,
            )
        elif search_type == "semantic":
            return await self._execute_semantic(query, max_results, start_time)
        elif search_type == "hybrid":
            return await self._execute_hybrid(
                query=query,
                file_pattern=file_pattern,
                max_results=max_results,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                paths=paths,
                max_output_bytes=max_output_bytes,
                start_time=start_time,
            )
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
        *,
        literal: bool = False,
        case_sensitive: bool = True,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        paths: list[str] | None = None,
        max_output_bytes: int = 262144,
    ) -> ToolResult:
        """执行正则搜索（ripgrep）。"""
        cmd = ["rg", "--json", "-n"]
        if literal:
            cmd.append("--fixed-strings")
        if not case_sensitive:
            cmd.append("--ignore-case")
        for glob in (
            "!.git/**", "!.codeagent/**", "!.venv/**", "!venv/**",
            "!node_modules/**", "!build/**", "!dist/**", "!.env*",
            "!*.pem", "!*.key", "!*.p12",
        ):
            cmd.extend(["--glob", glob])

        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        if file_pattern:
            normalized_glob = file_pattern.replace("\\", "/")
            if "/" in normalized_glob and not normalized_glob.startswith("**/"):
                # rg receives an absolute search root below. A path-qualified glob
                # such as ``src/pkg/*.py`` otherwise fails to match that absolute
                # path on Windows (and can do so on other platforms as well).
                normalized_glob = f"**/{normalized_glob.lstrip('/')}"
            cmd.extend(["--glob", normalized_glob])
        for glob in include_globs or []:
            cmd.extend(["--glob", self._normalize_glob(glob)])
        for glob in exclude_globs or []:
            cmd.extend(["--glob", "!" + self._normalize_glob(glob).lstrip("!")])

        cmd.append(query)
        search_roots = self._resolve_search_paths(paths)
        if isinstance(search_roots, ToolResult):
            return search_roots
        cmd.extend(str(path) for path in search_roots)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return self._execute_python_regex(
                query=query,
                file_pattern=file_pattern,
                max_results=max_results,
                start_time=start_time,
                literal=literal,
                case_sensitive=case_sensitive,
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

        output = stdout[:max_output_bytes].decode("utf-8", errors="replace")
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
                "engine": "ripgrep",
                "truncated": len(stdout) > max_output_bytes,
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    def _execute_python_regex(
        self,
        *,
        query: str,
        file_pattern: str | None,
        max_results: int,
        start_time: float,
        literal: bool = False,
        case_sensitive: bool = True,
    ) -> ToolResult:
        """Bounded stdlib fallback used when ripgrep is unavailable."""
        try:
            pattern = re.compile(
                re.escape(query) if literal else query,
                0 if case_sensitive else re.IGNORECASE,
            )
        except re.error as exc:
            return ToolResult(
                success=False,
                error_message=f"Invalid regular expression: {exc}",
                error_code="REGEX_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        excluded_dirs = {
            ".git", ".codeagent", ".mypy_cache", ".pytest_cache",
            ".ruff_cache", ".venv", "__pycache__", "build", "dist",
            "node_modules", "venv",
        }
        sensitive_names = {".env", ".token", "credentials", "secrets"}
        results: list[dict[str, Any]] = []
        scanned_files = 0
        max_files = 10_000
        max_file_bytes = 2 * 1024 * 1024
        result_limit = max(1, min(int(max_results), 200))

        for path in sorted(self._project_root.rglob("*")):
            try:
                relative = path.relative_to(self._project_root)
                if any(part in excluded_dirs for part in relative.parts[:-1]):
                    continue
                if not path.is_file() or path.is_symlink():
                    continue
                name_lower = path.name.lower()
                if (
                    name_lower in sensitive_names
                    or name_lower.startswith(".env.")
                    or name_lower.endswith((".key", ".pem", ".p12"))
                ):
                    continue
                relative_posix = relative.as_posix()
                if file_pattern and not (
                    fnmatch.fnmatch(relative_posix, file_pattern)
                    or fnmatch.fnmatch(path.name, file_pattern)
                ):
                    continue
                if path.stat().st_size > max_file_bytes:
                    continue
                scanned_files += 1
                if scanned_files > max_files:
                    break
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            for line_number, line_content in enumerate(content.splitlines(), start=1):
                match = pattern.search(line_content)
                if match is None:
                    continue
                results.append({
                    "file_path": relative_posix,
                    "line": line_number,
                    "column": match.start() + 1,
                    "line_content": line_content,
                })
                if len(results) >= result_limit:
                    break
            if len(results) >= result_limit:
                break

        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_results": len(results),
                "search_type": "regex",
                "engine": "python-fallback",
                "scanned_files": min(scanned_files, max_files),
                "warning": "ripgrep was unavailable; used bounded Python regex search",
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

    async def _execute_hybrid(self, **kwargs: Any) -> ToolResult:
        """Fuse lexical and semantic ranks without requiring a ready vector index."""
        start_time = float(kwargs.pop("start_time"))
        query = str(kwargs.pop("query"))
        max_results = int(kwargs.pop("max_results", 20))
        regex_result = await self._execute_regex(
            query,
            kwargs.pop("file_pattern", None),
            max_results,
            int(kwargs.pop("context_lines", 3)),
            start_time,
            literal=False,
            case_sensitive=bool(kwargs.pop("case_sensitive", True)),
            include_globs=kwargs.pop("include_globs", None),
            exclude_globs=kwargs.pop("exclude_globs", None),
            paths=kwargs.pop("paths", None),
            max_output_bytes=int(kwargs.pop("max_output_bytes", 262144)),
        )
        semantic_result = await self._execute_semantic(query, max_results, start_time)
        structural_result = await self._execute_structural(query, max_results, start_time)
        ranked: dict[tuple[str, int], dict[str, Any]] = {}
        scores: dict[tuple[str, int], float] = {}
        sources: dict[tuple[str, int], set[str]] = {}
        for source, result in (
            ("regex", regex_result),
            ("ast", structural_result),
            ("semantic", semantic_result),
        ):
            if not result.success or not result.data:
                continue
            for rank, item in enumerate(result.data.get("results", []), start=1):
                key = (str(item.get("file_path", "")), int(item.get("line", 0)))
                ranked.setdefault(key, dict(item))
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
                sources.setdefault(key, set()).add(source)
        ordered = sorted(ranked, key=lambda key: scores[key], reverse=True)[:max_results]
        results = []
        for key in ordered:
            item = ranked[key]
            item["score"] = round(scores[key], 6)
            item["sources"] = sorted(sources[key])
            results.append(item)
        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_results": len(results),
                "search_type": "hybrid",
                "semantic_available": semantic_result.success,
                "ast_available": structural_result.success,
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    async def _execute_structural(
        self, query: str, max_results: int, start_time: float,
    ) -> ToolResult:
        if self._context_engine is None or not hasattr(self._context_engine, "search_structural"):
            return ToolResult(success=False, error_code="CONTEXT_ENGINE_NOT_CONFIGURED")
        try:
            snippets = await self._context_engine.search_structural(query, top_k=max_results)
        except Exception as exc:
            return ToolResult(
                success=False,
                error_message=f"Structural search error: {exc}",
                error_code="STRUCTURAL_SEARCH_ERROR",
            )
        results = []
        for snippet in snippets:
            results.append({
                "file_path": snippet.file_path.replace("\\", "/"),
                "line": snippet.start_line,
                "end_line": snippet.end_line,
                "column": 1,
                "line_content": snippet.code.split("\n")[0] if snippet.code else "",
                "code_snippet": snippet.code,
                "score": round(snippet.score, 4),
            })
        return ToolResult(
            success=True,
            data={"results": results, "total_results": len(results), "search_type": "ast"},
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    @staticmethod
    def _normalize_glob(value: str) -> str:
        return value.replace("\\", "/").strip()

    def _resolve_search_paths(self, paths: list[str] | None) -> list[Path] | ToolResult:
        if not paths:
            return [self._project_root]
        resolved: list[Path] = []
        for value in paths[:20]:
            candidate = (self._project_root / value).resolve()
            try:
                candidate.relative_to(self._project_root)
            except ValueError:
                return ToolResult(
                    success=False,
                    error_message=f"Search path escapes project root: {value}",
                    error_code="PATH_OUTSIDE_PROJECT",
                )
            if candidate.exists() and not candidate.is_symlink():
                resolved.append(candidate)
        return resolved or [self._project_root]
