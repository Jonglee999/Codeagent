"""ContextEngine — 上下文引擎统一外观，实现 IContextGateway 接口。

整合 FileTreeIndexer、CodeAnalyzer、SemanticSearchEngine、ContextAssembler
为统一入口，支持全量构建和基于 mtime 缓存的增量构建。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeagent.context_engine.code_analyzer import CodeAnalyzer
from codeagent.context_engine.context_assembler import ContextAssembler
from codeagent.context_engine.file_tree_indexer import FileTreeIndexer
from codeagent.context_engine.semantic_search import SemanticSearchEngine
from codeagent.gateway.context_gateway import (
    CodeSnippet,
    ContextPackage,
    IContextGateway,
)

logger = logging.getLogger(__name__)

_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"

# 监控的源文件扩展名
_SOURCE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}

# 缓存中排除的目录
_EXCLUDE_CACHE_DIRS = {".git", "node_modules", "__pycache__", ".codeagent", ".claude", ".venv", "venv"}


@dataclass
class ContextConfig:
    """上下文引擎配置。

    Attributes:
        total_budget: ContextAssembler 总 token 预算（默认 8000）
        cache_enabled: 是否启用基于 mtime 的索引缓存（默认 True）
        cache_dir: 缓存目录（相对项目根目录，默认 ".codeagent/cache"）
        languages: 支持的语言列表
        search_top_k: 语义搜索返回结果数上限（默认 10）
        use_mock_embeddings: 是否使用 mock 嵌入模型（测试用，默认 False）
    """

    total_budget: int = 8000
    cache_enabled: bool = True
    cache_dir: str = ".codeagent/cache"
    languages: tuple[str, ...] = ("python", "typescript", "javascript")
    search_top_k: int = 10
    use_mock_embeddings: bool = False


@dataclass
class ContextStats:
    """上下文引擎性能统计。

    Attributes:
        total_invocations: build_context 总调用次数
        avg_build_time_ms: 平均构建耗时（毫秒）
        last_build_time_ms: 最近一次构建耗时（毫秒）
        cache_hits: 缓存命中次数（quick_build 且无文件变化）
        cache_misses: 缓存未命中次数（full_build）
        last_query: 最近一次查询文本
        last_build_mode: 最近一次构建模式（"full" / "quick"）
    """

    total_invocations: int = 0
    avg_build_time_ms: float = 0.0
    last_build_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_query: str = ""
    last_build_mode: str = ""


class ContextEngine(IContextGateway):
    """上下文引擎统一外观——实现 IContextGateway 接口。

    整合 FileTreeIndexer、CodeAnalyzer、SemanticSearchEngine、
    ContextAssembler 为统一入口，支持全量构建和增量缓存构建。
    """

    def __init__(self, config: ContextConfig | None = None) -> None:
        """初始化 ContextEngine。

        Args:
            config: 引擎配置，省略则使用默认配置。
        """
        self.config = config or ContextConfig()
        self.file_tree_indexer = FileTreeIndexer()
        self.code_analyzer = CodeAnalyzer(list(self.config.languages))
        self.semantic_search = SemanticSearchEngine(
            use_mock=self.config.use_mock_embeddings,
        )
        self.context_assembler = ContextAssembler(
            total_budget=self.config.total_budget,
        )
        self._stats = ContextStats()
        self._project_root: str | None = None

    # ── IContextGateway 接口实现 ────────────────────────────────────────────

    async def build_context(
        self, project_root: str, query: str,
    ) -> ContextPackage:
        """构建完整上下文数据包。

        按顺序执行：文件树扫描 → 代码分析 → 语义搜索 → 数据组装。
        根据缓存状态自动选择 full_build 或 quick_build 模式。

        Args:
            project_root: 项目根目录路径
            query: 用户的查询/需求描述

        Returns:
            ContextPackage: 上下文数据包
        """
        start = time.monotonic()
        self._project_root = str(Path(project_root).resolve())

        # ── 确定构建模式 ──────────────────────────────────
        mode = self._determine_build_mode()
        changed_files: list[str] = []

        if mode == "quick" and self.config.cache_enabled:
            changed_files = self._scan_changed_files()
            if not changed_files:
                self._stats.cache_hits += 1
            logger.info(
                "Quick build for %s (%d changed files)",
                self._project_root, len(changed_files),
            )
        else:
            self._stats.cache_misses += 1
            logger.info("Full build for %s", self._project_root)

        # ── 1. 文件树扫描 ─────────────────────────────────
        file_tree = await self.file_tree_indexer.scan(self._project_root)

        # ── 2. 代码分析 ───────────────────────────────────
        if mode == "full" or not self._cache_exists():
            await self.code_analyzer.build(self._project_root)
        else:
            for f in changed_files:
                await self.code_analyzer.update_file(f)

        # ── 3. 语义搜索 ───────────────────────────────────
        if mode == "full" or not self._cache_exists():
            await self.semantic_search.index_project(self._project_root)
        else:
            for f in changed_files:
                await self.semantic_search.reindex_file(f)

        search_results = await self.semantic_search.search(
            query, top_k=self.config.search_top_k,
        )

        # ── 4. 组装数据包 ─────────────────────────────────
        related_code = [
            CodeSnippet(
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                code=r.code_snippet,
                score=r.score,
            )
            for r in search_results
        ]

        symbol_table = self.code_analyzer.symbol_table.to_json()
        dep_info = self._build_dep_info()

        package = ContextPackage(
            file_tree=file_tree,
            related_code=related_code,
            dependency_info=dep_info,
            symbol_table=symbol_table,
        )

        # ── 5. 保存缓存快照 ───────────────────────────────
        if self.config.cache_enabled:
            self._save_cache()

        # ── 记录性能指标 ───────────────────────────────────
        elapsed = time.monotonic() - start
        self._record_stats(elapsed, mode, query)

        return package

    async def update_index(self, project_root: str) -> None:
        """全量更新所有索引。

        Args:
            project_root: 项目根目录路径
        """
        self._project_root = str(Path(project_root).resolve())
        await self.code_analyzer.build(self._project_root)
        await self.semantic_search.index_project(self._project_root)
        if self.config.cache_enabled:
            self._save_cache()

    async def search_semantic(
        self, query: str, top_k: int = 5,
    ) -> list[CodeSnippet]:
        """语义搜索相关代码片段（委托给 SemanticSearchEngine）。

        Args:
            query: 搜索查询
            top_k: 返回结果数量上限

        Returns:
            list[CodeSnippet]: 相关代码片段列表（按相关性降序）
        """
        results = await self.semantic_search.search(query, top_k=top_k)
        return [
            CodeSnippet(
                file_path=r.file_path,
                start_line=r.start_line,
                end_line=r.end_line,
                code=r.code_snippet,
                score=r.score,
            )
            for r in results
        ]

    # ── 查询接口 ───────────────────────────────────────────────────────────

    def get_stats(self) -> ContextStats:
        """返回引擎性能统计。"""
        return self._stats

    def get_cache_report(self) -> dict[str, Any]:
        """返回缓存状态报告。"""
        cache_path = self._get_cache_path()
        report: dict[str, Any] = {
            "cache_enabled": self.config.cache_enabled,
            "cache_dir": self.config.cache_dir,
            "cache_exists": cache_path is not None and cache_path.exists(),
            "stats": {
                "total_invocations": self._stats.total_invocations,
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "avg_build_time_ms": round(self._stats.avg_build_time_ms, 1),
                "last_build_mode": self._stats.last_build_mode,
            },
        }
        return report

    def assemble_context(self, package: ContextPackage) -> str:
        """将 ContextPackage 组装为 LLM-ready XML 格式。

        Args:
            package: 上下文数据包

        Returns:
            str: XML 格式的上下文文本
        """
        return self.context_assembler.assemble(package)

    def get_budget_report(self) -> Any:
        """获取最近一次组装的预算使用报告。"""
        return self.context_assembler.get_budget_report()

    # ── 构建模式判断 ───────────────────────────────────────────────────────

    def _determine_build_mode(self) -> str:
        """确定构建模式。

        Returns:
            "full" 或 "quick"。
            首次构建或无缓存时返回 full，否则返回 quick。
        """
        if not self.config.cache_enabled:
            return "full"
        if not self._cache_exists():
            return "full"
        return "quick"

    def _cache_exists(self) -> bool:
        """检查 mtime 缓存文件是否存在。"""
        cache_path = self._get_cache_path()
        return cache_path is not None and cache_path.is_file()

    # ── 变更检测 ───────────────────────────────────────────────────────────

    def _scan_changed_files(self) -> list[str]:
        """扫描自上次缓存快照后发生变更的文件。

        读取 mtime 快照并与当前文件系统对比，返回变更文件的绝对路径列表。
        包括：修改的文件、新增的文件。删除的文件不返回（已由缓存处理）。
        """
        cache_path = self._get_cache_path()
        if not cache_path or not cache_path.is_file():
            return []

        try:
            with open(cache_path, encoding="utf-8") as f:
                cache_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

        cached_mtimes: dict[str, str] = cache_data.get("files", {})
        root = Path(self._project_root) if self._project_root else None
        if not root:
            return []

        changed: list[str] = []

        # 遍历当前文件系统，与缓存对比
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SOURCE_EXTENSIONS:
                continue
            if any(d in file_path.parts for d in _EXCLUDE_CACHE_DIRS):
                continue

            rel = str(file_path.relative_to(root)).replace("\\", "/")
            try:
                current_mtime = _format_timestamp(file_path.stat().st_mtime)
            except OSError:
                continue

            cached_mtime = cached_mtimes.get(rel)

            if cached_mtime is None or cached_mtime != current_mtime:
                changed.append(str(file_path))

        return changed

    # ── 缓存读写 ───────────────────────────────────────────────────────────

    def _get_cache_dir(self) -> Path | None:
        """获取缓存目录路径。

        cache_dir 支持绝对路径和相对路径两种格式。
        相对路径时相对于项目根目录。
        """
        cache_path = Path(self.config.cache_dir)
        if cache_path.is_absolute():
            return cache_path
        if not self._project_root:
            return None
        return Path(self._project_root) / self.config.cache_dir

    def _get_cache_path(self) -> Path | None:
        """获取 mtime 缓存文件路径。"""
        cache_dir = self._get_cache_dir()
        if cache_dir is None:
            return None
        return cache_dir / "mtime_cache.json"

    def _save_cache(self) -> None:
        """将当前文件 mtime 快照写入缓存文件。"""
        cache_path = self._get_cache_path()
        if cache_path is None:
            return

        root = Path(self._project_root) if self._project_root else None
        if not root:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        files: dict[str, str] = {}
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SOURCE_EXTENSIONS:
                continue
            if any(d in file_path.parts for d in _EXCLUDE_CACHE_DIRS):
                continue
            rel = str(file_path.relative_to(root)).replace("\\", "/")
            try:
                files[rel] = _format_timestamp(file_path.stat().st_mtime)
            except OSError:
                continue

        cache_data = {
            "project_root": str(root),
            "files": files,
            "last_build": _format_timestamp(time.time()),
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except OSError:
            logger.warning("Failed to write cache to %s", cache_path)

    # ── 依赖信息 ───────────────────────────────────────────────────────────

    def _build_dep_info(self) -> dict[str, Any]:
        """构建全量依赖信息摘要（供 ContextAssembler 使用）。"""
        dep_graph = self.code_analyzer.dependency_graph
        if not self._project_root:
            return {}

        deps = dep_graph.to_dict()
        circles = dep_graph.find_circular_dependencies()

        result: dict[str, Any] = {
            "dependencies": deps,
            "total_modules": dep_graph.graph.number_of_nodes(),
        }
        if circles:
            result["circular_dependencies"] = circles

        return result

    # ── 统计记录 ───────────────────────────────────────────────────────────

    def _record_stats(self, elapsed: float, mode: str, query: str) -> None:
        """记录一次构建的性能统计。"""
        elapsed_ms = elapsed * 1000
        total = self._stats.total_invocations

        self._stats.last_build_time_ms = elapsed_ms
        self._stats.avg_build_time_ms = (
            (self._stats.avg_build_time_ms * total + elapsed_ms) / (total + 1)
            if total > 0
            else elapsed_ms
        )
        self._stats.total_invocations += 1
        self._stats.last_query = query
        self._stats.last_build_mode = mode


def _format_timestamp(timestamp: float) -> str:
    """将 Unix 时间戳格式化为 ISO 字符串（UTC）。"""
    try:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt.strftime(_ISO_FORMAT)
    except (OSError, ValueError, OverflowError):
        return ""
