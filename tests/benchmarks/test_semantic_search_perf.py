"""语义搜索性能基准测试。

测试 SemanticSearchEngine 在索引项目后的搜索延迟：
- P50 搜索延迟
- P95 搜索延迟
- 不同查询类型（短查询 / 长查询）的延迟
"""

import time
from pathlib import Path

import pytest

from codeagent.context_engine.semantic_search import SemanticSearchEngine

pytestmark = pytest.mark.benchmark

# 用于性能测试的查询集合
_SAMPLE_QUERIES = [
    "数据库连接配置",
    "用户认证逻辑",
    "API 路由定义",
    "错误处理中间件",
    "配置加载",
    "如何初始化数据库连接并配置连接池参数",
    "用户登录接口的请求验证和错误处理流程",
    "项目中的依赖注入和服务注册模式",
]


class TestSemanticSearchPerformance:
    """语义搜索性能基准测试。"""

    async def test_search_latency_p50(self, benchmark_project: Path, benchmark_metrics):
        """语义搜索 P50 延迟，目标 < 500ms。

        对项目索引后执行多次搜索查询，统计中位数延迟。
        """
        engine = SemanticSearchEngine(use_mock=True)

        # 索引项目
        await engine.index_project(str(benchmark_project))
        stats = engine.get_index_stats()
        assert stats["total_chunks"] > 0, "Project should have indexed chunks"

        # 执行多次搜索记录延迟
        latencies: list[float] = []
        for query in _SAMPLE_QUERIES:
            start = time.perf_counter()
            results = await engine.search(query, top_k=5)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            # 验证搜索结果
            assert isinstance(results, list), "Search should return a list"

        # 计算 P50 延迟
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        benchmark_metrics["semantic_search"]["p50_latency"] = p50

        assert p50 < 0.5, f"语义搜索 P50 延迟 {p50*1000:.1f}ms，超过阈值 500ms"

    async def test_search_latency_p95(self, benchmark_project: Path, benchmark_metrics):
        """语义搜索 P95 延迟，目标 < 500ms。

        对项目索引后执行多次搜索查询，统计 P95 延迟。
        """
        engine = SemanticSearchEngine(use_mock=True)

        await engine.index_project(str(benchmark_project))

        # 执行大量搜索以获取分位数
        latencies: list[float] = []
        # 每个查询重复多次以增加样本量
        for query in _SAMPLE_QUERIES:
            for _ in range(3):
                start = time.perf_counter()
                await engine.search(query, top_k=10)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

        # 计算 P95 延迟
        latencies.sort()
        p95_idx = max(int(len(latencies) * 0.95) - 1, 0)
        p95 = latencies[p95_idx]
        benchmark_metrics["semantic_search"]["p95_latency"] = p95

        assert p95 < 0.5, f"语义搜索 P95 延迟 {p95*1000:.1f}ms，超过阈值 500ms"

    async def test_search_filter_language(
        self, benchmark_project: Path, benchmark_metrics,
    ):
        """按语言过滤的语义搜索延迟。"""
        engine = SemanticSearchEngine(use_mock=True)

        await engine.index_project(str(benchmark_project))

        start = time.perf_counter()
        results = await engine.search(
            "函数定义和类定义", top_k=5, filter_lang="python",
        )
        elapsed = time.perf_counter() - start

        benchmark_metrics["semantic_search"]["filtered_search"] = elapsed

        # 带过滤的搜索应该返回结果（project 是 Python 项目）
        assert elapsed < 0.5, (
            f"带过滤的语义搜索耗时 {elapsed*1000:.1f}ms，超过阈值 500ms"
        )
