"""上下文引擎性能基准测试。

测试 ContextEngine 在不同场景下的构建性能：
- 首次全量构建（cold start）
- 文件变更后的增量更新
"""

import time
from pathlib import Path

import pytest

from codeagent.context_engine.engine import ContextConfig, ContextEngine

pytestmark = pytest.mark.benchmark


class TestContextEnginePerformance:
    """上下文引擎性能基准测试。"""

    async def test_first_build_time(self, benchmark_project: Path, benchmark_metrics):
        """首次全量构建上下文，目标 < 30 秒。

        cold start 场景：无缓存、无索引，需要完整扫描项目文件树、
        代码分析、语义索引全量构建。
        """
        config = ContextConfig(
            use_mock_embeddings=True,
            cache_enabled=False,
        )
        engine = ContextEngine(config=config)

        start = time.perf_counter()
        package = await engine.build_context(
            project_root=str(benchmark_project),
            query="理解项目结构并分析主要功能模块",
        )
        elapsed = time.perf_counter() - start

        benchmark_metrics["context"]["first_build"] = elapsed

        # 验证构建结果非空
        assert package.file_tree, "File tree should not be empty after build"
        assert elapsed < 30.0, (
            f"首次上下文构建耗时 {elapsed:.2f}s，超过阈值 30s"
        )

    async def test_incremental_update(self, benchmark_project: Path, benchmark_metrics):
        """文件变更后增量更新上下文，目标 < 2 秒。

        先执行一次全量构建，然后模拟文件变更（修改 mtime），
        验证增量更新的性能。
        """
        config = ContextConfig(
            use_mock_embeddings=True,
            cache_enabled=False,
        )
        engine = ContextEngine(config=config)

        # 先执行全量构建建立索引
        await engine.build_context(
            project_root=str(benchmark_project),
            query="全量构建以建立基准索引",
        )

        # 模拟文件变更：touch 项目中的一个源文件
        test_file = benchmark_project / "main.py"
        assert test_file.exists(), f"Test file {test_file} should exist"
        test_file.touch()

        start = time.perf_counter()
        await engine.update_index(project_root=str(benchmark_project))
        elapsed = time.perf_counter() - start

        benchmark_metrics["context"]["incremental_update"] = elapsed

        assert elapsed < 2.0, (
            f"增量更新耗时 {elapsed:.2f}s，超过阈值 2s"
        )
