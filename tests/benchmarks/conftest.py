"""基准测试共享 fixtures 和阈值配置。"""

from collections import defaultdict
from pathlib import Path

import pytest

# =============================================================================
# 基准测试阈值配置（单位：秒）
#
# 所有阈值基于 CI ubuntu-latest 环境设定，本地开发环境可能更快或更慢。
# 若在本地运行时断言失败，先检查是否由于资源限制而非代码退化。
# =============================================================================
BENCHMARK_THRESHOLDS = {
    "context_build_first": 30.0,          # 首次全量构建（秒）
    "context_build_incremental": 2.0,     # 增量更新（秒）
    "semantic_search_p95": 0.5,           # 语义搜索 P95 延迟（秒）
    "tool_call_file_read": 0.1,           # 文件读取工具调用（秒）
    "tool_call_terminal": 5.0,            # 终端命令工具调用（秒）
    "llm_call_p95": 10.0,                 # LLM 调用 P95 延迟（秒）
    "websocket_latency": 0.2,             # WebSocket 事件延迟（秒）
    "end_to_end_task": 300.0,             # 端到端任务执行（秒，5 分钟）
}


@pytest.fixture(scope="session")
def benchmark_project() -> Path:
    """基准测试项目目录。

    使用 tests/fixtures/sample_python_project/ 作为固定大小的测试项目，
    确保基准测试在不同环境下可重复。
    """
    return Path(__file__).parent.parent / "fixtures" / "sample_python_project"


@pytest.fixture
def benchmark_metrics():
    """收集测试耗时，用于 CI 中比较和历史追踪。

    用法：
        def test_something(self, benchmark_metrics):
            start = time.perf_counter()
            ...  # 被测代码
            elapsed = time.perf_counter() - start
            benchmark_metrics["my_group"]["my_test"] = elapsed
    """
    results: dict = defaultdict(dict)
    yield results
    # 测试结束后输出 JSON 格式的结果
    import json
    print(f"\n[BENCHMARK] Results:\n{json.dumps(dict(results), indent=2)}")
