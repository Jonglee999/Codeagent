"""端到端任务执行性能基准测试。

测试整个 Agent 工作流从任务提交到完成的执行时间。
使用真实 LLM（从 .env 配置）和真实 Redis 服务。

基准测试场景：
- 简单代码分析任务端到端执行
"""

import os
import time
from pathlib import Path

import pytest

from codeagent.config import load_env_file

# 在模块加载时加载 .env 文件（确保 LLM_API_KEY 可用）
load_env_file(Path.cwd() / ".env")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.slow,
]

_HAS_API_KEY = bool(os.environ.get("LLM_API_KEY"))


def _build_llm():
    """构建 LLM 调用函数（使用 litellm）。"""
    import litellm
    from codeagent.config import get_env

    api_key = get_env("LLM_API_KEY", "")
    api_base = get_env("LLM_API_BASE", "")
    timeout = int(get_env("LLM_TIMEOUT", "60"))
    model = get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")

    litellm.set_verbose = False

    async def llm_call(**kwargs):
        last_exc = None
        for attempt in range(3):
            try:
                call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                call_kwargs["timeout"] = timeout
                if api_key:
                    call_kwargs["api_key"] = api_key
                if api_base:
                    call_kwargs["api_base"] = api_base
                return await litellm.acompletion(**call_kwargs)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc

    from codeagent.interaction.api.metrics import wrap_llm_call
    return wrap_llm_call(model, llm_call)


def _build_gateways(project_root: str):
    """构建工具和验证 Gateway。"""
    from codeagent.context_engine.engine import ContextConfig, ContextEngine
    from codeagent.gateway.validation_gateway_impl import ValidationGateway
    from codeagent.tools.gateway import ToolGateway

    # 上下文引擎 — 使用 mock 嵌入避免模型下载
    config = ContextConfig(
        use_mock_embeddings=True,
        cache_enabled=False,
    )
    context_gateway = ContextEngine(config=config)

    # 工具 Gateway
    tool_gateway = ToolGateway(project_root=project_root)

    # 验证 Gateway
    validation_gateway = ValidationGateway()

    return context_gateway, tool_gateway, validation_gateway


@pytest.mark.skipif(not _HAS_API_KEY, reason="需要 LLM_API_KEY 环境变量")
@pytest.mark.asyncio
class TestEndToEndPerformance:
    """端到端任务执行性能基准。"""

    async def test_simple_task_duration(self, benchmark_metrics, benchmark_project: Path):
        """简单代码分析任务的端到端执行，目标 < 5 分钟。

        使用真实 LLM 执行一个代码分析任务，记录完整耗时。
        """
        from codeagent.config import get_env

        context_gateway, tool_gateway, validation_gateway = _build_gateways(
            str(benchmark_project),
        )
        llm = _build_llm()
        model_name = get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")

        from codeagent.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
            model_name=model_name,
        )

        start = time.perf_counter()
        final_state = await orchestrator.run(
            user_request="分析 main.py 的结构，列出所有函数和类的定义",
            project_root=str(benchmark_project),
            auto_mode=True,
        )
        elapsed = time.perf_counter() - start

        benchmark_metrics["e2e"]["simple_task_duration"] = elapsed

        # 验证结果
        assert final_state is not None, "Orchestrator 应返回最终状态"
        assert elapsed < 300.0, (
            f"端到端任务耗时 {elapsed:.2f}s，超过阈值 300s"
        )

    async def test_task_with_repair(self, benchmark_metrics, benchmark_project: Path):
        """含验证修复的任务端到端执行。

        通过 ValidateNode 验证执行结果，触发修复流程。
        """
        from codeagent.config import get_env

        context_gateway, tool_gateway, validation_gateway = _build_gateways(
            str(benchmark_project),
        )
        llm = _build_llm()
        model_name = get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")

        from codeagent.orchestration.orchestrator import Orchestrator

        orchestrator = Orchestrator(
            context_gateway=context_gateway,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
            model_name=model_name,
        )

        start = time.perf_counter()
        final_state = await orchestrator.run(
            user_request="为 main.py 中每个函数添加类型注解",
            project_root=str(benchmark_project),
            auto_mode=True,
        )
        elapsed = time.perf_counter() - start

        benchmark_metrics["e2e"]["task_with_repair_duration"] = elapsed

        assert final_state is not None, "Orchestrator 应返回最终状态"
        assert elapsed < 300.0, (
            f"含修复任务耗时 {elapsed:.2f}s，超过阈值 300s"
        )
