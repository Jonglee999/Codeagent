"""Phase 9.2: 分布式追踪单元测试。

覆盖 OpenTelemetry 配置、装饰器 Span 创建、属性绑定、异常保护。
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import codeagent.tracing as tracing_module


# ── 辅助 Fixture ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_tracer():
    """每个测试前重置全局 tracer 状态。"""
    tracing_module._tracer = None
    yield


@pytest.fixture
def span_exporter():
    """提供 InMemorySpanExporter 并绑定到 tracing_module._tracer。

    OTel SDK 1.42 不允许重复 set_tracer_provider，
    因此直接设置模块级的 _tracer 以绕过限制。
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    tracing_module._tracer = tracer
    yield exporter
    tracing_module._tracer = None


# ── setup_tracing / get_tracer ───────────────────────────────────


class TestSetupTracing:
    """setup_tracing 和 get_tracer 基础功能。"""

    def test_setup_tracing_no_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未配置 OTEL_EXPORTER_OTLP_ENDPOINT 时降级为 NoOp。"""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        tracer = tracing_module.setup_tracing("test-service")
        assert tracer is not None

    def test_setup_tracing_with_endpoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """配置了端点时创建 SDK TracerProvider。"""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-codeagent")
        tracer = tracing_module.setup_tracing("test-codeagent")
        assert tracer is not None

    def test_get_tracer_not_initialized(self) -> None:
        """get_tracer() 未初始化时返回 NoOpTracer。"""
        tracer = tracing_module.get_tracer()
        assert tracer is not None

    def test_get_tracer_after_setup(self, span_exporter: InMemorySpanExporter) -> None:
        """get_tracer() 在 setup 后返回配置的 tracer。"""
        tracer = tracing_module.get_tracer()
        assert tracer is not None


# ── trace_node 装饰器 ────────────────────────────────────────


class TestTraceNode:
    """trace_node 装饰器 Span 创建验证。"""

    @pytest.mark.asyncio
    async def test_trace_node_creates_span(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """@trace_node 创建一个名为 node.{name} 的 Span。"""
        @tracing_module.trace_node("test_node")
        async def sample_func(state: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"result": "ok"}

        result = await sample_func()
        assert result["result"] == "ok"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "node.test_node"

    @pytest.mark.asyncio
    async def test_trace_node_attributes(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """Span 包含 node_name 和 success 属性。"""
        @tracing_module.trace_node("planning")
        async def plan(state: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"plan": ["step1"]}

        await plan()
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("node_name") == "planning"
        assert spans[0].attributes.get("success") is True

    @pytest.mark.asyncio
    async def test_trace_node_records_exception(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """函数抛出异常时 span 记录 success=False。"""
        @tracing_module.trace_node("failing")
        async def failing_func(state: dict[str, Any] | None = None) -> dict[str, Any]:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await failing_func()

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("success") is False

    @pytest.mark.asyncio
    async def test_trace_node_preserves_return(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """装饰器不改变函数返回值。"""
        @tracing_module.trace_node("returner")
        async def returner(state: dict[str, Any] | None = None) -> str:
            return "hello"

        result = await returner()
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_trace_node_works_on_method(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """装饰器在类方法上正常工作。"""
        class MyNode:
            @tracing_module.trace_node("my_node")
            async def __call__(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
                return {"done": True}

        node = MyNode()
        result = await node()
        assert result["done"] is True

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "node.my_node"
        assert spans[0].attributes.get("node_name") == "my_node"


# ── trace_llm_call 装饰器 ────────────────────────────────────


class TestTraceLlmCall:
    """trace_llm_call 装饰器 Span 创建验证。"""

    @pytest.mark.asyncio
    async def test_trace_llm_call_creates_span(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """@trace_llm_call 创建一个名为 llm.call 的 Span。"""
        async def mock_llm(**kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "ok"}}]}

        traced = tracing_module.trace_llm_call("test-model")(mock_llm)
        result = await traced(model="test-model", messages=[])
        assert result["choices"][0]["message"]["content"] == "ok"

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "llm.call"

    @pytest.mark.asyncio
    async def test_trace_llm_call_attributes(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """Span 包含 model、span.kind、success 属性。"""
        async def mock_llm(**kwargs: Any) -> dict[str, Any]:
            return {"result": "ok"}

        traced = tracing_module.trace_llm_call("deepseek/test")(mock_llm)
        await traced()

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("model") == "deepseek/test"
        assert attrs.get("span.kind") == "client"
        assert attrs.get("success") is True

    @pytest.mark.asyncio
    async def test_trace_llm_call_exception(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """LLM 调用异常时记录 span 失败。"""
        async def failing_llm(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("API error")

        traced = tracing_module.trace_llm_call("test")(failing_llm)
        with pytest.raises(RuntimeError):
            await traced()

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("success") is False


# ── trace_tool_call 装饰器 ──────────────────────────────────


class TestTraceToolCall:
    """trace_tool_call 装饰器 Span 创建验证。"""

    @pytest.mark.asyncio
    async def test_trace_tool_call_creates_span(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """@trace_tool_call 创建一个名为 tool.call 的 Span。"""
        async def mock_tool(**kwargs: Any) -> dict[str, Any]:
            return {"success": True, "data": {}}

        traced = tracing_module.trace_tool_call("write_file")(mock_tool)
        result = await traced(file_path="test.txt", content="hello")
        assert result["success"] is True

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "tool.call"

    @pytest.mark.asyncio
    async def test_trace_tool_call_attributes(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """Span 包含 tool_name、span.kind、success 属性。"""
        async def mock_tool(**kwargs: Any) -> dict[str, Any]:
            return {"success": True, "data": {}}

        traced = tracing_module.trace_tool_call("read_file")(mock_tool)
        await traced(file_path="test.txt")

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs.get("tool_name") == "read_file"
        assert attrs.get("span.kind") == "client"
        assert attrs.get("success") is True


# ── 集成场景 ─────────────────────────────────────────────


class TestIntegration:
    """多装饰器组合和实际使用场景。"""

    @pytest.mark.asyncio
    async def test_chained_spans(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """多个装饰器嵌套调用时创建正确的 Span。"""
        @tracing_module.trace_node("execution")
        async def execution_node(state: dict[str, Any] | None = None) -> dict[str, Any]:
            return {"result": "executed"}

        await execution_node()
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "node.execution"

    @pytest.mark.asyncio
    async def test_decorator_passthrough_no_op(
        self, span_exporter: InMemorySpanExporter,
    ) -> None:
        """装饰器不影响原始函数签名和调用。"""
        @tracing_module.trace_node("passthrough")
        async def identity(x: int, y: str = "default") -> dict[str, Any]:
            return {"x": x, "y": y}

        result = await identity(42, y="hello")
        assert result == {"x": 42, "y": "hello"}

    def test_trace_node_is_decorator_factory(self) -> None:
        """trace_node 返回一个可调用的装饰器。"""
        decorator = tracing_module.trace_node("test")
        assert callable(decorator)

    def test_trace_llm_call_is_decorator_factory(self) -> None:
        """trace_llm_call 返回一个可调用的装饰器。"""
        decorator = tracing_module.trace_llm_call("test")
        assert callable(decorator)

    def test_trace_tool_call_is_decorator_factory(self) -> None:
        """trace_tool_call 返回一个可调用的装饰器。"""
        decorator = tracing_module.trace_tool_call("test")
        assert callable(decorator)
