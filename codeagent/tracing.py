"""OpenTelemetry 分布式追踪配置与装饰器。

提供 setup_tracing/get_tracer 初始化，以及 trace_node/trace_llm_call/trace_tool_call
三个装饰器工厂，用于在 LangGraph 节点、LLM 调用、工具调用中创建 Span。

环境变量：
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC 端点（未设置时降级为 NoOp）
- OTEL_SERVICE_NAME: 服务名称（默认 codeagent）
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

_tracer: trace.Tracer | None = None

F = TypeVar("F", bound=Callable[..., Any])


def setup_tracing(service_name: str = "codeagent") -> trace.Tracer:
    """初始化 OpenTelemetry TracerProvider。

    环境变量控制：
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP gRPC 端点（默认 None，=NoOp）
    - OTEL_SERVICE_NAME: 服务名称（默认 codeagent）
    """
    global _tracer

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    service = os.environ.get("OTEL_SERVICE_NAME", service_name)

    if not endpoint:
        _tracer = trace.get_tracer(service)
        return _tracer

    resource = Resource(attributes={SERVICE_NAME: service})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service)
    return _tracer


def get_tracer() -> trace.Tracer:
    """获取全局 tracer，未初始化时返回 NoOpTracer。"""
    global _tracer
    if _tracer is None:
        return trace.get_tracer("codeagent")
    return _tracer


def trace_node(node_name: str) -> Callable[[F], F]:
    """装饰器：为 LangGraph 节点创建 Span。

    Span name: node.{node_name}
    属性: node_name

    用法::

        @trace_node("context")
        async def __call__(self, state: AgentState) -> dict[str, Any]:
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tr = get_tracer()
            with tr.start_as_current_span(f"node.{node_name}") as span:
                span.set_attribute("node_name", node_name)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as exc:
                    span.set_attribute("success", False)
                    span.record_exception(exc)
                    raise
        return wrapper  # type: ignore[return-value]
    return decorator


def _make_traced_decorator(
    span_name: str,
    extra_attrs: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """内部工厂：创建一个创建固定 Span 的装饰器。"""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tr = get_tracer()
            with tr.start_as_current_span(span_name) as span:
                if extra_attrs:
                    for k, v in extra_attrs.items():
                        span.set_attribute(k, v)
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as exc:
                    span.set_attribute("success", False)
                    span.record_exception(exc)
                    raise
        return wrapper  # type: ignore[return-value]
    return decorator


def trace_llm_call(model: str) -> Callable[[F], F]:
    """装饰器工厂：为 LLM 调用创建 Span。

    Span name: llm.call
    属性: model, span.kind=client

    用法::

        traced_llm = trace_llm_call("deepseek/deepseek-v4-flash")(llm_callable)
    """
    return _make_traced_decorator(
        "llm.call",
        {"model": model, "span.kind": "client"},
    )


def trace_tool_call(tool_name: str) -> Callable[[F], F]:
    """装饰器工厂：为工具调用创建 Span。

    Span name: tool.call
    属性: tool_name, span.kind=client

    用法::

        traced_execute = trace_tool_call("write_file")(tool_instance.execute)
    """
    return _make_traced_decorator(
        "tool.call",
        {"tool_name": tool_name, "span.kind": "client"},
    )
