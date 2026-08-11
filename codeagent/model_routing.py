"""Three-tier model registry and task-aware routing.

Routing tiers are internal runtime roles and are deliberately independent from
SWE-bench selection labels that happen to use the same L1/L2/L3 spelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Mapping

from codeagent import config
from codeagent.model_gateway import ModelEndpoint, ModelGateway, ProviderCall

ModelTier = Literal["L1", "L2", "L3"]

_HIGH_RISK = re.compile(
    r"认证|授权|安全|密钥|支付|迁移|并发|锁|事务|权限|"
    r"\b(auth|security|secret|payment|migration|concurren|transaction|permission)\b",
    re.I,
)
_CROSS_MODULE = re.compile(
    r"跨模块|跨服务|架构|重构|调用链|多文件|"
    r"\b(cross[- ]module|architecture|refactor|call chain|multiple files)\b",
    re.I,
)


@dataclass(frozen=True)
class ModelCapability:
    tier: ModelTier
    model: str
    context_window: int = 128_000
    max_output_tokens: int = 32_768
    supports_tools: bool = True
    supports_reasoning: bool = False
    api_key: str = field(default="", repr=False)
    api_base: str = ""
    timeout_seconds: float = 300.0

    def public_metadata(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "model": self.model,
            "configured": bool(self.model and self.api_key),
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "supports_tools": self.supports_tools,
            "supports_reasoning": self.supports_reasoning,
        }


class ModelRegistry:
    def __init__(self, capabilities: Mapping[ModelTier, ModelCapability]) -> None:
        missing = {"L1", "L2", "L3"} - set(capabilities)
        if missing:
            raise ValueError(f"Missing model tiers: {', '.join(sorted(missing))}")
        self._capabilities = dict(capabilities)

    @classmethod
    def from_env(cls) -> "ModelRegistry":
        shared_key = config.get_env("LLM_API_KEY", "")
        shared_base = config.get_env("LLM_API_BASE", "")
        shared_timeout = float(config.get_env("LLM_TIMEOUT", "300"))
        flash = config.get_env("LLM_MODEL", "deepseek/deepseek-v4-flash")
        defaults = {"L1": flash, "L2": flash, "L3": "deepseek/deepseek-v4-pro"}
        capabilities: dict[ModelTier, ModelCapability] = {}
        for tier in ("L1", "L2", "L3"):  # type: ModelTier
            name = config.get_env(f"MODEL_{tier}_NAME", defaults[tier])
            capabilities[tier] = ModelCapability(
                tier=tier,
                model=name,
                api_key=config.get_env(f"MODEL_{tier}_API_KEY", "") or shared_key,
                api_base=config.get_env(f"MODEL_{tier}_API_BASE", "") or shared_base,
                context_window=max(
                    1,
                    int(config.get_env(f"MODEL_{tier}_CONTEXT_WINDOW", "128000")),
                ),
                max_output_tokens=max(
                    1,
                    int(config.get_env(
                        f"MODEL_{tier}_MAX_OUTPUT_TOKENS",
                        config.get_env("MAX_COMPLETION_TOKENS_PER_CALL", "32768"),
                    )),
                ),
                supports_reasoning=tier == "L3",
                timeout_seconds=float(
                    config.get_env(f"MODEL_{tier}_TIMEOUT", str(shared_timeout))
                ),
            )
        return cls(capabilities)

    def get(self, tier: ModelTier) -> ModelCapability:
        return self._capabilities[tier]

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            self.get("L1").public_metadata(),
            self.get("L2").public_metadata(),
            self.get("L3").public_metadata(),
        ]


class ModelRouter:
    """Callable model client that selects a tier and aggregates usage."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        provider_call: ProviderCall | None = None,
        event_callback: Any | None = None,
    ) -> None:
        self.registry = registry
        self._event_callback = event_callback
        self._gateways: dict[ModelTier, ModelGateway] = {}
        for tier in ("L1", "L2", "L3"):
            capability = registry.get(tier)
            endpoint = ModelEndpoint(
                role="primary",
                model=capability.model,
                api_key=capability.api_key,
                api_base=capability.api_base,
                timeout_seconds=capability.timeout_seconds,
            )
            self._gateways[tier] = ModelGateway(
                endpoint,
                provider_call=provider_call,
                event_callback=event_callback,
            )
        self._selection_count = {"L1": 0, "L2": 0, "L3": 0}
        self._escalation_count = 0
        self._last_selection: dict[str, Any] | None = None
        self._shadow_count = 0
        self.primary = self._gateways["L2"].primary

    @classmethod
    def from_env(cls, **kwargs: Any) -> "ModelRouter":
        return cls(ModelRegistry.from_env(), **kwargs)

    def select(self, role: str, task_state: Mapping[str, Any] | None = None) -> ModelTier:
        state = dict(task_state or {})
        if config.get_model_routing_mode() == "off":
            return "L2"
        requested = str(state.get("tier", "")).upper()
        if requested in {"L1", "L2", "L3"}:
            return requested  # type: ignore[return-value]
        if role in {"classification", "search", "summary", "memory", "reflection", "chat"}:
            return "L1"
        text = str(state.get("request", ""))
        failures = int(state.get("validation_failures", 0) or 0)
        retries = int(state.get("retry_count", 0) or 0)
        files = int(state.get("affected_file_count", 0) or 0)
        high_risk = bool(state.get("high_risk")) or bool(_HIGH_RISK.search(text))
        cross_module = bool(state.get("cross_module")) or bool(_CROSS_MODULE.search(text))
        if high_risk or cross_module or files >= 3 or failures >= 2 or retries >= 2:
            return "L3"
        return "L2"

    async def __call__(self, **kwargs: Any) -> Any:
        role = str(kwargs.pop("model_role", "execution"))
        task_state = kwargs.pop("routing_context", None)
        proposed_tier = self.select(role, task_state)
        mode = config.get_model_routing_mode()
        tier: ModelTier = "L2" if mode == "shadow" else proposed_tier
        if mode == "shadow" and proposed_tier != tier:
            self._shadow_count += 1
        capability = self.registry.get(tier)
        kwargs.pop("model", None)
        configured_max = kwargs.get("max_tokens")
        if configured_max is not None:
            kwargs["max_tokens"] = min(int(configured_max), capability.max_output_tokens)
        self._selection_count[tier] += 1
        previous = self._last_selection["tier"] if self._last_selection else None
        if previous is not None and tier > previous:
            self._escalation_count += 1
        self._last_selection = {
            "tier": tier,
            "proposed_tier": proposed_tier,
            "role": role,
            "model": capability.model,
            "mode": mode,
        }
        return await self._gateways[tier](**kwargs)

    def snapshot(self) -> dict[str, Any]:
        gateways = {tier: gateway.snapshot() for tier, gateway in self._gateways.items()}
        return {
            "routing_enabled": config.get_model_routing_enabled(),
            "routing_mode": config.get_model_routing_mode(),
            "active_model": (
                self._last_selection["model"] if self._last_selection else self.primary.model
            ),
            "last_selection": self._last_selection,
            "selection_count": dict(self._selection_count),
            "escalation_count": self._escalation_count,
            "shadow_difference_count": self._shadow_count,
            "tiers": self.registry.snapshot(),
            "prompt_tokens": sum(item["prompt_tokens"] for item in gateways.values()),
            "completion_tokens": sum(item["completion_tokens"] for item in gateways.values()),
            "total_tokens": sum(item["total_tokens"] for item in gateways.values()),
            "fallback_configured": False,
            "fallback_activated": False,
            "gateways": gateways,
        }
