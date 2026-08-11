from __future__ import annotations

import json
from pathlib import Path


FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "benchmark_hardening"
    / "smoke15_failures.json"
)


def _payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_smoke15_hardening_fixture_records_the_audited_baseline() -> None:
    payload = _payload()

    assert payload["schema_version"] == 1
    assert payload["baseline"] == {
        "attempted": 15,
        "non_empty_patches": 8,
        "empty_patches": 7,
        "new_paid_instances": 10,
        "new_paid_token_usage": 645406,
    }
    assert set(payload["cases"]) == {
        "reasoning_only_then_mutation",
        "patch_without_explicit_file_path",
        "repeated_invalid_tool",
        "missing_dependency_validation",
        "corrupt_official_container",
        "empty_paid_prediction",
        "zero_token_preparation_failure",
        "terminal_discovery_is_not_test_evidence",
        "explicit_target_discovery_stagnation",
        "provider_requests_unexposed_tool",
        "benchmark_url_does_not_enable_mcp",
        "mutation_phase_requires_tool_call",
    }


def test_smoke15_hardening_fixture_is_sanitized_and_portable() -> None:
    payload = _payload()
    serialized = json.dumps(payload, ensure_ascii=False).lower()

    assert payload["contains_private_reasoning"] is False
    assert payload["contains_credentials"] is False
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "d:\\" not in serialized
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
