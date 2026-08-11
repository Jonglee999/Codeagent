from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "evals" / "swe_smoke" / "tasks.json"


def test_smoke_manifest_is_reproducible_and_answer_free() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert payload["selection"]["count"] == 30
    assert len(payload["tasks"]) == 30

    ids = [task["instance_id"] for task in payload["tasks"]]
    assert len(ids) == len(set(ids))
    for task in payload["tasks"]:
        assert task["problem_statement"].strip()
        assert task["base_commit"]
        assert task["gold_patch_changed_lines"] <= 2
        assert "patch" not in task
        assert "test_patch" not in task


def test_harness_exposes_required_commands() -> None:
    harness = (ROOT / "scripts" / "harness.ps1").read_text(encoding="utf-8")
    for command in ("setup", "doctor", "up", "check", "test", "eval", "verify"):
        assert f"'{command}'" in harness
