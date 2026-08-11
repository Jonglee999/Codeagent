from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sync_swe_verified_50.py"
_SPEC = importlib.util.spec_from_file_location("sync_swe_verified_50", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
selection = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = selection
_SPEC.loader.exec_module(selection)


def _row(instance_id: str, patch: str, problem: str = "Correct the result") -> dict:
    return {
        "instance_id": instance_id,
        "repo": "example/project",
        "base_commit": "abc123",
        "version": "1",
        "problem_statement": problem,
        "patch": patch,
        "FAIL_TO_PASS": ["tests/test_regression.py::test_case"],
        "PASS_TO_PASS": [],
    }


def _patch(*paths: str, changed_lines: int = 2) -> str:
    sections = []
    for path in paths:
        changes = "\n".join(f"+line {index}" for index in range(changed_lines))
        sections.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{changes}")
    return "\n".join(sections)


def test_classification_covers_structural_and_stateful_tiers() -> None:
    assert selection.classify_tier(_row("l1", _patch("pkg/a.py")))[0] == "L1"
    assert selection.classify_tier(
        _row("l2", _patch("pkg/a.py", changed_lines=9))
    )[0] == "L2"
    assert selection.classify_tier(
        _row("l3", _patch("pkg/a.py", "pkg/b.py"))
    )[0] == "L3"
    assert selection.classify_tier(
        _row("l4", _patch("pkg/a.py"), "Invalidate the cache after an update")
    )[0] == "L4"
    assert selection.classify_tier(
        _row("l5", _patch("pkg/a.py", changed_lines=81))
    )[0] == "L5"


def test_test_files_do_not_inflate_production_file_count() -> None:
    stats = selection.patch_stats(_patch("pkg/a.py", "tests/test_a.py"))
    assert stats.production_files == 1


def test_inference_task_excludes_gold_derived_fields() -> None:
    row = _row("example__project-1", _patch("pkg/a.py"))
    tier, stats = selection.classify_tier(row)
    candidate = selection.Candidate(row=row, tier=tier, stats=stats)

    task = selection.inference_task(candidate)

    assert set(task) == {
        "instance_id",
        "repo",
        "base_commit",
        "version",
        "problem_statement",
        "fail_to_pass",
        "pass_to_pass",
    }
    assert "patch" not in task
    assert "tier" not in task
    assert "gold_patch_changed_lines" not in task


def test_quality_audit_excludes_known_implementation_specific_task() -> None:
    audit = json.loads(
        Path("evals/swe_verified_50/quality-audit.json").read_text(encoding="utf-8")
    )

    finding = audit["instances"]["pylint-dev__pylint-4551"]
    assert finding["diagnostic_use"] == "mechanism_only"
    assert finding["capability_gate_eligible"] is False
    assert finding["test_quality"] == "implementation_specific"
