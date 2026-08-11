from __future__ import annotations

import json
import subprocess

from codeagent.benchmarks import BenchmarkCatalog, PredictionExporter
from codeagent.orchestration.runtime import persist_run_artifacts
from codeagent.product_state import ProductStateStore


def _catalog(tmp_path, base_commit: str) -> BenchmarkCatalog:
    manifest = tmp_path / "evals/swe_smoke/tasks.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tasks": [{
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": base_commit,
        "version": "1",
        "problem_statement": "Fix it",
        "fail_to_pass": [],
        "pass_to_pass": [],
        "gold_patch_changed_lines": 1,
    }]}), encoding="utf-8")
    return BenchmarkCatalog(tmp_path)


def _record_run_with_patch(
    tmp_path,
    *,
    state: str,
    change_source: bool = True,
    add_cancelled_latest: bool = False,
    token_usage: int | None = None,
):
    placeholder = tmp_path / "placeholder"
    placeholder.mkdir()
    subprocess.run(["git", "init"], cwd=placeholder, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.test"], cwd=placeholder, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent Test"], cwd=placeholder, check=True,
    )
    (placeholder / "fix.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "fix.py"], cwd=placeholder, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=placeholder, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=placeholder, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    catalog = _catalog(tmp_path, base_commit)
    workspace = catalog.workspace_for("owner__repo-1", "run-1")
    workspace.parent.mkdir(parents=True)
    placeholder.replace(workspace)
    source = workspace / "fix.py"
    marker = workspace / ".git" / "codeagent-benchmark.json"
    marker.write_text(json.dumps({
        "schema_version": 1,
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": base_commit,
    }), encoding="utf-8")
    if change_source:
        source.write_text("value = 2\n", encoding="utf-8")

    store = ProductStateStore(tmp_path / "state.sqlite3")
    store.record_run(
        "task-1",
        conversation_id="conversation-1",
        query="Fix it",
        workspace_root=str(workspace),
        benchmark_instance_id="owner__repo-1",
    )
    report = {"task_id": "task-1", "validation_results": [{"passed": True}]}
    if token_usage is not None:
        report["token_usage"] = token_usage
        report["model_runtime"] = {"successful_calls": int(token_usage > 0)}
    persist_run_artifacts(str(workspace), "task-1", report)
    store.save_report("task-1", report)
    store.update_run_state("task-1", state)
    if add_cancelled_latest:
        store.record_run(
            "task-2",
            conversation_id="conversation-2",
            query="Fix it again",
            workspace_root=str(tmp_path / "cancelled-workspace"),
            benchmark_instance_id="owner__repo-1",
        )
        store.update_run_state("task-2", "cancelled")
    return PredictionExporter(catalog, store)


def test_prediction_export_uses_completed_explicit_benchmark_run(tmp_path) -> None:
    exporter = _record_run_with_patch(tmp_path, state="completed")

    status = exporter.statuses()
    assert status["counts"]["completed"] == 1
    assert status["tasks"][0]["prediction_ready"] is True
    assert status["tasks"][0]["prediction_source_state"] == "completed"
    assert status["tasks"][0]["officially_resolved"] is None
    assert status["benchmark_summary"]["attempted"] == 1
    assert status["benchmark_summary"]["patches"] == 1
    assert status["benchmark_summary"]["patch_yield_percent"] == 100.0
    destination = tmp_path / "predictions.jsonl"
    assert exporter.export(destination, "test/model") == 1
    prediction = json.loads(destination.read_text(encoding="utf-8"))
    assert prediction["instance_id"] == "owner__repo-1"
    assert prediction["model_name_or_path"] == "test/model"
    assert "value = 2" in prediction["model_patch"]


def test_prediction_export_keeps_failed_run_with_valid_patch(tmp_path) -> None:
    exporter = _record_run_with_patch(tmp_path, state="failed")

    status = exporter.statuses()
    assert status["counts"]["failed"] == 1
    assert status["tasks"][0]["prediction_ready"] is True
    assert status["tasks"][0]["prediction_source_state"] == "failed"
    destination = tmp_path / "predictions.jsonl"
    assert exporter.export(destination, "test/model") == 1
    prediction = json.loads(destination.read_text(encoding="utf-8"))
    assert "value = 2" in prediction["model_patch"]


def test_prediction_export_rejects_failed_run_without_patch(tmp_path) -> None:
    exporter = _record_run_with_patch(tmp_path, state="failed", change_source=False)

    status = exporter.statuses()
    assert status["tasks"][0]["prediction_ready"] is False
    assert status["tasks"][0]["prediction_source_state"] is None
    assert status["tasks"][0]["prediction_error"] == (
        "failed run has no non-empty git patch"
    )
    assert exporter.export(tmp_path / "predictions.jsonl", "test/model") == 0


def test_explicit_empty_attempt_mode_keeps_paid_failure_in_denominator(tmp_path) -> None:
    exporter = _record_run_with_patch(
        tmp_path, state="failed", change_source=False, token_usage=42
    )
    destination = tmp_path / "predictions.jsonl"

    assert exporter.export(
        destination, "test/model", include_empty_attempts=True
    ) == 1
    prediction = json.loads(destination.read_text(encoding="utf-8"))
    assert prediction["instance_id"] == "owner__repo-1"
    assert prediction["model_patch"] == ""


def test_empty_attempt_mode_excludes_zero_token_preparation_failure(tmp_path) -> None:
    exporter = _record_run_with_patch(
        tmp_path, state="failed", change_source=False, token_usage=0
    )

    assert exporter.export(
        tmp_path / "predictions.jsonl",
        "test/model",
        include_empty_attempts=True,
    ) == 0


def test_cancelled_retry_does_not_hide_earlier_valid_candidate(tmp_path) -> None:
    exporter = _record_run_with_patch(
        tmp_path,
        state="failed",
        add_cancelled_latest=True,
    )

    status = exporter.statuses()
    assert status["counts"]["cancelled"] == 1
    assert status["tasks"][0]["inference_state"] == "cancelled"
    assert status["tasks"][0]["prediction_ready"] is True
    assert status["tasks"][0]["prediction_source_state"] == "failed"
    assert status["tasks"][0]["prediction_task_id"] == "task-1"
    assert exporter.export(tmp_path / "predictions.jsonl", "test/model") == 1
