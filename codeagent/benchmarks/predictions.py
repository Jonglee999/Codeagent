"""Durable SWE inference status and official-format prediction export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeagent.orchestration.runtime import collect_git_patch
from codeagent.product_state import ProductStateStore

from .catalog import BenchmarkCatalog
from .official import OfficialEvaluationStore


class PredictionExporter:
    """Build prediction JSONL from explicitly tagged, real Agent runs."""

    _CANDIDATE_STATES = frozenset({"completed", "failed"})

    def __init__(
        self,
        catalog: BenchmarkCatalog,
        state_store: ProductStateStore,
    ) -> None:
        self.catalog = catalog
        self.state_store = state_store
        self.official_store = OfficialEvaluationStore(catalog.repository_root)

    def statuses(self) -> dict[str, Any]:
        runs = {item["instance_id"]: item for item in self.state_store.benchmark_runs()}
        history: dict[str, list[dict[str, Any]]] = {}
        for item in self.state_store.benchmark_run_history():
            history.setdefault(item["instance_id"], []).append(item)
        items = []
        for task in self.catalog.list_tasks():
            run = runs.get(task.instance_id)
            official = self.official_store.status_for(task.instance_id)
            candidate, prediction_error = self._candidate_run(
                task.instance_id, history.get(task.instance_id, [])
            )
            prediction_ready = candidate is not None
            items.append({
                "instance_id": task.instance_id,
                "repo": task.repo,
                "inference_state": run["state"] if run else "not_started",
                "task_id": run["task_id"] if run else None,
                "updated_at": run["updated_at"] if run else None,
                "prediction_ready": prediction_ready,
                "prediction_error": prediction_error,
                "prediction_source_state": candidate["state"] if candidate else None,
                "prediction_task_id": candidate["task_id"] if candidate else None,
                "official_evaluation_state": (
                    official.get("evaluation_state", "completed")
                    if official
                    else "not_evaluated"
                ),
                "officially_resolved": official.get("resolved") if official else None,
                "official_run_id": official.get("run_id") if official else None,
                "official_report_path": official.get("report_path") if official else None,
                "benchmark_metrics": (
                    (run.get("report") or {}).get("benchmark_metrics", {})
                    if run
                    else {}
                ),
            })
        counts = {
            state: sum(item["inference_state"] == state for item in items)
            for state in ("not_started", "pending", "running", "completed", "failed", "cancelled")
        }
        attempted = sum(item["inference_state"] != "not_started" for item in items)
        patches = sum(bool(item["prediction_ready"]) for item in items)
        resolved = sum(item["officially_resolved"] is True for item in items)
        total_tokens = sum(
            int(((run.get("report") or {}).get("token_usage", 0)) or 0)
            for run in runs.values()
        )
        benchmark_summary = {
            "attempted": attempted,
            "patches": patches,
            "patch_yield_percent": (
                round((patches / attempted) * 100, 2) if attempted else 0.0
            ),
            "resolved": resolved,
            "resolved_yield_percent": (
                round((resolved / attempted) * 100, 2) if attempted else 0.0
            ),
            "total_tokens": total_tokens,
            "tokens_per_attempted": (
                round(total_tokens / attempted, 2) if attempted else 0.0
            ),
            "tokens_per_resolved": (
                round(total_tokens / resolved, 2) if resolved else None
            ),
        }
        return {
            "tasks": items,
            "counts": counts,
            "total": len(items),
            "benchmark_summary": benchmark_summary,
        }

    @staticmethod
    def _artifact_patch(run: dict[str, Any]) -> str:
        workspace = Path(run["workspace_root"]).resolve()
        report = run.get("report") or {}
        for artifact in report.get("artifacts", []) or []:
            if not isinstance(artifact, dict) or artifact.get("kind") != "git_diff":
                continue
            candidate = Path(str(artifact.get("path") or "")).resolve()
            try:
                candidate.relative_to(workspace)
            except ValueError:
                continue
            if candidate.is_file():
                patch = candidate.read_text(encoding="utf-8")
                if patch.strip() and not patch.startswith("# git artifact unavailable"):
                    return patch
        return collect_git_patch(str(workspace)) if workspace.is_dir() else ""

    def _candidate_run(
        self,
        instance_id: str,
        runs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        candidate_error: str | None = None
        for run in runs:
            if run["state"] not in self._CANDIDATE_STATES:
                continue
            valid, reason = self.catalog.validate_workspace(
                instance_id, run["workspace_root"], require_clean=False
            )
            if not valid:
                candidate_error = candidate_error or reason
                continue
            patch = self._artifact_patch(run)
            if patch.strip() and not patch.startswith("# git artifact unavailable"):
                return run, None
            candidate_error = candidate_error or (
                f"{run['state']} run has no non-empty git patch"
            )
        return None, candidate_error

    def _attempted_empty_run(
        self,
        instance_id: str,
        runs: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the newest genuine paid attempt eligible for an empty prediction."""
        for run in runs:
            if run["state"] not in self._CANDIDATE_STATES:
                continue
            valid, _reason = self.catalog.validate_workspace(
                instance_id, run["workspace_root"], require_clean=False
            )
            if not valid:
                continue
            report = run.get("report") or {}
            runtime = report.get("model_runtime") or {}
            successful_calls = int(runtime.get("successful_calls") or 0)
            token_usage = int(report.get("token_usage") or 0)
            if successful_calls > 0 or token_usage > 0:
                return run
        return None

    def predictions(
        self,
        model_name: str,
        *,
        include_empty_attempts: bool = False,
    ) -> list[dict[str, str]]:
        known = [task.instance_id for task in self.catalog.list_tasks()]
        history: dict[str, list[dict[str, Any]]] = {}
        for run in self.state_store.benchmark_run_history():
            history.setdefault(run["instance_id"], []).append(run)
        predictions = []
        for instance_id in known:
            run, _error = self._candidate_run(instance_id, history.get(instance_id, []))
            if run is None:
                if not include_empty_attempts or self._attempted_empty_run(
                    instance_id, history.get(instance_id, [])
                ) is None:
                    continue
                patch = ""
            else:
                patch = self._artifact_patch(run)
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            })
        return sorted(predictions, key=lambda item: item["instance_id"])

    def export(
        self,
        destination: Path,
        model_name: str,
        *,
        include_empty_attempts: bool = False,
    ) -> int:
        predictions = self.predictions(
            model_name, include_empty_attempts=include_empty_attempts
        )
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return len(predictions)
