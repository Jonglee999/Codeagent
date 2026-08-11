"""Official SWE-bench evaluator installation, reports, and status integration."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SWE_BENCH_REPOSITORY = "https://github.com/SWE-bench/SWE-bench.git"
SWE_BENCH_REVISION = "cd37836ffec01d01a0d699a80a039d84ff2cebfe"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Lite"

_INFRASTRUCTURE_LOG_SIGNATURES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("packfile", "does not match index"), "git_pack_corruption"),
    (("fatal: unable to read object",), "git_object_corruption"),
    (("fatal: bad object",), "git_object_corruption"),
    (("input/output error",), "container_io_error"),
    (("cannot connect to the docker daemon",), "docker_daemon_unavailable"),
    (("error during connect", "docker"), "docker_daemon_unavailable"),
    (("container setup failed",), "container_setup_failed"),
    (("failed to create container",), "container_setup_failed"),
)


def _read_prediction_patches(path: Path | str) -> dict[str, str]:
    candidate = Path(path) if isinstance(path, str) else path
    if not candidate.is_file():
        return {}
    patches: dict[str, str] = {}
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(item, dict)
            and isinstance(item.get("instance_id"), str)
            and isinstance(item.get("model_patch"), str)
        ):
            patches[item["instance_id"]] = item["model_patch"]
    return patches


def _scan_infrastructure_logs(instance_root: Path) -> tuple[str, str] | None:
    """Return the first bounded infrastructure signature found in raw logs."""

    if not instance_root.is_dir():
        return None
    for path in sorted(instance_root.rglob("*.log"))[:20]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:1_000_000]
        except OSError:
            continue
        lowered = content.lower()
        for needles, code in _INFRASTRUCTURE_LOG_SIGNATURES:
            if all(needle in lowered for needle in needles):
                return code, path.name
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def windows_to_wsl(path: Path) -> str:
    """Translate an absolute Windows path for the Ubuntu WSL distribution."""
    resolved = path.resolve()
    if os.name != "nt":
        return resolved.as_posix()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix()[3:]
    return f"/mnt/{drive}/{tail}"


@dataclass(frozen=True)
class OfficialEvaluatorPaths:
    repository_root: Path

    @property
    def checkout(self) -> Path:
        return self.repository_root / ".codeagent" / "evaluators" / "SWE-bench"

    @property
    def windows_python(self) -> Path:
        return self.checkout / ".venv" / "Scripts" / "python.exe"

    @property
    def wsl_python(self) -> Path:
        return self.checkout / ".venv-wsl" / "bin" / "python"

    @property
    def result_root(self) -> Path:
        return self.repository_root / ".codeagent" / "benchmarks" / "official"

    @property
    def index_path(self) -> Path:
        return self.result_root / "index.json"


class OfficialEvaluationStore:
    """Durable index over official harness reports stored below `.codeagent`."""

    def __init__(self, repository_root: Path | None = None) -> None:
        self.paths = OfficialEvaluatorPaths(
            (repository_root or Path(__file__).resolve().parents[2]).resolve()
        )

    def _load(self) -> dict[str, Any]:
        if not self.paths.index_path.is_file():
            return {
                "schema_version": 1,
                "runs": {},
                "latest_by_instance": {},
                "latest_gold_by_instance": {},
            }
        try:
            payload = json.loads(self.paths.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "schema_version": 1,
                "runs": {},
                "latest_by_instance": {},
                "latest_gold_by_instance": {},
            }
        payload.setdefault("schema_version", 1)
        payload.setdefault("runs", {})
        payload.setdefault("latest_by_instance", {})
        payload.setdefault("latest_gold_by_instance", {})
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        self.paths.result_root.mkdir(parents=True, exist_ok=True)
        temporary = self.paths.index_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.paths.index_path)

    def status_for(self, instance_id: str) -> dict[str, Any] | None:
        value = self._load()["latest_by_instance"].get(instance_id)
        return dict(cast(dict[str, Any], value)) if isinstance(value, dict) else None

    def summary(self) -> dict[str, Any]:
        payload = self._load()
        latest = list(payload["latest_by_instance"].values())
        return {
            "runs": list(payload["runs"].values()),
            "counts": {
                "evaluated": len(latest),
                "resolved": sum(item.get("resolved") is True for item in latest),
                "unresolved": sum(
                    item.get("evaluation_state") == "unresolved" for item in latest
                ),
                "empty_patches": sum(
                    item.get("evaluation_state") == "empty_patch" for item in latest
                ),
                "infrastructure_errors": sum(
                    item.get("evaluation_state") == "infrastructure_error"
                    for item in latest
                ),
                "errors": sum(item.get("completed") is False for item in latest),
            },
            "latest_by_instance": payload["latest_by_instance"],
            "latest_gold_by_instance": payload["latest_gold_by_instance"],
        }

    def import_run(
        self,
        run_id: str,
        *,
        model_name: str,
        instance_ids: list[str],
        predictions_path: Path | str,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Parse official summary and per-instance reports after evaluator exit."""
        run_root = self.paths.result_root / run_id
        model_slug = model_name.replace("/", "__")
        log_root = run_root / "logs" / "run_evaluation" / run_id / model_slug
        payload = self._load()
        imported: dict[str, Any] = {}
        run_kind = "gold_preflight" if model_name == "gold" else "prediction"
        prediction_patches = (
            {} if run_kind == "gold_preflight" else _read_prediction_patches(predictions_path)
        )
        for instance_id in instance_ids:
            report_path = log_root / instance_id / "report.json"
            instance_root = report_path.parent
            completed = report_path.is_file()
            resolved: bool | None = None
            error: str | None = None
            evaluation_state = "evaluation_error"
            infrastructure = _scan_infrastructure_logs(instance_root)
            is_empty_patch = (
                instance_id in prediction_patches
                and not prediction_patches[instance_id].strip()
            )
            if is_empty_patch:
                completed = True
                resolved = False
                error = None
                evaluation_state = "empty_patch"
            elif infrastructure is not None:
                completed = False
                resolved = None
                code, log_name = infrastructure
                error = f"Official evaluator infrastructure failure: {code} ({log_name})"
                evaluation_state = "infrastructure_error"
            elif completed:
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    report_resolved = report[instance_id]["resolved"]
                    if not isinstance(report_resolved, bool):
                        raise TypeError("resolved must be a boolean")
                    resolved = report_resolved
                    evaluation_state = "resolved" if resolved else "unresolved"
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    completed = False
                    error = f"Invalid official report: {type(exc).__name__}: {exc}"
            else:
                error = "Official report.json was not produced"
            item = {
                "instance_id": instance_id,
                "run_id": run_id,
                "completed": completed,
                "resolved": resolved,
                "error": error,
                "report_path": str(report_path.resolve()),
                "evaluated_at": _now(),
                "evaluator_revision": SWE_BENCH_REVISION,
                "kind": run_kind,
                "evaluation_state": evaluation_state,
            }
            imported[instance_id] = item
            target_index = (
                "latest_gold_by_instance"
                if run_kind == "gold_preflight"
                else "latest_by_instance"
            )
            payload[target_index][instance_id] = item

        attempted = len(instance_ids)
        completed_count = sum(item["completed"] for item in imported.values())
        resolved_count = sum(item["resolved"] is True for item in imported.values())
        unresolved_count = sum(
            item["evaluation_state"] == "unresolved" for item in imported.values()
        )
        empty_patches = sum(
            item["evaluation_state"] == "empty_patch" for item in imported.values()
        )
        infrastructure_errors = sum(
            item["evaluation_state"] == "infrastructure_error"
            for item in imported.values()
        )
        evaluation_errors = sum(
            item["evaluation_state"] == "evaluation_error"
            for item in imported.values()
        )
        prediction_sha256 = None
        prediction_file = Path(predictions_path) if isinstance(predictions_path, str) else predictions_path
        if prediction_file.is_file():
            prediction_sha256 = hashlib.sha256(prediction_file.read_bytes()).hexdigest()
        run_record = {
            "run_id": run_id,
            "kind": run_kind,
            "model_name": model_name,
            "instance_ids": instance_ids,
            "predictions_path": (
                str(predictions_path.resolve())
                if isinstance(predictions_path, Path)
                else predictions_path
            ),
            "result_root": str(run_root.resolve()),
            "evaluator_revision": SWE_BENCH_REVISION,
            "imported_at": _now(),
            "attempted": attempted,
            "completed": completed_count,
            "resolved": resolved_count,
            "unresolved": unresolved_count,
            "empty_patches": empty_patches,
            "infrastructure_errors": infrastructure_errors,
            "evaluation_errors": evaluation_errors,
            "score_percent": round((resolved_count / attempted) * 100, 2) if attempted else 0.0,
            "predictions_sha256": prediction_sha256,
            "parent_run_id": parent_run_id,
            "instance_results": imported,
        }
        if parent_run_id is not None:
            parent = payload["runs"].get(parent_run_id)
            if not isinstance(parent, dict):
                raise ValueError(f"Unknown parent official run: {parent_run_id}")
            parent_results = dict(parent.get("instance_results") or {})
            corrected_results = dict(parent_results)
            corrected_results.update(
                {
                    instance_id: item
                    for instance_id, item in imported.items()
                    if item["evaluation_state"]
                    in {"resolved", "unresolved", "empty_patch"}
                }
            )
            denominator = int(parent.get("attempted", len(parent_results)))
            corrected_resolved = sum(
                item.get("resolved") is True for item in corrected_results.values()
            )
            run_record["corrected_aggregate"] = {
                "parent_run_id": parent_run_id,
                "correction_run_id": run_id,
                "attempted": denominator,
                "resolved": corrected_resolved,
                "score_percent": (
                    round((corrected_resolved / denominator) * 100, 2)
                    if denominator
                    else 0.0
                ),
            }
        payload["runs"][run_id] = run_record
        self._save(payload)
        return {"run": run_record, "instances": imported}

    def doctor(self) -> dict[str, Any]:
        """Read-only official evaluator readiness report."""
        checkout = self.paths.checkout
        revision = ""
        if (checkout / ".git").exists():
            result = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
            revision = result.stdout.strip() if result.returncode == 0 else ""

        wsl_available = False
        wsl_python_ready = False
        docker_ready = False
        harness_ready = False
        dataset_network_ready = False
        if os.name == "nt" and shutil.which("wsl"):
            probe = _run(
                [
                    "wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                    "command -v python3 >/dev/null",
                ],
                timeout=30,
            )
            wsl_available = probe.returncode == 0
            docker_ready = _run(
                [
                    "wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                    "docker info >/dev/null 2>&1",
                ],
                timeout=30,
            ).returncode == 0
            dataset_network_ready = _run(
                [
                    "wsl", "-d", "Ubuntu", "--", "curl", "-fsS", "-o", "/dev/null",
                    "https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Lite",
                ],
                timeout=30,
            ).returncode == 0
            wsl_python = windows_to_wsl(self.paths.wsl_python)
            wsl_python_ready = _run(
                ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", f"test -x '{wsl_python}'"],
                timeout=15,
            ).returncode == 0
            if wsl_python_ready:
                checkout_wsl = windows_to_wsl(checkout)
                help_probe = _run(
                    [
                        "wsl", "-d", "Ubuntu", "--", "bash", "-lc",
                        f"cd '{checkout_wsl}' && '{wsl_python}' "
                        "-m swebench.harness.run_evaluation --help >/dev/null",
                    ],
                    timeout=60,
                )
                harness_ready = help_probe.returncode == 0
        elif platform.system() == "Linux":
            wsl_available = True
            docker_ready = _run(["docker", "info"], timeout=30).returncode == 0
            dataset_network_ready = _run(
                [
                    "curl", "-fsS", "-o", "/dev/null",
                    "https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Lite",
                ],
                timeout=30,
            ).returncode == 0
            harness_ready = False

        disk = shutil.disk_usage(self.paths.repository_root)
        return {
            "ready": all((
                revision == SWE_BENCH_REVISION,
                wsl_available,
                wsl_python_ready,
                docker_ready,
                harness_ready,
                dataset_network_ready,
            )),
            "checkout": str(checkout),
            "expected_revision": SWE_BENCH_REVISION,
            "actual_revision": revision,
            "revision_matches": revision == SWE_BENCH_REVISION,
            "wsl_ubuntu": wsl_available,
            "wsl_python": wsl_python_ready,
            "docker": docker_ready,
            "harness": harness_ready,
            "dataset_network": dataset_network_ready,
            "free_disk_gb": round(disk.free / (1024 ** 3), 1),
        }
