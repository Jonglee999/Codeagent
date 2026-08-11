"""Manage the pinned official SWE-bench evaluator and import official scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codeagent.benchmarks import BenchmarkCatalog, OfficialEvaluationStore
from codeagent.benchmarks.official import (
    SWE_BENCH_DATASET,
    SWE_BENCH_REPOSITORY,
    SWE_BENCH_REVISION,
    windows_to_wsl,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_SAFE_IMAGE_NAMESPACE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")


def _checked(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> None:
    process = subprocess.run(arguments, cwd=cwd, check=False, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {arguments[0]}"
        )


def _wsl_bash(command: str, *, timeout: int | None = None) -> None:
    _checked(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
        timeout=timeout,
    )


def _wsl_result(
    command: str,
    *,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )


def _official_image_name(instance_id: str) -> str:
    suffix = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{suffix}:latest"


def _configured_image_namespaces() -> list[str]:
    configured = [
        item.strip().rstrip("/")
        for item in os.getenv("SWE_IMAGE_MIRRORS", "").split(",")
        if item.strip()
    ]
    for namespace in configured:
        if not _SAFE_IMAGE_NAMESPACE.fullmatch(namespace):
            raise ValueError(f"Unsafe SWE image mirror namespace: {namespace!r}")
    return [*configured, "swebench"]


def _snapshot_predictions(source: Path, run_root: Path) -> tuple[Path, str]:
    """Copy the exact evaluator input into its immutable run directory."""

    destination = run_root / "predictions.snapshot.jsonl"
    shutil.copyfile(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination, digest


def prepare_official_image(
    instance_id: str,
    store: OfficialEvaluationStore,
    *,
    retries: int = 2,
) -> dict[str, Any]:
    """Pull, retag, and integrity-probe one official instance image."""

    official = _official_image_name(instance_id)
    suffix = official.split("/", 1)[1]
    candidates = [
        f"{namespace}/{suffix}" for namespace in _configured_image_namespaces()
    ]
    attempts: list[dict[str, Any]] = []
    selected_source: str | None = None
    integrity_already_verified = False
    existing = _wsl_result(
        f"docker image inspect {shlex.quote(official)} >/dev/null",
        timeout=60,
    )
    if existing.returncode == 0:
        existing_integrity = _wsl_result(
            " ".join([
                "docker run --rm --entrypoint git",
                shlex.quote(official),
                "-C /testbed fsck --no-dangling",
            ]),
            timeout=300,
        )
        if existing_integrity.returncode == 0:
            selected_source = official
            integrity_already_verified = True
            attempts.append({
                "source": official,
                "attempt": 0,
                "returncode": 0,
                "reused": True,
            })
        else:
            removal = _wsl_result(
                f"docker image rm -f {shlex.quote(official)}",
                timeout=120,
            )
            attempts.append({
                "source": official,
                "attempt": 0,
                "returncode": existing_integrity.returncode,
                "reused": False,
                "removed_corrupt_image": removal.returncode == 0,
            })
    for source in dict.fromkeys(candidates):
        if selected_source:
            break
        for attempt in range(1, max(1, retries) + 1):
            pull = _wsl_result(
                f"docker pull {shlex.quote(source)}",
                timeout=1800,
            )
            attempts.append({
                "source": source,
                "attempt": attempt,
                "returncode": pull.returncode,
            })
            if pull.returncode == 0:
                selected_source = source
                break
        if selected_source:
            break
    if selected_source is None:
        raise RuntimeError(
            f"Could not pull official image for {instance_id} after bounded retries"
        )
    if selected_source != official:
        tag = _wsl_result(
            f"docker tag {shlex.quote(selected_source)} {shlex.quote(official)}",
            timeout=60,
        )
        if tag.returncode != 0:
            raise RuntimeError(f"Could not tag prepared image as {official}")

    if not integrity_already_verified:
        integrity = _wsl_result(
            " ".join([
                "docker run --rm --entrypoint git",
                shlex.quote(official),
                "-C /testbed fsck --no-dangling",
            ]),
            timeout=300,
        )
        if integrity.returncode != 0:
            raise RuntimeError(
                f"Prepared image failed /testbed/.git integrity probe: {instance_id}"
            )
    inspect_result = _wsl_result(
        "docker image inspect --format "
        + shlex.quote("{{json .RepoDigests}}|{{.Id}}")
        + " "
        + shlex.quote(official),
        timeout=60,
    )
    if inspect_result.returncode != 0:
        raise RuntimeError(f"Could not inspect prepared image: {official}")
    manifest_json, _, image_id = inspect_result.stdout.strip().partition("|")
    try:
        repo_digests = json.loads(manifest_json)
    except json.JSONDecodeError:
        repo_digests = []
    manifest_digest = None
    if isinstance(repo_digests, list) and repo_digests:
        manifest_digest = str(repo_digests[0]).partition("@")[2] or None
    record = {
        "instance_id": instance_id,
        "official_image": official,
        "source_image": selected_source,
        "repo_digests": repo_digests,
        "manifest_digest": manifest_digest,
        "image_id": image_id or None,
        "integrity_probe": "git -C /testbed fsck --no-dangling",
        "integrity_ok": True,
        "attempts": attempts,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    record_root = store.paths.result_root / "image-preparation"
    record_root.mkdir(parents=True, exist_ok=True)
    record_path = record_root / f"{instance_id.replace('/', '_')}.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record["record_path"] = str(record_path.resolve())
    return record


def setup_evaluator(store: OfficialEvaluationStore) -> dict[str, Any]:
    paths = store.paths
    checkout = paths.checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").is_dir():
        _checked(
            ["git", "clone", "--no-checkout", SWE_BENCH_REPOSITORY, str(checkout)],
            timeout=600,
        )
    _checked(
        ["git", "fetch", "origin", SWE_BENCH_REVISION, "--depth", "1"],
        cwd=checkout,
        timeout=600,
    )
    _checked(["git", "checkout", "--detach", SWE_BENCH_REVISION], cwd=checkout)

    if os.name != "nt":
        raise RuntimeError("This setup command currently targets the Windows + WSL Harness")
    checkout_wsl = windows_to_wsl(checkout)
    python_wsl = windows_to_wsl(paths.wsl_python)
    venv_wsl = python_wsl.removesuffix("/bin/python")
    lock_wsl = windows_to_wsl(
        REPOSITORY_ROOT / "evals" / "swe_official" / "requirements-lock-wsl.txt"
    )
    command = " && ".join(
        [
            f"cd {shlex.quote(checkout_wsl)}",
            (
                f"test -x {shlex.quote(python_wsl)} || "
                f"python3 -m venv {shlex.quote(venv_wsl)}"
            ),
            (
                f"{shlex.quote(python_wsl)} -m pip install -r "
                f"{shlex.quote(lock_wsl)}"
            ),
            f"{shlex.quote(python_wsl)} -m pip install --no-deps -e .",
            (
                f"{shlex.quote(python_wsl)} -m pip freeze > "
                f"{shlex.quote(checkout_wsl + '/requirements-lock-wsl.txt')}"
            ),
        ]
    )
    _wsl_bash(command, timeout=1800)
    return store.doctor()


def _load_predictions(path: Path) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid prediction JSON on line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Prediction line {line_number} must be an object")
        missing = {
            key for key in ("instance_id", "model_name_or_path")
            if not isinstance(item.get(key), str) or not item[key].strip()
        }
        if not isinstance(item.get("model_patch"), str):
            missing.add("model_patch")
        if missing:
            raise ValueError(
                f"Prediction line {line_number} has empty/missing fields: {sorted(missing)}"
            )
        predictions.append({
            "instance_id": item["instance_id"],
            "model_name_or_path": item["model_name_or_path"],
            "model_patch": item["model_patch"],
        })
    if not predictions:
        raise ValueError("Prediction file contains no predictions")
    instance_ids = [item["instance_id"] for item in predictions]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Prediction file contains duplicate instance_id values")
    return predictions


def validate_predictions(
    predictions_path: Path,
    store: OfficialEvaluationStore,
) -> dict[str, Any]:
    """Validate CodeAgent policy and load the JSONL with upstream SWE-bench code."""
    predictions_path = predictions_path.expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = _load_predictions(predictions_path)
    known = {task.instance_id for task in BenchmarkCatalog(REPOSITORY_ROOT).list_tasks()}
    unknown = {item["instance_id"] for item in predictions} - known
    if unknown:
        raise ValueError(f"Predictions contain non-catalog instances: {sorted(unknown)}")
    models = {item["model_name_or_path"] for item in predictions}
    if len(models) != 1:
        raise ValueError("One prediction file must use exactly one model_name_or_path")
    if os.name != "nt":
        raise RuntimeError("Pinned WSL evaluator is not installed; run swe-setup first")

    python_wsl = windows_to_wsl(store.paths.wsl_python)
    predictions_wsl = windows_to_wsl(predictions_path)
    validation_code = (
        "from swebench.harness.utils import get_predictions_from_file; "
        f"items=get_predictions_from_file({predictions_wsl!r}, "
        f"{SWE_BENCH_DATASET!r}, 'test'); "
        f"assert len(items) == {len(predictions)}"
    )
    _wsl_bash(
        f"{shlex.quote(python_wsl)} -c {shlex.quote(validation_code)}",
        timeout=120,
    )
    return {
        "valid": True,
        "predictions_path": str(predictions_path),
        "count": len(predictions),
        "model_name": next(iter(models)),
        "instance_ids": [item["instance_id"] for item in predictions],
        "loaded_by_official_revision": SWE_BENCH_REVISION,
    }


def evaluate(args: argparse.Namespace, store: OfficialEvaluationStore) -> dict[str, Any]:
    if not args.confirm_resources:
        raise ValueError(
            "--confirm-resources is required before official Docker image pull/build"
        )
    doctor = store.doctor()
    if not doctor["ready"]:
        raise RuntimeError(f"Official evaluator is not ready: {json.dumps(doctor)}")

    predictions_path = Path(args.predictions).expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = _load_predictions(predictions_path)
    known = {task.instance_id for task in BenchmarkCatalog(REPOSITORY_ROOT).list_tasks()}
    prediction_ids = [item["instance_id"] for item in predictions]
    unknown = set(prediction_ids) - known
    if unknown:
        raise ValueError(f"Predictions contain non-catalog instances: {sorted(unknown)}")

    selected = list(args.instance or prediction_ids)
    missing = set(selected) - set(prediction_ids)
    if missing:
        raise ValueError(f"Selected instances have no prediction: {sorted(missing)}")
    models = {item["model_name_or_path"] for item in predictions if item["instance_id"] in selected}
    if len(models) != 1:
        raise ValueError("One official run must use exactly one model_name_or_path")
    model_name = next(iter(models))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("codeagent-%Y%m%d-%H%M%S")
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and dash")

    run_root = store.paths.result_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    snapshot_path, snapshot_sha256 = _snapshot_predictions(
        predictions_path, run_root
    )
    patches_by_instance = {
        item["instance_id"]: item["model_patch"] for item in predictions
    }
    image_preparation = [
        prepare_official_image(instance_id, store)
        for instance_id in selected
        if patches_by_instance[instance_id].strip()
    ]
    python_wsl = windows_to_wsl(store.paths.wsl_python)
    predictions_wsl = windows_to_wsl(snapshot_path)
    run_root_wsl = windows_to_wsl(run_root)
    arguments = [
        shlex.quote(python_wsl),
        "-m", "swebench.harness.run_evaluation",
        "--dataset_name", shlex.quote(SWE_BENCH_DATASET),
        "--predictions_path", shlex.quote(predictions_wsl),
        "--max_workers", str(args.max_workers),
        "--cache_level", args.cache_level,
        "--clean", "True" if args.clean else "False",
        "--timeout", str(args.timeout),
        "--run_id", shlex.quote(run_id),
        "--report_dir", shlex.quote(run_root_wsl),
        "--instance_ids", *[shlex.quote(item) for item in selected],
    ]
    command = (
        f"cd {shlex.quote(run_root_wsl)} && "
        + " ".join(arguments)
    )
    metadata = {
        "run_id": run_id,
        "dataset": SWE_BENCH_DATASET,
        "model_name": model_name,
        "instance_ids": selected,
        "predictions_path": str(snapshot_path),
        "predictions_source_path": str(predictions_path),
        "predictions_sha256": snapshot_sha256,
        "image_preparation": image_preparation,
        "parent_run_id": args.parent_run_id,
        "evaluator_revision": SWE_BENCH_REVISION,
        "max_workers": args.max_workers,
        "cache_level": args.cache_level,
        "clean": args.clean,
        "timeout": args.timeout,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "codeagent-evaluation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        _wsl_bash(command, timeout=args.process_timeout)
    finally:
        result = store.import_run(
            run_id,
            model_name=model_name,
            instance_ids=selected,
            predictions_path=snapshot_path,
            parent_run_id=args.parent_run_id,
        )
    return result


def evaluate_gold(
    args: argparse.Namespace,
    store: OfficialEvaluationStore,
) -> dict[str, Any]:
    """Run one official gold patch as a no-model evaluator preflight."""
    if not args.confirm_resources:
        raise ValueError(
            "--confirm-resources is required before official Docker image pull/build"
        )
    doctor = store.doctor()
    if not doctor["ready"]:
        raise RuntimeError(f"Official evaluator is not ready: {json.dumps(doctor)}")
    try:
        BenchmarkCatalog(REPOSITORY_ROOT).get_task(args.instance)
    except KeyError as exc:
        raise ValueError(f"Unknown catalog instance: {args.instance}") from exc

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "codeagent-gold-%Y%m%d-%H%M%S"
    )
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, underscore, and dash")
    run_root = store.paths.result_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    image_preparation = prepare_official_image(args.instance, store)
    python_wsl = windows_to_wsl(store.paths.wsl_python)
    run_root_wsl = windows_to_wsl(run_root)
    arguments = [
        shlex.quote(python_wsl),
        "-m", "swebench.harness.run_evaluation",
        "--dataset_name", shlex.quote(SWE_BENCH_DATASET),
        "--predictions_path", "gold",
        "--max_workers", "1",
        "--cache_level", args.cache_level,
        "--clean", "True" if args.clean else "False",
        "--timeout", str(args.timeout),
        "--run_id", shlex.quote(run_id),
        "--report_dir", shlex.quote(run_root_wsl),
        "--instance_ids", shlex.quote(args.instance),
    ]
    metadata = {
        "run_id": run_id,
        "dataset": SWE_BENCH_DATASET,
        "model_name": "gold",
        "instance_ids": [args.instance],
        "predictions_path": "gold",
        "evaluator_revision": SWE_BENCH_REVISION,
        "max_workers": 1,
        "cache_level": args.cache_level,
        "clean": args.clean,
        "timeout": args.timeout,
        "image_preparation": [image_preparation],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "codeagent-evaluation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    command = f"cd {shlex.quote(run_root_wsl)} && " + " ".join(arguments)
    try:
        _wsl_bash(command, timeout=args.process_timeout)
    finally:
        result = store.import_run(
            run_id,
            model_name="gold",
            instance_ids=[args.instance],
            predictions_path="gold",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("setup")
    subparsers.add_parser("doctor")
    subparsers.add_parser("results")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--predictions",
        default=".codeagent/benchmarks/predictions.jsonl",
    )
    gold_parser = subparsers.add_parser("gold")
    gold_parser.add_argument("--instance", required=True)
    gold_parser.add_argument("--run-id")
    gold_parser.add_argument(
        "--cache-level", choices=("none", "base", "env", "instance"), default="env"
    )
    gold_parser.add_argument(
        "--clean", action=argparse.BooleanOptionalAction, default=True
    )
    gold_parser.add_argument("--timeout", type=int, default=1800)
    gold_parser.add_argument("--process-timeout", type=int, default=7200)
    gold_parser.add_argument("--confirm-resources", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--predictions",
        default=".codeagent/benchmarks/predictions.jsonl",
    )
    evaluate_parser.add_argument("--instance", action="append")
    evaluate_parser.add_argument("--run-id")
    evaluate_parser.add_argument("--max-workers", type=int, default=1, choices=range(1, 17))
    evaluate_parser.add_argument("--cache-level", choices=("none", "base", "env", "instance"), default="env")
    evaluate_parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    evaluate_parser.add_argument("--timeout", type=int, default=1800)
    evaluate_parser.add_argument("--process-timeout", type=int, default=7200)
    evaluate_parser.add_argument("--parent-run-id")
    evaluate_parser.add_argument("--confirm-resources", action="store_true")
    image_parser = subparsers.add_parser("image-prepare")
    image_parser.add_argument("--instance", required=True)
    image_parser.add_argument("--retries", type=int, default=2, choices=range(1, 5))
    image_parser.add_argument("--confirm-resources", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = OfficialEvaluationStore(REPOSITORY_ROOT)
    try:
        if args.command == "setup":
            result = setup_evaluator(store)
        elif args.command == "doctor":
            result = store.doctor()
        elif args.command == "results":
            result = store.summary()
        elif args.command == "validate":
            result = validate_predictions(Path(args.predictions), store)
        elif args.command == "gold":
            result = evaluate_gold(args, store)
        elif args.command == "image-prepare":
            if not args.confirm_resources:
                raise ValueError("--confirm-resources is required before image preparation")
            try:
                BenchmarkCatalog(REPOSITORY_ROOT).get_task(args.instance)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown catalog instance: {args.instance}"
                ) from exc
            result = prepare_official_image(
                args.instance, store, retries=args.retries
            )
        else:
            result = evaluate(args, store)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"SWE evaluator error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
