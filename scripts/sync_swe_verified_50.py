"""Build a frozen, stratified SWE-bench Verified 50-task manifest.

Selection may inspect gold patches to estimate difficulty, but the inference
manifest deliberately excludes patch contents, target paths, difficulty labels,
and gold-derived statistics.  Analysis metadata is written separately and must
never be included in an Agent prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as parquet


DATASET = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DATASET_FILE = "data/test-00000-of-00001.parquet"
SOURCE_URL = f"https://huggingface.co/datasets/{DATASET}"
SELECTION_SEED = "codeagent-verified-50-v1"
TIERS = ("L1", "L2", "L3", "L4", "L5")
BATCHES = (
    ("dev-1", "development"),
    ("dev-2", "development"),
    ("iteration-1", "iteration"),
    ("iteration-2", "iteration"),
    ("blind-holdout", "holdout"),
)
TASKS_PER_TIER = 10
MAX_TASKS_PER_REPOSITORY = 8
MIN_REPOSITORIES = 8

_TEST_PATH = re.compile(
    r"(^|/)(?:test|tests|testing)/|(^|/)test_[^/]+$|(?:^|/)[^/]+_test\.py$",
    re.IGNORECASE,
)
_STATEFUL_ISSUE = re.compile(
    r"cache|state|serializ|config|setting|lifecycle|session|transaction|"
    r"compatib|deprecat|migration|registry|metadata|environment|database|"
    r"queryset|middleware",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PatchStats:
    production_files: int
    changed_lines: int


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    tier: str
    stats: PatchStats

    @property
    def instance_id(self) -> str:
        return str(self.row["instance_id"])

    @property
    def repo(self) -> str:
        return str(self.row["repo"])


def _decode_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    return []


def patch_stats(patch: str) -> PatchStats:
    files: list[str] = []
    changed_lines = 0
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            files.append(match.group(2) if match else line)
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed_lines += 1
    unique_files = sorted(set(files))
    production = [path for path in unique_files if not _TEST_PATH.search(path)]
    return PatchStats(
        production_files=len(production or unique_files),
        changed_lines=changed_lines,
    )


def classify_tier(row: dict[str, Any]) -> tuple[str, PatchStats]:
    """Assign a selection-only difficulty proxy; it is never sent to the Agent."""
    stats = patch_stats(str(row.get("patch", "")))
    problem = str(row.get("problem_statement", ""))
    if stats.production_files >= 4 or stats.changed_lines > 80 or len(problem) > 5_000:
        return "L5", stats
    if _STATEFUL_ISSUE.search(problem):
        return "L4", stats
    if stats.production_files >= 2:
        return "L3", stats
    if stats.changed_lines > 8:
        return "L2", stats
    return "L1", stats


def _stable_key(instance_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{instance_id}".encode()).hexdigest()


def select_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    excluded_ids: set[str] | None = None,
) -> dict[str, list[Candidate]]:
    excluded = excluded_ids or set()
    candidates: dict[str, list[Candidate]] = {tier: [] for tier in TIERS}
    for row in rows:
        if str(row["instance_id"]) in excluded:
            continue
        tier, stats = classify_tier(row)
        candidates[tier].append(Candidate(row=row, tier=tier, stats=stats))
    for tier in TIERS:
        candidates[tier].sort(key=lambda item: _stable_key(item.instance_id))

    repository_counts: Counter[str] = Counter()
    selected: dict[str, list[Candidate]] = {tier: [] for tier in TIERS}
    for tier in TIERS:
        for candidate in candidates[tier]:
            if repository_counts[candidate.repo] >= MAX_TASKS_PER_REPOSITORY:
                continue
            selected[tier].append(candidate)
            repository_counts[candidate.repo] += 1
            if len(selected[tier]) == TASKS_PER_TIER:
                break
        if len(selected[tier]) != TASKS_PER_TIER:
            raise RuntimeError(
                f"Could not select {TASKS_PER_TIER} tasks for {tier}; "
                f"selected {len(selected[tier])}"
            )

    selected_repositories = {item.repo for tier in TIERS for item in selected[tier]}
    if len(selected_repositories) < MIN_REPOSITORIES:
        raise RuntimeError(
            f"Selection covers only {len(selected_repositories)} repositories; "
            f"requires at least {MIN_REPOSITORIES}"
        )
    return selected


def assign_batches(selected: dict[str, list[Candidate]]) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for batch_index, (name, purpose) in enumerate(BATCHES):
        tasks: list[Candidate] = []
        for tier in TIERS:
            start = batch_index * 2
            tasks.extend(selected[tier][start : start + 2])
        tasks.sort(key=lambda item: _stable_key(f"{name}:{item.instance_id}"))
        batches.append(
            {
                "name": name,
                "purpose": purpose,
                "instance_ids": [item.instance_id for item in tasks],
            }
        )
    return batches


def inference_task(candidate: Candidate) -> dict[str, Any]:
    row = candidate.row
    return {
        "instance_id": candidate.instance_id,
        "repo": candidate.repo,
        "base_commit": str(row["base_commit"]),
        "version": str(row.get("version", "")),
        "problem_statement": str(row["problem_statement"]),
        "fail_to_pass": _decode_list(row.get("FAIL_TO_PASS", [])),
        "pass_to_pass": _decode_list(row.get("PASS_TO_PASS", [])),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _download_rows(revision: str) -> list[dict[str, Any]]:
    url = f"{SOURCE_URL}/resolve/{revision}/{DATASET_FILE}?download=true"
    request = urllib.request.Request(url, headers={"User-Agent": "CodeAgent-Harness/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        content = response.read()
    table = parquet.read_table(io.BytesIO(content))
    rows = [dict(row) for row in table.to_pylist()]
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 Verified rows, received {len(rows)}")
    return rows


def _excluded_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", payload) if isinstance(payload, dict) else payload
    return {str(item["instance_id"]) for item in tasks}


def build(output: Path, *, revision: str, exclude_manifest: Path | None) -> None:
    rows = _download_rows(revision)
    excluded = _excluded_ids(exclude_manifest)
    selected = select_candidates(rows, excluded_ids=excluded)
    batches = assign_batches(selected)
    by_id = {
        item.instance_id: item for tier in TIERS for item in selected[tier]
    }
    ordered_ids = [instance_id for batch in batches for instance_id in batch["instance_ids"]]

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_revision": revision,
        "split": "test",
        "source_url": SOURCE_URL,
        "selection": {
            "name": SELECTION_SEED,
            "count": len(ordered_ids),
            "gold_data_in_inference_manifest": False,
            "excluded_prior_smoke_tasks": len(excluded),
        },
        "tasks": [inference_task(by_id[instance_id]) for instance_id in ordered_ids],
    }
    manifest_bytes = _json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    splits = {
        "schema_version": 1,
        "selection": SELECTION_SEED,
        "tasks_sha256": manifest_sha256,
        "batches": batches,
    }
    analysis = {
        "schema_version": 1,
        "selection": SELECTION_SEED,
        "warning": "Selection-only gold-derived metadata; never include in Agent prompts.",
        "tasks_sha256": manifest_sha256,
        "repository_counts": dict(
            sorted(Counter(item.repo for item in by_id.values()).items())
        ),
        "tasks": [
            {
                "instance_id": instance_id,
                "tier": by_id[instance_id].tier,
                "production_file_count": by_id[instance_id].stats.production_files,
                "gold_patch_changed_lines": by_id[instance_id].stats.changed_lines,
            }
            for instance_id in ordered_ids
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "tasks.json").write_bytes(manifest_bytes)
    (output / "splits.json").write_bytes(_json_bytes(splits))
    (output / "selection-analysis.json").write_bytes(_json_bytes(analysis))
    print(f"Wrote {len(ordered_ids)} tasks to {output}")
    print(f"tasks.json sha256: {manifest_sha256}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evals/swe_verified_50"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=Path("evals/swe_smoke/tasks.json"),
        help="Prior tasks excluded to reduce benchmark contamination.",
    )
    args = parser.parse_args()
    build(
        args.output,
        revision=args.dataset_revision,
        exclude_manifest=args.exclude_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
