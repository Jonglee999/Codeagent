"""Build the curated SWE-bench Lite smoke manifest from the official dataset API.

The generated manifest intentionally excludes gold patches. Selection is based on
gold-patch size, but CodeAgent only receives the issue statement and repository at
runtime. This prevents accidental answer leakage while keeping the subset reproducible.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "princeton-nlp/SWE-bench_Lite"
SOURCE_URL = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite"
ROWS_API = "https://datasets-server.huggingface.co/rows"

SELECTED_IDS = [
    "django__django-11179",
    "django__django-12908",
    "django__django-13230",
    "django__django-13447",
    "django__django-15814",
    "matplotlib__matplotlib-23563",
    "matplotlib__matplotlib-25433",
    "pylint-dev__pylint-7080",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-6116",
    "astropy__astropy-12907",
    "astropy__astropy-6938",
    "django__django-10914",
    "django__django-10924",
    "django__django-11049",
    "django__django-11133",
    "django__django-12113",
    "django__django-12125",
    "django__django-13964",
    "django__django-14017",
    "django__django-14238",
    "django__django-14534",
    "matplotlib__matplotlib-23314",
    "mwaskom__seaborn-3190",
    "pydata__xarray-4094",
    "pylint-dev__pylint-7993",
    "pytest-dev__pytest-11148",
    "pytest-dev__pytest-5227",
    "pytest-dev__pytest-7168",
    "sympy__sympy-13647",
]


def _decode_test_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    return []


def _fetch_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for offset in range(0, 300, 100):
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": 100,
            }
        )
        request = urllib.request.Request(
            f"{ROWS_API}?{query}", headers={"User-Agent": "CodeAgent-Harness/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            payload = json.load(response)
        for item in payload.get("rows", []):
            row = item.get("row", {})
            instance_id = row.get("instance_id")
            if instance_id in SELECTED_IDS:
                rows[instance_id] = row
    return rows


def _changed_lines(patch: str) -> int:
    return sum(
        1
        for line in patch.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    )


def build_manifest(output: Path) -> None:
    rows = _fetch_rows()
    missing = sorted(set(SELECTED_IDS) - rows.keys())
    if missing:
        raise RuntimeError(f"Official dataset did not return selected instances: {missing}")

    tasks = []
    for instance_id in SELECTED_IDS:
        row = rows[instance_id]
        tasks.append(
            {
                "instance_id": instance_id,
                "repo": row["repo"],
                "base_commit": row["base_commit"],
                "version": row.get("version", ""),
                "problem_statement": row["problem_statement"],
                "fail_to_pass": _decode_test_list(row.get("FAIL_TO_PASS", [])),
                "pass_to_pass": _decode_test_list(row.get("PASS_TO_PASS", [])),
                "gold_patch_changed_lines": _changed_lines(row.get("patch", "")),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "split": "test",
        "source_url": SOURCE_URL,
        "selection": {
            "name": "codeagent-swe-smoke-30-v1",
            "count": len(tasks),
            "method": "Curated cross-project tasks with one production file and <=2 gold changed lines.",
            "answer_leakage": "Gold patches are not stored in this manifest.",
        },
        "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {output}")


if __name__ == "__main__":
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/swe_smoke/tasks.json")
    build_manifest(destination)
