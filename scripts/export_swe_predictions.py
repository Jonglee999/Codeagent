"""Export real completed SWE smoke runs to upstream-compatible JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from codeagent.benchmarks import BenchmarkCatalog, PredictionExporter
from codeagent.product_state import product_state_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination", nargs="?", default=".codeagent/benchmarks/predictions.jsonl",
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--include-empty-attempts",
        action="store_true",
        help=(
            "Include an empty model_patch for terminal benchmark runs that made "
            "at least one successful paid model call"
        ),
    )
    args = parser.parse_args()
    exporter = PredictionExporter(BenchmarkCatalog(), product_state_store)
    if args.status:
        print(json.dumps(exporter.statuses(), ensure_ascii=False, indent=2))
        return 0
    destination = Path(args.destination)
    count = exporter.export(
        destination,
        os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        include_empty_attempts=args.include_empty_attempts,
    )
    print(f"Exported {count} completed real-run prediction(s) to {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
