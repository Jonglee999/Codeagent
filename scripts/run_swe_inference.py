"""Run a real, quota-consuming SWE smoke batch through the public HTTP API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


def _request(
    base_url: str,
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: int = 30,
) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {detail}") from exc
    if not body.get("success"):
        raise RuntimeError(body.get("error") or f"API request failed: {path}")
    return body.get("data")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        choices=("smoke-1", "smoke-5", "smoke-15", "smoke-30"),
        default="smoke-1",
    )
    parser.add_argument(
        "--instance",
        help="Run one allow-listed instance instead of a positional smoke prefix",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--confirm-api-cost", action="store_true")
    args = parser.parse_args()
    if not args.confirm_api_cost:
        parser.error("--confirm-api-cost is required because every task consumes LLM API quota")

    limit = int(args.set.split("-")[1])
    all_tasks = _request(args.base_url, "GET", "/api/v1/catalog/tasks")["tasks"]
    if args.instance:
        catalog = [task for task in all_tasks if task["instance_id"] == args.instance]
        if not catalog:
            parser.error(f"Unknown catalog instance: {args.instance}")
        limit = 1
    else:
        catalog = all_tasks[:limit]
    statuses = _request(args.base_url, "GET", "/api/v1/catalog/runs")["tasks"]
    exportable = {
        item["instance_id"] for item in statuses
        if item.get("prediction_ready")
    }
    failures: list[str] = []
    for index, task in enumerate(catalog, start=1):
        instance_id = task["instance_id"]
        if instance_id in exportable and not args.rerun_completed:
            print(f"[{index}/{limit}] skip exportable candidate {instance_id}")
            continue
        print(f"[{index}/{limit}] preparing {instance_id}")
        prepared = _request(
            args.base_url, "POST",
            f"/api/v1/catalog/tasks/{urllib.parse.quote(instance_id, safe='')}/prepare",
            timeout=600,
        )
        created = _request(args.base_url, "POST", "/api/v1/tasks", {
            "query": task["problem_statement"],
            "project_root": prepared["project_root"],
            "auto_mode": True,
            "max_retries": 3,
            "response_mode": "execute",
            "conversation_id": f"swe-{uuid.uuid4().hex}",
            "benchmark_instance_id": instance_id,
        })
        task_id = created["task_id"]
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            status = _request(args.base_url, "GET", f"/api/v1/tasks/{task_id}")
            if status["state"] in {"completed", "failed", "cancelled"}:
                print(f"[{index}/{limit}] {instance_id}: {status['state']}")
                if status["state"] != "completed":
                    failures.append(instance_id)
                break
            time.sleep(2)
        else:
            _request(args.base_url, "DELETE", f"/api/v1/tasks/{task_id}")
            failures.append(instance_id)
            print(f"[{index}/{limit}] {instance_id}: timed out")

    exported = _request(args.base_url, "POST", "/api/v1/catalog/predictions/export")
    print(f"Exported {exported['count']} prediction(s) to {exported['path']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
