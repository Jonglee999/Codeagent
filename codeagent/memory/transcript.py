"""Structured, bounded task transcript persistence."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_CONTENT = 4_000
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)


def _clean(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, str):
        text = value[:_MAX_CONTENT]
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda match: f"{match.group(1) if match.lastindex else ''}[REDACTED]", text)
        return text
    return value


def build_transcript(state: Any, success: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for message in getattr(state, "conversation_history", []) or []:
        entries.append({
            "kind": "message",
            "role": message.get("role", "unknown"),
            "content": _clean(message.get("content", "")),
        })
    for item in getattr(state, "execution_log", []) or []:
        item_type = item.get("type", "observation")
        entries.append({
            "kind": "decision" if item_type in {"llm_response", "plan_generated", "reflection"} else "observation",
            "type": item_type,
            "data": _clean(item),
        })
    for result in getattr(state, "validation_results", []) or []:
        entries.append({"kind": "validation", "data": _clean(result)})
    entries.append({
        "kind": "outcome",
        "success": success,
        "reflection": _clean(getattr(state, "reflection", None)),
        "warnings": _clean(getattr(state, "warnings", [])),
    })
    return entries


def persist_transcript(
    project_root: str,
    task_id: str,
    entries: list[dict[str, Any]],
) -> str:
    root = Path(project_root).resolve() / ".codeagent" / "transcripts"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{task_id}.json"
    payload = {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": _clean(entries),
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return str(destination)


def memory_messages(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = [
        {"role": entry.get("role", "user"), "content": str(entry.get("content", ""))}
        for entry in entries
        if entry.get("kind") == "message" and entry.get("role") in {"user", "assistant"}
    ]
    evidence = [entry for entry in entries if entry.get("kind") in {"validation", "outcome"}]
    messages.append({
        "role": "assistant",
        "content": "Verified execution evidence:\n" + json.dumps(evidence, ensure_ascii=False)[:_MAX_CONTENT],
    })
    return messages
