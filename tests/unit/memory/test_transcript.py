from __future__ import annotations

import json
from pathlib import Path

from codeagent.memory.transcript import build_transcript, persist_transcript
from codeagent.orchestration.state import AgentState


def test_transcript_persists_structured_evidence_and_redacts_secrets(tmp_path: Path) -> None:
    state = AgentState(
        user_request="fix",
        project_root=str(tmp_path),
        conversation_history=[{"role": "user", "content": "API_KEY=secret-value"}],
        execution_log=[{"type": "tool_call", "result": "sk-abcdefghijklmnop"}],
        reflection={"next_action": "finish"},
    )
    entries = build_transcript(state, success=True)
    path = Path(persist_transcript(str(tmp_path), "task-1", entries))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["task_id"] == "task-1"
    assert any(item["kind"] == "outcome" for item in payload["entries"])
    assert "secret-value" not in path.read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in path.read_text(encoding="utf-8")


def test_persist_transcript_redacts_raw_chat_entries(tmp_path: Path) -> None:
    path = Path(
        persist_transcript(
            str(tmp_path),
            "chat-task",
            [{"kind": "message", "content": "api_key=sk-direct-chat-secret"}],
        )
    )

    stored = path.read_text(encoding="utf-8")
    assert "sk-direct-chat-secret" not in stored
    assert "[REDACTED]" in stored
