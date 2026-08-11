from __future__ import annotations

import json
import subprocess

from codeagent.benchmarks.official import OfficialEvaluationStore
from scripts import swe_evaluator
from scripts.swe_evaluator import (
    _load_predictions,
    _official_image_name,
    _snapshot_predictions,
    prepare_official_image,
)


def test_official_loader_contract_allows_empty_model_patch(tmp_path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({
            "instance_id": "owner__repo-1",
            "model_name_or_path": "test/model",
            "model_patch": "",
        }) + "\n",
        encoding="utf-8",
    )

    assert _load_predictions(predictions)[0]["model_patch"] == ""


def test_prediction_snapshot_preserves_exact_bytes_and_records_digest(tmp_path) -> None:
    source = tmp_path / "input.jsonl"
    source.write_bytes(b'{"instance_id":"x"}\r\n')
    run_root = tmp_path / "run"
    run_root.mkdir()

    snapshot, digest = _snapshot_predictions(source, run_root)

    assert snapshot.read_bytes() == source.read_bytes()
    assert len(digest) == 64


def test_official_image_name_matches_pinned_harness_convention() -> None:
    assert _official_image_name("django__django-13230") == (
        "swebench/sweb.eval.x86_64.django_1776_django-13230:latest"
    )


def test_image_prepare_reuses_only_after_integrity_probe(
    tmp_path, monkeypatch
) -> None:
    commands: list[str] = []

    def fake_wsl(command: str, *, timeout: int | None = None):
        commands.append(command)
        if "--format" in command:
            return subprocess.CompletedProcess(
                [], 0, '["image@sha256:digest"]|sha256:image-id\n', ""
            )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(swe_evaluator, "_wsl_result", fake_wsl)

    result = prepare_official_image(
        "django__django-13230",
        OfficialEvaluationStore(tmp_path),
    )

    assert sum("fsck --no-dangling" in command for command in commands) == 1
    assert not any("docker pull" in command for command in commands)
    assert result["manifest_digest"] == "sha256:digest"
    assert result["image_id"] == "sha256:image-id"
    assert result["attempts"][0]["reused"] is True
