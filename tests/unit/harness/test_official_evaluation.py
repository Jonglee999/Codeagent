from __future__ import annotations

import json
import os

import pytest

from codeagent.benchmarks.official import OfficialEvaluationStore, windows_to_wsl


def test_import_official_report_updates_latest_status(tmp_path) -> None:
    store = OfficialEvaluationStore(tmp_path)
    run_id = "official-1"
    model = "test/model"
    instance_id = "owner__repo-1"
    report = (
        store.paths.result_root
        / run_id
        / "logs"
        / "run_evaluation"
        / run_id
        / "test__model"
        / instance_id
        / "report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({instance_id: {"resolved": True}}), encoding="utf-8"
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")

    result = store.import_run(
        run_id,
        model_name=model,
        instance_ids=[instance_id],
        predictions_path=predictions,
    )

    assert result["instances"][instance_id]["resolved"] is True
    assert result["run"]["attempted"] == 1
    assert result["run"]["score_percent"] == 100.0
    assert store.status_for(instance_id)["run_id"] == run_id
    assert store.summary()["counts"]["resolved"] == 1


def test_missing_official_report_is_an_error_not_a_score(tmp_path) -> None:
    store = OfficialEvaluationStore(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")

    store.import_run(
        "official-error",
        model_name="test/model",
        instance_ids=["owner__repo-1"],
        predictions_path=predictions,
    )

    status = store.status_for("owner__repo-1")
    assert status["completed"] is False
    assert status["resolved"] is None
    assert store.summary()["counts"]["errors"] == 1


def test_gold_preflight_does_not_overwrite_prediction_score(tmp_path) -> None:
    store = OfficialEvaluationStore(tmp_path)
    run_id = "gold-check"
    instance_id = "owner__repo-1"
    report = (
        store.paths.result_root
        / run_id
        / "logs"
        / "run_evaluation"
        / run_id
        / "gold"
        / instance_id
        / "report.json"
    )
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps({instance_id: {"resolved": True}}), encoding="utf-8"
    )

    result = store.import_run(
        run_id,
        model_name="gold",
        instance_ids=[instance_id],
        predictions_path="gold",
    )

    assert result["run"]["kind"] == "gold_preflight"
    assert store.status_for(instance_id) is None
    assert store.summary()["counts"]["evaluated"] == 0
    assert store.summary()["latest_gold_by_instance"][instance_id]["resolved"] is True


def test_correction_run_preserves_raw_result_and_computes_aggregate(tmp_path) -> None:
    store = OfficialEvaluationStore(tmp_path)
    parent_id = "raw-15"
    model = "test/model"
    instance_ids = [f"owner__repo-{index}" for index in range(15)]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps({
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": "diff --git a/a.py b/a.py\n",
            })
            + "\n"
            for instance_id in instance_ids
        ),
        encoding="utf-8",
    )
    for index, instance_id in enumerate(instance_ids):
        report = (
            store.paths.result_root
            / parent_id
            / "logs"
            / "run_evaluation"
            / parent_id
            / "test__model"
            / instance_id
            / "report.json"
        )
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps({instance_id: {"resolved": index < 4}}),
            encoding="utf-8",
        )
    raw = store.import_run(
        parent_id,
        model_name=model,
        instance_ids=instance_ids,
        predictions_path=predictions,
    )

    corrected_instance = instance_ids[4]
    correction_id = "correction-1"
    correction_report = (
        store.paths.result_root
        / correction_id
        / "logs"
        / "run_evaluation"
        / correction_id
        / "test__model"
        / corrected_instance
        / "report.json"
    )
    correction_report.parent.mkdir(parents=True)
    correction_report.write_text(
        json.dumps({corrected_instance: {"resolved": True}}),
        encoding="utf-8",
    )
    correction = store.import_run(
        correction_id,
        model_name=model,
        instance_ids=[corrected_instance],
        predictions_path=predictions,
        parent_run_id=parent_id,
    )

    assert raw["run"]["resolved"] == 4
    assert store.summary()["runs"][0]["resolved"] == 4
    assert correction["run"]["corrected_aggregate"] == {
        "parent_run_id": parent_id,
        "correction_run_id": correction_id,
        "attempted": 15,
        "resolved": 5,
        "score_percent": 33.33,
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows-to-WSL translation is Windows-only")
def test_windows_path_translation_is_stable(tmp_path) -> None:
    translated = windows_to_wsl(tmp_path)
    assert translated.startswith("/mnt/")
    assert "\\" not in translated
