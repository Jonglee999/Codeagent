import json

from codeagent.task_control import cancel_key, decode_steering_entries, steering_key


def test_task_control_keys_are_task_local() -> None:
    assert steering_key("run-123") == "task:run-123:steering"
    assert cancel_key("run-123") == "task:run-123:cancel"


def test_decode_steering_entries_accepts_bytes_and_strings() -> None:
    entries = [
        json.dumps({"instruction": "focus on tests"}).encode(),
        json.dumps({"instruction": "  preserve the API  "}),
    ]

    assert decode_steering_entries(entries) == [
        "focus on tests",
        "preserve the API",
    ]


def test_decode_steering_entries_ignores_invalid_and_bounds_content() -> None:
    entries = [
        b"not-json",
        json.dumps({"other": "missing"}),
        json.dumps({"instruction": "x" * 5000}),
    ]

    assert decode_steering_entries(entries) == ["x" * 4000]
