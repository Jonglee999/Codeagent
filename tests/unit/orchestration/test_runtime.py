from codeagent.gateway.validation_gateway import ValidationResult
import json
import subprocess

from codeagent.orchestration.runtime import (
    assess_task_completion,
    build_execution_report,
    build_learning_services,
    collect_git_patch,
    persist_run_artifacts,
)
from codeagent.orchestration.state import AgentState, PlanStep


def test_completion_requires_all_plan_steps_and_validation() -> None:
    state = AgentState(
        user_request="build it",
        project_root="/test",
        plan=[PlanStep(step_id=1, description="create", action="create")],
    )

    success, errors = assess_task_completion(state)

    assert success is False
    assert "executed 0 of 1" in errors[0]
    assert any("validation did not run" in error for error in errors)


def test_completion_accepts_completed_plan_with_passing_validation() -> None:
    state = AgentState(
        user_request="build it",
        project_root="/test",
        plan=[PlanStep(step_id=1, description="create", action="create")],
        current_step_index=1,
        validation_results=[ValidationResult(passed=True)],
    )

    success, errors = assess_task_completion(state)

    assert success is True
    assert errors == []


def test_benchmark_completion_requires_patch_and_test_output(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent Test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "base"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    state = AgentState(
        user_request="fix benchmark issue",
        project_root=str(tmp_path),
        validation_results=[ValidationResult(passed=True, output="1 passed")],
    )

    success, errors = assess_task_completion(
        state, require_patch=True, require_test_evidence=True
    )
    assert success is False
    assert any("no code patch" in error for error in errors)

    (tmp_path / "fix.py").write_text("fixed = True\n", encoding="utf-8")
    state.validation_results[-1].output = ""
    success, errors = assess_task_completion(
        state, require_patch=True, require_test_evidence=True
    )
    assert success is False
    assert any("no executed test output" in error for error in errors)

    state.validation_results[-1].output = "1 passed in 0.01s"
    success, errors = assess_task_completion(
        state, require_patch=True, require_test_evidence=True
    )
    assert success is True
    assert errors == []


def test_memory_capability_can_be_disabled_without_disabling_core_agent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("codeagent.orchestration.runtime.get_memory_enabled", lambda: False)
    monkeypatch.setattr("codeagent.orchestration.runtime.get_evolution_enabled", lambda: False)

    memory, evolution, strategy = build_learning_services(
        str(tmp_path), llm_client=object(), auto_mode=True
    )

    assert memory is None
    assert evolution is None
    assert strategy is None


def test_persist_run_artifacts_captures_tracked_and_untracked_patch(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True)
    tracked.write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("created = True\n", encoding="utf-8")
    report = {"task_id": "task-1", "validation_results": [{"output": "full stdout"}]}

    artifacts = persist_run_artifacts(str(tmp_path), "task-1", report)

    patch = (tmp_path / ".codeagent/artifacts/task-1/git.diff").read_text(encoding="utf-8")
    assert "value = 2" in patch
    assert "new.py" in patch
    validation = json.loads(
        (tmp_path / ".codeagent/artifacts/task-1/validation.json").read_text(encoding="utf-8")
    )
    assert validation[0]["output"] == "full stdout"
    assert {item["kind"] for item in artifacts} == {
        "git_diff",
        "validation_output",
        "report",
    }


def test_patch_for_nested_session_does_not_leak_parent_repository(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "agent@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Agent Test"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "parent-secret.txt").write_text("do-not-export\n", encoding="utf-8")
    subprocess.run(["git", "add", "parent-secret.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True)
    workspace = tmp_path / ".codeagent" / "workspaces" / "sessions" / "task-1"
    workspace.mkdir(parents=True)
    (workspace / "calculator.py").write_text("value = 2\n", encoding="utf-8")

    patch = collect_git_patch(str(workspace))

    assert "calculator.py" in patch
    assert "value = 2" in patch
    assert "parent-secret" not in patch
    assert "do-not-export" not in patch


def test_execution_report_has_same_complete_contract_for_all_runners() -> None:
    state = AgentState(
        user_request="build it",
        project_root="/test",
        validation_results=[ValidationResult(passed=True, output="1 passed")],
        memory_hits=[{"id": "memory-1"}],
        resolved_skills=[{"name": "python"}],
        recovered_from_task_id="task-source",
    )

    report = build_execution_report(
        "task-1",
        state,
        duration=1.5,
        success=True,
        errors=[],
        model_runtime={"active_model": "test/model"},
        infrastructure_runtime={"redis": {}},
    )

    assert report["assistant_response"]
    assert report["validation_results"][0]["output"] == "1 passed"
    assert report["memory_hits"] == [{"id": "memory-1"}]
    assert report["resolved_skills"] == [{"name": "python"}]
    assert report["error"] is None
    assert report["recovered_from_task_id"] == "task-source"


def test_benchmark_report_contains_no_cost_efficiency_diagnostics(monkeypatch) -> None:
    monkeypatch.setenv("MAX_TOKENS_PER_TASK", "100")
    state = AgentState(
        user_request="fix it",
        project_root="/test",
        benchmark_instance_id="owner__repo-1",
        llm_call_count=2,
        estimated_tokens=95,
        memory_mode="off",
        learning_mode="off",
        validation_state="validation_degraded",
        accumulated_changes=[{"file_path": "src/app.py"}],
        execution_log=[
            {
                "type": "tool_call",
                "tool_name": "read_file",
                "success": True,
                "duration_ms": 2,
            },
            {
                "type": "tool_call",
                "tool_name": "apply_patch",
                "success": False,
                "error_code": "INVALID_PARAMS",
                "duration_ms": 3,
            },
            {
                "type": "tool_call",
                "tool_name": "read_file",
                "success": False,
                "error_code": "CAPABILITY_NOT_EXPOSED",
                "duration_ms": 1,
            },
            {"type": "duplicate_tool_failure_breaker"},
        ],
    )

    report = build_execution_report(
        "task-benchmark",
        state,
        duration=1,
        success=False,
        errors=["stopped"],
        model_runtime={
            "attempt_count": 3,
            "prompt_tokens": 80,
            "completion_tokens": 15,
            "context_reduction_count": 1,
        },
        infrastructure_runtime={},
    )

    metrics = report["benchmark_metrics"]
    assert metrics["tool_calls_by_name"] == {"read_file": 2, "apply_patch": 1}
    assert metrics["invalid_tool_calls"] == 2
    assert metrics["executed_tool_calls"] == 2
    assert metrics["capability_violations"] == 1
    assert metrics["duplicate_failures"] == 1
    assert metrics["context_reductions"] == 1
    assert metrics["patch_produced"] is True
    assert metrics["memory_mode"] == "off"
    assert metrics["learning_mode"] == "off"
    assert any("token budget" in warning.lower() for warning in report["warnings"])
