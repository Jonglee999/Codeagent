"""AgentState 单元测试。

覆盖：
- PlanStep 构建和字段验证
- AgentState 默认值
- Phase 1b 新增字段
- 状态不可变性约束
- backward compatibility（Phase 1a 字段仍可正常工作）
"""

from __future__ import annotations

from codeagent.gateway.validation_gateway import ValidationResult
from codeagent.orchestration.state import AgentState, PlanStep


class TestPlanStep:
    """PlanStep dataclass 构建测试。"""

    def test_basic_creation(self) -> None:
        step = PlanStep(step_id=1, description="Create file", action="create")
        assert step.step_id == 1
        assert step.description == "Create file"
        assert step.action == "create"
        assert step.target_file is None
        assert step.risk == "low"
        assert step.dependencies == []

    def test_full_creation(self) -> None:
        step = PlanStep(
            step_id=2,
            description="Delete old module",
            action="delete",
            target_file="src/old.py",
            risk="high",
            dependencies=[1],
        )
        assert step.step_id == 2
        assert step.target_file == "src/old.py"
        assert step.risk == "high"
        assert step.dependencies == [1]

    def test_all_actions(self) -> None:
        for action in ("create", "modify", "delete", "read", "command"):
            step = PlanStep(step_id=1, description="test", action=action)  # type: ignore[arg-type]
            assert step.action == action

    def test_target_file_optional(self) -> None:
        step = PlanStep(step_id=3, description="Run tests", action="command")
        assert step.target_file is None

    def test_dependencies_default_empty(self) -> None:
        step = PlanStep(step_id=4, description="Read file", action="read")
        assert step.dependencies == []

    def test_multiple_dependencies(self) -> None:
        step = PlanStep(
            step_id=5,
            description="Refactor module",
            action="modify",
            target_file="src/mod.py",
            dependencies=[1, 2, 3],
        )
        assert len(step.dependencies) == 3


class TestAgentStateDefaults:
    """AgentState 默认值测试。"""

    def test_minimal_creation(self) -> None:
        state = AgentState(user_request="test", project_root="/project")
        assert state.user_request == "test"
        assert state.project_root == "/project"

    def test_phase1a_fields_defaults(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.file_tree is None
        assert state.context == ""
        assert state.auto_mode is False
        assert state.execution_log == []
        assert state.errors == []

    def test_phase1b_fields_defaults(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.semantic_context is None
        assert state.current_file_context is None
        assert state.dependency_graph is None
        assert state.plan is None
        assert state.current_step_index == 0
        assert state.accumulated_changes == []
        assert state.validation_results == []
        assert state.human_decision is None
        assert state.retry_count == 0
        assert state.degraded_mode is False

    def test_auto_mode_true(self) -> None:
        state = AgentState(user_request="req", project_root="/root", auto_mode=True)
        assert state.auto_mode is True


class TestAgentStatePhase1bFields:
    """Phase 1b 新字段赋值测试。"""

    def test_semantic_context(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.semantic_context = "analyzed context"
        assert state.semantic_context == "analyzed context"

    def test_current_file_context(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.current_file_context = "file content"
        assert state.current_file_context == "file content"

    def test_dependency_graph(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.dependency_graph = {"src/main.py": ["src/utils.py"]}
        assert state.dependency_graph == {"src/main.py": ["src/utils.py"]}

    def test_plan_assignment(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        steps = [
            PlanStep(step_id=1, description="Step 1", action="read",
                     target_file="file1.py"),
            PlanStep(step_id=2, description="Step 2", action="modify",
                     target_file="file2.py", risk="high", dependencies=[1]),
        ]
        state.plan = steps
        assert state.plan is not None
        assert len(state.plan) == 2
        assert state.plan[0].step_id == 1
        assert state.plan[1].dependencies == [1]

    def test_current_step_index(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.current_step_index = 3
        assert state.current_step_index == 3

    def test_accumulated_changes(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        change = {"file": "main.py", "action": "modify", "lines_added": 5}
        state.accumulated_changes.append(change)
        assert len(state.accumulated_changes) == 1
        assert state.accumulated_changes[0]["file"] == "main.py"

    def test_validation_results(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        result = ValidationResult(passed=True)
        state.validation_results.append(result)
        assert len(state.validation_results) == 1
        assert state.validation_results[0].passed is True

    def test_human_decision(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.human_decision = "approve"
        assert state.human_decision == "approve"

    def test_retry_count_increment(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.retry_count += 1
        assert state.retry_count == 1
        state.retry_count += 1
        assert state.retry_count == 2

    def test_degraded_mode(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.degraded_mode is False
        state.degraded_mode = True
        assert state.degraded_mode is True


class TestAgentStateBackwardCompat:
    """Phase 1a 向后兼容性测试。

    确保现有的 Phase 1a 代码在升级后仍能正常工作。
    """

    def test_phase1a_construction_still_works(self) -> None:
        """Phase 1a 的构建方式仍然有效。"""
        state = AgentState(
            user_request="Create a hello.py file",
            project_root="/test/project",
            auto_mode=True,
        )
        assert state.user_request == "Create a hello.py file"
        assert state.project_root == "/test/project"
        assert state.auto_mode is True

    def test_phase35_fields_defaults(self) -> None:
        """Phase 3.5 TaskFocus 字段应有正确的默认值。"""
        state = AgentState(user_request="req", project_root="/root")
        assert state.original_goal_summary == ""
        assert state.completed_steps_summary == ""
        assert state.tasks_remaining == []
        assert state.deviation_detected is False
        assert state.deviation_count == 0
        assert state.conversation_history == []
        assert state.human_review_required is False
        assert state.review_request is None
        assert state.review_type is None

    def test_execution_log_still_works(self) -> None:
        """execution_log 仍可正常使用。"""
        state = AgentState(user_request="req", project_root="/root")
        state.execution_log.append({"type": "llm_response", "content": "Done"})
        assert len(state.execution_log) == 1

    def test_errors_still_works(self) -> None:
        """errors 列表仍可正常使用。"""
        state = AgentState(user_request="req", project_root="/root")
        state.errors.append("Something went wrong")
        assert len(state.errors) == 1

    def test_context_still_works(self) -> None:
        """context 字段仍可正常使用。"""
        state = AgentState(user_request="req", project_root="/root")
        state.context = "some context"
        assert state.context == "some context"

    def test_file_tree_still_works(self) -> None:
        """file_tree 字段仍可正常使用。"""
        state = AgentState(user_request="req", project_root="/root")
        state.file_tree = {"type": "directory", "name": "root", "children": []}
        assert state.file_tree["type"] == "directory"


class TestAgentStatePhase35Fields:
    """Phase 3.5 TaskFocus 字段赋值测试。"""

    def test_original_goal_summary(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.original_goal_summary = "Create a Flask app"
        assert state.original_goal_summary == "Create a Flask app"

    def test_completed_steps_summary(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.completed_steps_summary = "Read config; Write app"
        assert state.completed_steps_summary == "Read config; Write app"

    def test_tasks_remaining(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.tasks_remaining = ["[2] Modify main.py", "[3] Run tests"]
        assert len(state.tasks_remaining) == 2
        assert "[2] Modify main.py" in state.tasks_remaining

    def test_deviation_fields(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.deviation_detected is False
        assert state.deviation_count == 0
        state.deviation_detected = True
        state.deviation_count += 1
        assert state.deviation_detected is True
        assert state.deviation_count == 1

    def test_human_review_fields(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.human_review_required is False
        assert state.review_request is None
        assert state.review_type is None
        state.human_review_required = True
        state.review_request = {"review_type": "deviation_detected", "title": "test"}
        state.review_type = "deviation_detected"
        assert state.human_review_required is True
        assert state.review_request["review_type"] == "deviation_detected"
        assert state.review_type == "deviation_detected"

    def test_conversation_history(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.conversation_history.append({"role": "user", "content": "hello"})
        state.conversation_history.append({"role": "assistant", "content": "hi"})
        assert len(state.conversation_history) == 2


class TestAgentStatePhase7Fields:
    """Phase 7 自进化系统字段测试。"""

    def test_task_id_default(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.task_id == ""
        assert isinstance(state.task_id, str)

    def test_task_id_assignment(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.task_id = "550e8400-e29b-41d4-a716-446655440000"
        assert state.task_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_trajectory_steps_default(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.trajectory_steps == []
        assert isinstance(state.trajectory_steps, list)

    def test_trajectory_steps_append(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        step = {"node_name": "execution", "step_type": "tool_call", "tool_name": "read_file"}
        state.trajectory_steps.append(step)
        assert len(state.trajectory_steps) == 1
        assert state.trajectory_steps[0]["tool_name"] == "read_file"

    def test_repair_rounds_default(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.repair_rounds == 0

    def test_repair_rounds_increment(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.repair_rounds += 1
        assert state.repair_rounds == 1
        state.repair_rounds += 1
        assert state.repair_rounds == 2

    def test_applied_strategy_ids_default(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.applied_strategy_ids == []
        assert isinstance(state.applied_strategy_ids, list)

    def test_applied_strategy_ids_assignment(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.applied_strategy_ids = ["s1", "s2"]
        assert len(state.applied_strategy_ids) == 2
        assert "s1" in state.applied_strategy_ids

    def test_evolution_enabled_default(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        assert state.evolution_enabled is True

    def test_evolution_enabled_disable(self) -> None:
        state = AgentState(user_request="req", project_root="/root")
        state.evolution_enabled = False
        assert state.evolution_enabled is False


class TestAgentStateFullConstruction:
    """完整构造测试。"""

    def test_all_fields_in_constructor(self) -> None:
        state = AgentState(
            user_request="full test",
            project_root="/project",
            file_tree={"name": "root"},
            context="ctx",
            auto_mode=True,
            execution_log=[{"type": "test"}],
            errors=["err1"],
            semantic_context="semantic",
            current_file_context="current",
            dependency_graph={"key": "val"},
            current_step_index=1,
            accumulated_changes=[{"file": "a.py"}],
            degraded_mode=True,
            retry_count=2,
            original_goal_summary="Goal summary",
            completed_steps_summary="Step 1 done",
            tasks_remaining=["[2] Step 2"],
            deviation_detected=True,
            deviation_count=2,
            conversation_history=[{"role": "user", "content": "hi"}],
            human_review_required=True,
            review_request={"review_type": "deviation"},
            review_type="deviation_detected",
        )
        assert state.user_request == "full test"
        assert state.semantic_context == "semantic"
        assert state.current_step_index == 1
        assert state.degraded_mode is True
        assert state.retry_count == 2
        # Phase 3.5 assertions
        assert state.original_goal_summary == "Goal summary"
        assert state.completed_steps_summary == "Step 1 done"
        assert state.tasks_remaining == ["[2] Step 2"]
        assert state.deviation_detected is True
        assert state.deviation_count == 2
        assert state.conversation_history == [{"role": "user", "content": "hi"}]
        assert state.human_review_required is True
        assert state.review_request == {"review_type": "deviation"}
        assert state.review_type == "deviation_detected"
