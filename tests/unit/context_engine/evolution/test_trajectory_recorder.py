"""TrajectoryRecorder 单元测试。

覆盖：
- start_task 创建目录和文件
- record_step 追加写入 JSONL
- complete_task 更新状态并刷盘
- load_recent 按时间排序和过滤
- 记录延迟 < 5ms（time.perf_counter 测量）
- 文件格式合法性（每行 JSON.parse 成功）
- 缓冲区自动刷盘（超过 10 步）
- get_trajectory 按 ID 检索
- 错误处理（不存在的任务）
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codeagent.context_engine.evolution import (
    Trajectory,
    TrajectoryRecorder,
    TrajectoryStep,
)


@pytest.fixture
def recorder(tmp_path: Path) -> TrajectoryRecorder:
    """使用临时目录创建 TrajectoryRecorder。"""
    base_path = tmp_path / ".codeagent" / "trajectories"
    return TrajectoryRecorder(base_path=str(base_path))


@pytest.fixture
def sample_step() -> TrajectoryStep:
    """创建一个示例轨迹步骤。"""
    return TrajectoryStep(
        step_id=str(uuid.uuid4()),
        node_name="planning",
        step_type="llm_call",
        timestamp=datetime.now(),
        input_summary='{"current_step": 0, "plan_steps": 3}',
        output_summary='{"plan": "[3 steps]"}',
        duration_ms=150,
        success=True,
    )


class TestTrajectoryStep:
    """TrajectoryStep dataclass 测试。"""

    def test_minimal_creation(self) -> None:
        step = TrajectoryStep(
            step_id="test-1",
            node_name="execution",
            step_type="tool_call",
            timestamp=datetime.now(),
            input_summary="input",
            output_summary="output",
        )
        assert step.step_id == "test-1"
        assert step.node_name == "execution"
        assert step.step_type == "tool_call"
        assert step.tool_calls == []
        assert step.duration_ms == 0
        assert step.success is True
        assert step.error is None

    def test_full_creation(self) -> None:
        now = datetime.now()
        step = TrajectoryStep(
            step_id="test-2",
            node_name="validation",
            step_type="validation",
            timestamp=now,
            input_summary='{"changes_count": 2}',
            output_summary='{"passed": true}',
            tool_calls=[{"tool": "run_syntax_check", "result": "ok"}],
            duration_ms=300,
            success=True,
        )
        assert step.step_id == "test-2"
        assert len(step.tool_calls) == 1
        assert step.duration_ms == 300

    def test_failure_step(self) -> None:
        step = TrajectoryStep(
            step_id="test-3",
            node_name="execution",
            step_type="tool_call",
            timestamp=datetime.now(),
            input_summary="input",
            output_summary="output",
            success=False,
            error="Tool execution failed",
        )
        assert step.success is False
        assert step.error == "Tool execution failed"

    def test_step_types(self) -> None:
        for step_type in ("llm_call", "tool_call", "human_review", "validation"):
            step = TrajectoryStep(
                step_id="t",
                node_name="test",
                step_type=step_type,  # type: ignore[arg-type]
                timestamp=datetime.now(),
                input_summary="in",
                output_summary="out",
            )
            assert step.step_type == step_type


class TestTrajectory:
    """Trajectory dataclass 测试。"""

    def test_minimal_creation(self) -> None:
        traj = Trajectory(
            task_id="task-1",
            user_request="Fix bug",
            started_at=datetime.now(),
        )
        assert traj.task_id == "task-1"
        assert traj.user_request == "Fix bug"
        assert traj.steps == []
        assert traj.final_status is None
        assert traj.repair_rounds == 0
        assert traj.validation_passed is False

    def test_full_creation(self) -> None:
        now = datetime.now()
        step = TrajectoryStep(
            step_id="s1", node_name="planning", step_type="llm_call",
            timestamp=now, input_summary="in", output_summary="out",
        )
        traj = Trajectory(
            task_id="task-2",
            user_request="Add route",
            started_at=now,
            completed_at=now + timedelta(seconds=10),
            steps=[step],
            final_status="success",
            repair_rounds=1,
            validation_passed=True,
        )
        assert traj.final_status == "success"
        assert traj.repair_rounds == 1
        assert traj.validation_passed is True
        assert len(traj.steps) == 1


class TestTrajectoryRecorderStartTask:
    """start_task 方法测试。"""

    def test_start_task_creates_directory(self, recorder: TrajectoryRecorder) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test request")
        date_dir = recorder._date_dir()
        assert date_dir.exists()
        assert date_dir.is_dir()

    def test_start_task_initializes_trajectory(self, recorder: TrajectoryRecorder) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test request")
        assert task_id in recorder._trajectories
        traj = recorder._trajectories[task_id]
        assert traj.task_id == task_id
        assert traj.user_request == "Test request"
        assert traj.started_at is not None
        assert traj.steps == []

    def test_start_task_initializes_buffer(self, recorder: TrajectoryRecorder) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test request")
        assert task_id in recorder._buffers
        assert recorder._buffers[task_id] == []

    def test_multiple_tasks_independent(self, recorder: TrajectoryRecorder) -> None:
        task_a = str(uuid.uuid4())
        task_b = str(uuid.uuid4())
        recorder.start_task(task_a, "Task A")
        recorder.start_task(task_b, "Task B")
        assert recorder._trajectories[task_a].user_request == "Task A"
        assert recorder._trajectories[task_b].user_request == "Task B"


class TestTrajectoryRecorderRecordStep:
    """record_step 方法测试。"""

    def test_record_step_appends_to_trajectory(
        self, recorder: TrajectoryRecorder, sample_step: TrajectoryStep,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.record_step(task_id, sample_step)
        assert len(recorder._trajectories[task_id].steps) == 1
        assert recorder._trajectories[task_id].steps[0].step_id == sample_step.step_id

    def test_record_step_buffers_lines(
        self, recorder: TrajectoryRecorder, sample_step: TrajectoryStep,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.record_step(task_id, sample_step)
        assert len(recorder._buffers[task_id]) == 1

    def test_record_step_buffer_line_is_valid_json(
        self, recorder: TrajectoryRecorder, sample_step: TrajectoryStep,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.record_step(task_id, sample_step)
        line = recorder._buffers[task_id][0]
        parsed = json.loads(line)
        assert parsed["step_id"] == sample_step.step_id
        assert parsed["node_name"] == sample_step.node_name
        assert parsed["step_type"] == sample_step.step_type
        assert parsed["success"] is True

    def test_record_step_unknown_task(
        self, recorder: TrajectoryRecorder, sample_step: TrajectoryStep,
    ) -> None:
        """记录不存在的任务应不抛异常。"""
        recorder.record_step("nonexistent", sample_step)
        # Should not raise

    def test_multiple_steps(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        steps_count = 5
        for i in range(steps_count):
            step = TrajectoryStep(
                step_id=f"s{i}", node_name="execution", step_type="tool_call",
                timestamp=datetime.now(), input_summary=f"in{i}", output_summary=f"out{i}",
            )
            recorder.record_step(task_id, step)
        assert len(recorder._trajectories[task_id].steps) == steps_count
        assert len(recorder._buffers[task_id]) == steps_count


class TestTrajectoryRecorderCompleteTask:
    """complete_task 方法测试。"""

    def test_complete_task_updates_status(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.complete_task(task_id, "success", repair_rounds=2, validation_passed=True)
        traj = recorder._trajectories[task_id]
        assert traj.final_status == "success"
        assert traj.repair_rounds == 2
        assert traj.validation_passed is True
        assert traj.completed_at is not None

    def test_complete_task_writes_file(
        self, recorder: TrajectoryRecorder, sample_step: TrajectoryStep,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.record_step(task_id, sample_step)
        recorder.complete_task(task_id, "success")
        # File should exist on disk
        file_path = recorder._file_path(task_id)
        assert file_path.exists()
        # File should contain the step
        content = file_path.read_text(encoding="utf-8")
        assert sample_step.step_id in content

    def test_complete_task_flushes_buffer(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        step = TrajectoryStep(
            step_id="s1", node_name="planning", step_type="llm_call",
            timestamp=datetime.now(), input_summary="in", output_summary="out",
        )
        recorder.record_step(task_id, step)
        recorder.complete_task(task_id, "failure", repair_rounds=3)
        # Buffer should be empty after flush
        assert task_id not in recorder._buffers or recorder._buffers[task_id] == []

    def test_complete_task_unknown_task(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        """完成不存在的任务应不抛异常。"""
        recorder.complete_task("nonexistent", "success")
        # Should not raise

    def test_complete_task_file_format(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        """验证写入的 JSONL 文件格式合法。"""
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        for i in range(3):
            step = TrajectoryStep(
                step_id=f"s{i}", node_name="execution", step_type="tool_call",
                timestamp=datetime.now(), input_summary=f"in{i}", output_summary=f"out{i}",
            )
            recorder.record_step(task_id, step)
        recorder.complete_task(task_id, "success")

        file_path = recorder._file_path(task_id)
        lines = file_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)  # should not raise
            assert "step_id" in parsed
            assert "node_name" in parsed
            assert "timestamp" in parsed


class TestTrajectoryRecorderFlush:
    """缓冲区刷盘测试。"""

    def test_auto_flush_after_10_steps(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        """超过 10 步应自动刷盘（在第 11 步时触发）。"""
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")

        # 先记录 10 步，缓冲区应有 10 条（尚未触发阈值）
        for i in range(10):
            step = TrajectoryStep(
                step_id=f"s{i}", node_name="execution", step_type="tool_call",
                timestamp=datetime.now(), input_summary=f"in{i}", output_summary=f"out{i}",
            )
            recorder.record_step(task_id, step)

        assert len(recorder._buffers.get(task_id, [])) == 10

        # 第 11 步触发刷盘（无事件循环时降级为同步）
        step = TrajectoryStep(
            step_id="s10", node_name="execution", step_type="tool_call",
            timestamp=datetime.now(), input_summary="in10", output_summary="out10",
        )
        recorder.record_step(task_id, step)

        # 刷盘后文件应已创建，缓冲区为空
        file_path = recorder._file_path(task_id)
        assert file_path.exists()
        assert len(recorder._buffers.get(task_id, [])) < 11

    def test_flush_sync_writes_file(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        step = TrajectoryStep(
            step_id="s1", node_name="planning", step_type="llm_call",
            timestamp=datetime.now(), input_summary="in", output_summary="out",
        )
        recorder.record_step(task_id, step)
        recorder._flush_sync(task_id)
        file_path = recorder._file_path(task_id)
        assert file_path.exists()
        content = file_path.read_text(encoding="utf-8")
        assert "s1" in content


class TestTrajectoryRecorderLoadRecent:
    """load_recent 方法测试。"""

    def test_load_recent_empty_when_no_data(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        results = recorder.load_recent(n=10)
        assert results == []

    def test_load_recent_returns_trajectories(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test request")
        step = TrajectoryStep(
            step_id="s1", node_name="planning", step_type="llm_call",
            timestamp=datetime.now(), input_summary="in", output_summary="out",
        )
        recorder.record_step(task_id, step)
        recorder.complete_task(task_id, "success")

        results = recorder.load_recent(n=10)
        assert len(results) >= 1
        assert any(t.task_id == task_id for t in results)
        loaded = next(t for t in results if t.task_id == task_id)
        assert loaded.user_request == "Test request"
        assert loaded.final_status == "success"

    def test_load_recent_success_only(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_ok = str(uuid.uuid4())
        task_fail = str(uuid.uuid4())
        recorder.start_task(task_ok, "OK")
        recorder.complete_task(task_ok, "success")
        recorder.start_task(task_fail, "Fail")
        recorder.complete_task(task_fail, "failure")

        results = recorder.load_recent(n=10, success_only=True)
        task_ids = [t.task_id for t in results]
        assert task_ok in task_ids
        assert task_fail not in task_ids

    def test_load_recent_respects_n(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        for i in range(5):
            tid = f"load-test-{i}"
            recorder.start_task(tid, f"Task {i}")
            recorder.complete_task(tid, "success")

        results = recorder.load_recent(n=3)
        assert len(results) <= 3


class TestTrajectoryRecorderGetTrajectory:
    """get_trajectory 方法测试。"""

    def test_get_trajectory_returns_none_for_missing(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        result = recorder.get_trajectory("nonexistent")
        assert result is None

    def test_get_trajectory_returns_cached(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        result = recorder.get_trajectory(task_id)
        assert result is not None
        assert result.task_id == task_id
        assert result.user_request == "Test"

    def test_get_trajectory_from_file(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "File test")
        step = TrajectoryStep(
            step_id="s1", node_name="execution", step_type="tool_call",
            timestamp=datetime.now(), input_summary="in", output_summary="out",
        )
        recorder.record_step(task_id, step)
        recorder.complete_task(task_id, "success")

        # 从内存移除，模拟从文件加载
        recorder._trajectories.pop(task_id, None)
        recorder._buffers.pop(task_id, None)

        result = recorder.get_trajectory(task_id)
        assert result is not None
        assert result.task_id == task_id


class TestTrajectoryRecorderPerformance:
    """性能测试。"""

    def test_record_step_under_5ms(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Perf test")
        step = TrajectoryStep(
            step_id="perf-1", node_name="planning", step_type="llm_call",
            timestamp=datetime.now(), input_summary="in", output_summary="out",
        )

        # 多次测量取平均值
        durations: list[float] = []
        for _ in range(10):
            s = TrajectoryStep(
                step_id=str(uuid.uuid4()), node_name="execution", step_type="tool_call",
                timestamp=datetime.now(), input_summary="in", output_summary="out",
            )
            start = time.perf_counter()
            recorder.record_step(task_id, s)
            elapsed = (time.perf_counter() - start) * 1000
            durations.append(elapsed)

        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        assert avg_duration < 5.0, f"Average record time {avg_duration:.3f}ms >= 5ms"
        assert max_duration < 10.0, f"Max record time {max_duration:.3f}ms >= 10ms"


class TestTrajectoryRecorderEdgeCases:
    """边界情况测试。"""

    def test_base_path_auto_creation(self, tmp_path: Path) -> None:
        """不存在的目录应自动创建。"""
        base = tmp_path / "new" / "deep" / "path" / "trajectories"
        rec = TrajectoryRecorder(base_path=str(base))
        task_id = str(uuid.uuid4())
        rec.start_task(task_id, "Auto create dir test")
        assert base.exists()
        rec.complete_task(task_id, "success")

    def test_special_characters_in_request(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "测试中文 & <special> chars! @#$%")
        step = TrajectoryStep(
            step_id="s1", node_name="planning", step_type="llm_call",
            timestamp=datetime.now(), input_summary="中文摘要",
            output_summary="<output> with 特殊 chars",
        )
        recorder.record_step(task_id, step)
        recorder.complete_task(task_id, "success")

        # 验证 step 数据正确写入 JSONL
        jsonl_path = recorder._file_path(task_id)
        content = jsonl_path.read_text(encoding="utf-8")
        assert "中文摘要" in content
        assert "<output> with 特殊 chars" in content

        # 验证元数据文件包含用户请求
        meta_path = recorder._metadata_path(task_id)
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "测试中文" in meta["user_request"]
        assert "<special>" in meta["user_request"]

    def test_repeated_complete_task_no_error(
        self, recorder: TrajectoryRecorder,
    ) -> None:
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.complete_task(task_id, "success")
        # 二次 complete 不应抛异常
        recorder.complete_task(task_id, "failure")
        assert recorder._trajectories[task_id].final_status == "failure"

    def test_date_dir_structure(self, recorder: TrajectoryRecorder) -> None:
        """验证目录结构包含 YYYY-MM 子目录。"""
        task_id = str(uuid.uuid4())
        recorder.start_task(task_id, "Test")
        recorder.complete_task(task_id, "success")
        date_dir = recorder._date_dir()
        assert date_dir.name == datetime.now().strftime("%Y-%m")
        assert date_dir.parent == recorder._base_path


class TestTrajectoryRecorderGraphIntegration:
    """TrajectoryRecorder 与 graph.py 集成相关测试。"""

    def test_wrap_node_with_recording_success(self) -> None:
        """验证包装后的节点正常执行并记录轨迹。"""
        from codeagent.context_engine.evolution import TrajectoryRecorder
        from codeagent.orchestration.graph import _wrap_node_with_recording
        from codeagent.orchestration.state import AgentState

        import tempfile
        import uuid as uid

        with tempfile.TemporaryDirectory() as tmp:
            rec = TrajectoryRecorder(base_path=str(Path(tmp) / "trajs"))
            task_id = str(uid.uuid4())
            state = AgentState(
                user_request="test",
                project_root="/tmp",
                task_id=task_id,
                evolution_enabled=True,
            )
            rec.start_task(task_id, "test")

            async def fake_node(s: AgentState) -> dict:
                return {"result": "ok"}

            wrapped = _wrap_node_with_recording("planning", fake_node, rec)

            import asyncio
            result = asyncio.run(wrapped(state))
            assert result["result"] == "ok"

    def test_wrap_node_with_recording_error(self) -> None:
        """验证包装后的节点在异常时记录失败轨迹。"""
        from codeagent.orchestration.graph import _wrap_node_with_recording
        from codeagent.orchestration.state import AgentState

        import tempfile
        import uuid as uid

        with tempfile.TemporaryDirectory() as tmp:
            rec = TrajectoryRecorder(base_path=str(Path(tmp) / "trajs"))
            task_id = str(uid.uuid4())
            state = AgentState(
                user_request="test",
                project_root="/tmp",
                task_id=task_id,
                evolution_enabled=True,
            )
            rec.start_task(task_id, "test")

            async def failing_node(s: AgentState) -> dict:
                raise ValueError("test error")

            wrapped = _wrap_node_with_recording("execution", failing_node, rec)

            import asyncio
            with pytest.raises(ValueError, match="test error"):
                asyncio.run(wrapped(state))
