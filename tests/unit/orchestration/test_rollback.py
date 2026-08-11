"""RollbackManager 单元测试。

覆盖：create/modify/delete 回滚、部分回滚、全部回滚、预览格式、文件不存在处理。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeagent.orchestration.rollback import RollbackManager
from codeagent.orchestration.state import AgentState


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def manager() -> RollbackManager:
    return RollbackManager()


@pytest.fixture
def state() -> AgentState:
    return AgentState(
        user_request="test",
        project_root="/tmp/test_project",
    )


# ── 单个操作回滚 ──────────────────────────────────────────────────────────


class TestRollbackSingleOperation:
    """单个操作类型的回滚测试。"""

    def test_rollback_create_deletes_file(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """create 操作回滚 = 删除文件。"""
        test_file = tmp_path / "new_file.py"
        test_file.write_text("# new content")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "create",
                    "file_path": str(test_file),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=1)

        assert result.success is True
        assert test_file.exists() is False
        assert str(test_file) in result.deleted_files

    def test_rollback_modify_restores_content(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """modify 操作回滚 = 恢复原始内容。"""
        test_file = tmp_path / "modified.py"
        test_file.write_text("# modified content")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "modify",
                    "file_path": str(test_file),
                    "original_content": "# original content",
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=1)

        assert result.success is True
        assert test_file.read_text() == "# original content"
        assert str(test_file) in result.restored_files

    def test_rollback_delete_rebuilds_file(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """delete 操作回滚 = 用 original_content 重建文件。"""
        test_file = tmp_path / "deleted.py"
        # 文件已被删除，不存在

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "delete",
                    "file_path": str(test_file),
                    "original_content": "# deleted content",
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=1)

        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == "# deleted content"
        assert str(test_file) in result.restored_files


# ── 部分回滚 ──────────────────────────────────────────────────────────────


class TestRollbackPartial:
    """指定 step_id 的部分回滚测试。"""

    def test_rollback_specific_step(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """只回滚指定步骤。"""
        file1 = tmp_path / "file1.py"
        file1.write_text("# file1 v2")
        file2 = tmp_path / "file2.py"
        file2.write_text("# file2 v2")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "modify",
                    "file_path": str(file1),
                    "original_content": "# file1 v1",
                    "timestamp": 1000.0,
                },
                {
                    "step_id": 2,
                    "action": "modify",
                    "file_path": str(file2),
                    "original_content": "# file2 v1",
                    "timestamp": 2000.0,
                },
            ],
        )

        # 只回滚 step 1
        result = manager.rollback_step(state, step_id=1)

        assert result.success is True
        assert file1.read_text() == "# file1 v1"  # 已恢复
        assert file2.read_text() == "# file2 v2"  # 未受影响
        assert str(file1) in result.restored_files
        assert str(file2) not in result.restored_files

    def test_rollback_nonexistent_step(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """回滚不存在的步骤返回错误。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "create",
                    "file_path": str(tmp_path / "test.py"),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=99)

        assert result.success is False
        assert len(result.errors) >= 1
        assert "No changes found" in result.errors[0]


# ── 全部回滚 ──────────────────────────────────────────────────────────────


class TestRollbackAll:
    """全部回滚测试。"""

    def test_rollback_all_steps(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """回滚所有步骤（按逆序）。"""
        file1 = tmp_path / "file1.py"
        file1.write_text("# file1 v2")
        file2 = tmp_path / "file2.py"
        file2.write_text("# file2 v2")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "modify",
                    "file_path": str(file1),
                    "original_content": "# file1 v1",
                    "timestamp": 1000.0,
                },
                {
                    "step_id": 2,
                    "action": "modify",
                    "file_path": str(file2),
                    "original_content": "# file2 v1",
                    "timestamp": 2000.0,
                },
            ],
        )

        result = manager.rollback_all(state)

        assert result.success is True
        assert file1.read_text() == "# file1 v1"
        assert file2.read_text() == "# file2 v1"
        assert len(result.restored_files) == 2

    def test_rollback_all_empty(self, manager: RollbackManager) -> None:
        """无变更时回滚全部返回错误。"""
        state = AgentState(
            user_request="test",
            project_root="/tmp/test",
        )

        result = manager.rollback_all(state)

        assert result.success is False
        assert "No accumulated changes" in result.errors[0]


# ── 错误处理 ──────────────────────────────────────────────────────────────


class TestRollbackErrorHandling:
    """回滚错误处理测试。"""

    def test_rollback_missing_original_content(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """modify 操作缺少 original_content 时返回错误。"""
        test_file = tmp_path / "test.py"
        test_file.write_text("# current content")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "modify",
                    "file_path": str(test_file),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=1)

        assert result.success is False
        assert "no original content" in result.errors[0].lower()

    def test_rollback_unknown_action(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """未知操作类型时返回错误。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "unknown_action",
                    "file_path": str(tmp_path / "test.py"),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
            ],
        )

        result = manager.rollback_step(state, step_id=1)

        assert result.success is False
        assert "unknown" in result.errors[0].lower()


# ── 回滚预览 ──────────────────────────────────────────────────────────────


class TestRollbackPreview:
    """get_rollback_preview 测试。"""

    def test_preview_format_all(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """全部回滚预览包含所有操作。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "create",
                    "file_path": str(tmp_path / "new.py"),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
                {
                    "step_id": 1,
                    "action": "modify",
                    "file_path": str(tmp_path / "mod.py"),
                    "original_content": "# old",
                    "timestamp": 1000.0,
                },
            ],
        )

        preview = manager.get_rollback_preview(state)

        assert "回滚预览" in preview
        assert "new.py" in preview
        assert "mod.py" in preview
        assert "摘要" in preview

    def test_preview_format_specific_step(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """指定步骤的回滚预览只包含该步骤的操作。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "create",
                    "file_path": str(tmp_path / "step1.py"),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
                {
                    "step_id": 2,
                    "action": "modify",
                    "file_path": str(tmp_path / "step2.py"),
                    "original_content": "# old",
                    "timestamp": 2000.0,
                },
            ],
        )

        preview = manager.get_rollback_preview(state, step_id=1)

        assert "Step 1" in preview
        assert "step1.py" in preview
        assert "step2.py" not in preview

    def test_preview_empty_changes(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """无变更时预览显示提示信息。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
        )

        preview = manager.get_rollback_preview(state)

        assert "无可回滚" in preview

    def test_preview_with_delete_action(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """delete 操作的回滚预览。"""
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "delete",
                    "file_path": str(tmp_path / "deleted.py"),
                    "original_content": "# deleted content",
                    "timestamp": 1000.0,
                },
            ],
        )

        preview = manager.get_rollback_preview(state)

        assert "重建" in preview
        assert "deleted.py" in preview


# ── 操作序列 ──────────────────────────────────────────────────────────────


class TestRollbackSequence:
    """多操作序列回滚测试。"""

    def test_create_then_modify_rollback(
        self, tmp_path: Path, manager: RollbackManager
    ) -> None:
        """先 create 后 modify，回滚时逆序操作。"""
        test_file = tmp_path / "sequence.py"
        test_file.write_text("# v2 content")

        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
            accumulated_changes=[
                {
                    "step_id": 1,
                    "action": "create",
                    "file_path": str(test_file),
                    "original_content": None,
                    "timestamp": 1000.0,
                },
                {
                    "step_id": 2,
                    "action": "modify",
                    "file_path": str(test_file),
                    "original_content": "# v1 content",
                    "timestamp": 2000.0,
                },
            ],
        )

        # 回滚全部（modify 先回滚，create 后回滚 → 删文件）
        result = manager.rollback_all(state)

        assert result.success is True
        # modify 恢复为 v1 → create 删除文件 → 文件不存在
        assert test_file.exists() is False
        assert len(result.restored_files) == 1  # modify 恢复
        assert len(result.deleted_files) == 1   # create 删除
