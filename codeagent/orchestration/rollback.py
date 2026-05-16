"""RollbackManager — 基于 accumulated_changes 的文件级回滚。

支持三种文件操作的回滚：
- create → 删除文件
- modify → 写回 original_content
- delete → 用 original_content 重建文件
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from codeagent.orchestration.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class RollbackResult:
    """回滚操作结果。

    Attributes:
        success: 是否全部成功
        restored_files: 已恢复的文件列表
        deleted_files: 已删除的文件列表
        errors: 错误信息列表
    """

    success: bool = True
    restored_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class RollbackManager:
    """文件级回滚管理器。

    基于 AgentState.accumulated_changes 中的操作记录执行回滚。
    每条 accumulated_changes 记录应包含：
        step_id: int          — 步骤编号
        action: str           — "create" | "modify" | "delete"
        file_path: str        — 文件路径
        original_content: str | None  — 操作前的原始内容（create 时为 None）
        timestamp: float      — 操作时间戳
    """

    def rollback_step(self, state: AgentState, step_id: int) -> RollbackResult:
        """回滚指定 step_id 的所有文件操作（按逆序）。

        Args:
            state: Agent 状态（内含 accumulated_changes）
            step_id: 要回滚的步骤编号

        Returns:
            RollbackResult: 回滚结果
        """
        changes = [
            c for c in state.accumulated_changes
            if c.get("step_id") == step_id
        ]
        if not changes:
            return RollbackResult(
                success=False,
                errors=[f"No changes found for step {step_id}"],
            )

        return self._rollback_changes(changes)

    def rollback_all(self, state: AgentState) -> RollbackResult:
        """回滚所有已执行步骤（按 step_id 逆序）。

        Args:
            state: Agent 状态

        Returns:
            RollbackResult: 回滚结果
        """
        if not state.accumulated_changes:
            return RollbackResult(
                success=False,
                errors=["No accumulated changes to roll back"],
            )

        # 按 step_id 分组后逆序（每个 step 内部保持操作顺序）
        step_groups: dict[int, list[dict]] = {}
        for c in state.accumulated_changes:
            sid = c.get("step_id", 0)
            step_groups.setdefault(sid, []).append(c)

        all_results = RollbackResult()
        for sid in sorted(step_groups.keys(), reverse=True):
            result = self._rollback_changes(step_groups[sid])
            all_results.restored_files.extend(result.restored_files)
            all_results.deleted_files.extend(result.deleted_files)
            all_results.errors.extend(result.errors)
            if not result.success:
                all_results.success = False

        return all_results

    def get_rollback_preview(
        self, state: AgentState, step_id: Optional[int] = None
    ) -> str:
        """返回回滚预览文本。

        Args:
            state: Agent 状态
            step_id: 指定步骤（None 时预览全部回滚）

        Returns:
            str: Markdown 格式的回滚预览
        """
        if step_id is not None:
            changes = [
                c for c in state.accumulated_changes
                if c.get("step_id") == step_id
            ]
            title = f"### 回滚预览 — Step {step_id}"
        else:
            changes = state.accumulated_changes
            title = "### 回滚预览 — 全部步骤"

        if not changes:
            return f"{title}\n\n无可回滚的操作。"

        lines = [title, ""]
        restore_count = 0
        delete_count = 0

        for c in reversed(changes):
            action = c.get("action", "unknown")
            file_path = c.get("file_path", "?")
            has_content = c.get("original_content") is not None

            if action == "create":
                lines.append(f"- 🗑️ 删除文件 `{file_path}`")
                delete_count += 1
            elif action == "modify":
                if has_content:
                    lines.append(f"- ♻️ 恢复文件 `{file_path}`")
                    restore_count += 1
                else:
                    lines.append(f"- ⚠️ 跳过 `{file_path}`（无原始内容）")
            elif action == "delete":
                if has_content:
                    lines.append(f"- ♻️ 重建文件 `{file_path}`")
                    restore_count += 1
                else:
                    lines.append(f"- ⚠️ 跳过 `{file_path}`（无原始内容）")
            else:
                lines.append(f"- ? 未知操作 `{action}`: {file_path}")

        lines.extend([
            "",
            f"**摘要：** 恢复 {restore_count} 个文件，删除 {delete_count} 个文件",
        ])

        return "\n".join(lines)

    # ── 内部方法 ──────────────────────────────────────────────

    def _rollback_changes(self, changes: list[dict]) -> RollbackResult:
        """按逆序执行一组文件操作的回滚。

        Args:
            changes: 操作记录列表（按正序，会逆序执行）

        Returns:
            RollbackResult: 回滚结果
        """
        result = RollbackResult()

        for c in reversed(changes):
            action = c.get("action", "")
            file_path = c.get("file_path", "")
            original_content = c.get("original_content")

            if not file_path:
                result.errors.append("Missing file_path in change record")
                result.success = False
                continue

            path = Path(file_path).resolve()

            try:
                if action == "create":
                    self._rollback_create(path, result)
                elif action == "modify":
                    self._rollback_modify(path, original_content, result)
                elif action == "delete":
                    self._rollback_delete(path, original_content, result)
                else:
                    result.errors.append(
                        f"Unknown action '{action}' for {file_path}"
                    )
                    result.success = False
            except Exception as e:
                logger.exception("Rollback failed for %s", file_path)
                result.errors.append(f"Rollback failed for {file_path}: {e}")
                result.success = False

        return result

    def _rollback_create(
        self, path: Path, result: RollbackResult
    ) -> None:
        """回滚 create 操作：删除文件。"""
        if path.exists():
            path.unlink()
            result.deleted_files.append(str(path))
            logger.info("Rollback create: deleted %s", path)
        else:
            logger.warning("Rollback create: file not found, skipping %s", path)

    def _rollback_modify(
        self, path: Path, original_content: Optional[str], result: RollbackResult
    ) -> None:
        """回滚 modify 操作：写回原始内容。"""
        if original_content is None:
            result.errors.append(
                f"Cannot rollback modify for {path}: no original content"
            )
            result.success = False
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(original_content, encoding="utf-8")
        result.restored_files.append(str(path))
        logger.info("Rollback modify: restored %s", path)

    def _rollback_delete(
        self, path: Path, original_content: Optional[str], result: RollbackResult
    ) -> None:
        """回滚 delete 操作：用 original_content 重建文件。"""
        if original_content is None:
            result.errors.append(
                f"Cannot rollback delete for {path}: no original content"
            )
            result.success = False
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(original_content, encoding="utf-8")
        result.restored_files.append(str(path))
        logger.info("Rollback delete: recreated %s", path)
