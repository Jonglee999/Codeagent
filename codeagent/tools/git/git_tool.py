"""GitTool — 执行 Git 操作（status, diff, log, commit, branch 等）。

安全策略：
- 只读操作（status, diff, log, show, branch）：✅ 允许
- 可写操作（add, commit, checkout, stash）：⚠️ 允许但有安全限制
- 禁止操作（push, reset --hard, rebase, merge, cherry-pick）：🚫 阻断
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from codeagent.tools.base import BaseTool, ToolResult

# 允许的操作（action enum 值）
_ALLOWED_ACTIONS = frozenset({
    "status", "diff", "log", "show", "branch",
    "add", "commit", "checkout", "stash",
})

# 明确禁止的操作（即使通过 args 也无法执行）
_BLOCKED_ACTIONS = frozenset({
    "push", "reset", "rebase", "merge", "cherry-pick",
})

# 只读操作（不会修改 git 状态）
_READONLY_ACTIONS = frozenset({"status", "diff", "log", "show", "branch"})


class GitTool(BaseTool):
    """执行 Git 操作（status, diff, log, commit, branch 等）。"""

    name = "git"
    category = "version_control"
    risk_level = "medium"
    latency_hint = "medium"
    idempotent = False
    description = "执行 Git 操作（status, diff, log, commit, branch 等）"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ALLOWED_ACTIONS),
                "description": "Git 操作类型",
            },
            "args": {
                "type": "string",
                "default": "",
                "description": "操作参数，如分支名、文件路径、commit message 等",
            },
        },
        "required": ["action"],
    }

    def __init__(self, project_root: str | Path = ".") -> None:
        """初始化 GitTool。

        Args:
            project_root: 项目根目录路径。
        """
        super().__init__()
        self._project_root = Path(project_root).resolve()

    async def execute(  # type: ignore[override]
        self,
        action: str,
        args: str = "",
    ) -> ToolResult:
        """执行 Git 操作。

        Args:
            action: Git 操作类型（status/diff/log/show/branch/add/commit/checkout/stash）。
            args: 操作参数字符串。

        Returns:
            ToolResult: 操作结果。
        """
        start_time = time.monotonic()

        # ── 1. 操作合法性校验 ──────────────────────────────
        # 先检查禁止操作，再检查允许操作
        if action in _BLOCKED_ACTIONS:
            return ToolResult(
                success=False,
                error_message=f"Action '{action}' is blocked for safety. Please run it manually.",
                error_code="BLOCKED_ACTION",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        if action not in _ALLOWED_ACTIONS:
            return ToolResult(
                success=False,
                error_message=(
                    f"Invalid action '{action}'. "
                    f"Allowed actions: {', '.join(sorted(_ALLOWED_ACTIONS))}"
                ),
                error_code="INVALID_ACTION",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 2. Git 仓库校验 ────────────────────────────────
        git_dir_check = self._check_git_dir(start_time)
        if not git_dir_check.success:
            return git_dir_check

        # ── 3. 安全检查 ────────────────────────────────────
        safety_check = self._check_safety(action, args)
        if not safety_check.success:
            return safety_check

        # ── 4. 构建并执行 git 命令 ──────────────────────────
        cmd = self._build_command(action, args)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error_message="Git not found. Please install Git first.",
                error_code="GIT_NOT_FOUND",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        is_readonly = action in _READONLY_ACTIONS

        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error_message=stderr_str.strip() or f"git {action} failed",
                error_code="GIT_ERROR",
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": proc.returncode,
                    "action": action,
                    "args": args,
                    "readonly": is_readonly,
                },
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        return ToolResult(
            success=True,
            data={
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": 0,
                "action": action,
                "args": args,
                "readonly": is_readonly,
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    # ── 安全检查 ───────────────────────────────────────────────────────────

    def _check_git_dir(self, start_time: float) -> ToolResult:
        """检查项目根目录是否在 Git 仓库中。"""
        git_dir = self._project_root / ".git"
        if not git_dir.exists():
            # 也检查 git worktree 等情况：执行 git rev-parse --git-dir
            # 这里用简单的 .git 目录检查
            return ToolResult(
                success=False,
                error_message="Not a git repository (or no .git directory found)",
                error_code="NOT_A_GIT_REPO",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )
        return ToolResult(
            success=True,
            data={},
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    def _check_safety(self, action: str, args: str) -> ToolResult:
        """执行安全检查。

        Returns:
            ToolResult: success=True 表示安全，否则返回错误。
        """
        # checkout 保护：拒绝 git checkout -- .（批量丢弃所有变更）
        if action == "checkout":
            stripped = args.strip()
            if stripped == "-- ." or stripped == "--.":
                return ToolResult(
                    success=False,
                    error_message=(
                        "Mass discard detected: 'checkout -- .' is blocked. "
                        "Use specific file paths instead."
                    ),
                    error_code="CHECKOUT_DOT_BLOCKED",
                    duration_ms=0,
                )

        # commit 必须有 message
        if action == "commit" and not args.strip():
            return ToolResult(
                success=False,
                error_message=(
                    "Commit message required. Provide message via args, "
                    "e.g. args='-m \"your message\"'"
                ),
                error_code="COMMIT_MESSAGE_REQUIRED",
                duration_ms=0,
            )

        return ToolResult(success=True, data={}, duration_ms=0)

    def _build_command(self, action: str, args: str) -> list[str]:
        """构建 git 命令列表。

        支持两种模式：
        1. 简单参数：args="--oneline -5" → ["git", "log", "--oneline", "-5"]
        2. 带引用参数：args='-m "fix: bug"' → ["git", "commit", "-m", "fix: bug"]
        """
        cmd = ["git", action]
        if args.strip():
            parsed = _split_args(args.strip())
            cmd.extend(parsed)
        return cmd


def _split_args(args_str: str) -> list[str]:
    """拆分参数字符串，支持引号包裹的参数。

    Args:
        args_str: 参数字符串，如 '-m "fix: bug" --oneline'

    Returns:
        拆分后的参数列表。
    """
    result: list[str] = []
    current: list[str] = []
    in_quote: str | None = None  # " 或 '

    i = 0
    while i < len(args_str):
        ch = args_str[i]

        if in_quote is not None:
            if ch == in_quote:
                in_quote = None
            else:
                current.append(ch)
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch == " ":
            if current:
                result.append("".join(current))
                current = []
        else:
            current.append(ch)

        i += 1

    if current:
        result.append("".join(current))

    return result
