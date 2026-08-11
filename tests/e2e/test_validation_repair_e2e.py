"""E2E 测试：验证自动修复闭环（Phase 4.A + Phase 5.4 结构化上下文）。

Scene I — 语法错误自动修复
  Agent 收到含语法错误的文件 → 通过 syntax_check + 重试机制自动修复。

Scene J — 测试失败自动修复
  Agent 收到有 bug 的函数 + 对应测试 → 分析并修复代码使测试通过。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from codeagent.gateway.validation_gateway import IValidationGateway, ValidationResult
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.registry import ToolRegistry

# ── API Key 检测 ──────────────────────────────────────────────────────────

_HAS_API_KEY = bool(os.environ.get("LLM_API_KEY"))

requires_api_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="LLM_API_KEY environment variable not set — E2E test requires real LLM call",
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _build_tool_gateway(project_root: str) -> ToolGateway:
    """构建包含 ReadFileTool 和 WriteFileTool 的工具 Gateway。"""
    from codeagent.tools.file.read_file import ReadFileTool
    from codeagent.tools.file.write_file import WriteFileTool

    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root=project_root))
    registry.register(WriteFileTool(project_root=project_root))
    return ToolGateway(registry)


def _build_validation_gateway() -> IValidationGateway:
    """构建验证 Gateway（支持 syntax_check + 测试运行）。"""
    from codeagent.validation.syntax_validator import SyntaxValidator

    class _ValidationGateway(IValidationGateway):
        def __init__(self) -> None:
            self._validator = SyntaxValidator()

        async def run_syntax_check(self, file_path: str) -> ValidationResult:
            return await self._validator.check_file(file_path)

        async def run_lint(self, files: list[str]) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_tests(self, project_root: str) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_runtime_check(self, file_path: str) -> ValidationResult:
            return ValidationResult(passed=True)

    return _ValidationGateway()


def _build_llm(model_name: str | None = None) -> callable:
    """构建 LLM 调用函数（litellm），含网络重试。"""
    import asyncio

    import litellm

    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")
    timeout = int(os.environ.get("LLM_TIMEOUT", "120"))
    model_name or os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash")

    async def llm_call(**kwargs: object) -> object:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                call_kwargs: dict[str, object] = {
                    **{k: v for k, v in kwargs.items() if v is not None},
                    "timeout": timeout,
                }
                if api_key:
                    call_kwargs["api_key"] = api_key
                if api_base:
                    call_kwargs["api_base"] = api_base
                return await litellm.acompletion(**call_kwargs)
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    return llm_call


# ── Scene I: 语法错误自动修复 ──────────────────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestSceneI_SyntaxErrorAutoFix:
    """Scene I: 语法错误自动修复。

    场景：Agent 生成了含语法错误的代码 → ValidationNode 检测到 → LLM 重试修复。
    """

    @pytest.mark.asyncio
    async def test_syntax_error_auto_fix(self, tmp_path: Path) -> None:
        """Agent 自动修复 broken_syntax.py 中的语法错误。"""
        # 准备含语法错误的文件（缺少冒号）
        broken_file = tmp_path / "broken_syntax.py"
        broken_file.write_text(
            "def greet(name)\n"          # 缺少冒号
            "    return f'Hello, {name}'\n"
        )

        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm()

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request="修复 broken_syntax.py 中的语法错误",
            project_root=project_root,
        )

        result = await node(state)

        # 验证执行无 Agent 侧错误
        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证文件已被修复 → 语法正确
        fixed_content = broken_file.read_text(encoding="utf-8")
        try:
            compile(fixed_content, "broken_syntax.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"File still has syntax error after repair: {e}")

        # 验证修复尝试次数未超限
        assert result.get("retry_count", 0) <= 3, (
            f"Repair exceeded max retries: {result.get('retry_count')}"
        )

        # 验证执行日志中包含修复模式的记录（语法重试或 repair_mode）
        execution_log = result.get("execution_log", [])
        repair_entries = [
            e for e in execution_log
            if e.get("type") in ("repair_mode", "syntax_check")
        ]
        assert len(repair_entries) >= 1, (
            "Should have at least one syntax check or repair entry"
        )


# ── Scene J: 测试失败自动修复 ──────────────────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestSceneJ_TestFailureAutoFix:
    """Scene J: 测试失败自动修复。

    场景：Agent 修改代码导致测试失败 → 分析代码 → 修复使测试通过。
    """

    @pytest.mark.asyncio
    async def test_failing_test_auto_fix(self, tmp_path: Path) -> None:
        """Agent 修复 calculator.py 使测试通过。"""
        # 准备有 bug 的函数（a - b 应为 a + b）
        calc_file = tmp_path / "calculator.py"
        calc_file.write_text(
            "def add(a, b):\n"
            "    return a - b  # bug: should be a + b\n"
        )

        # 准备测试文件
        test_file = tmp_path / "test_calculator.py"
        test_file.write_text(
            "from calculator import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
            "    assert add(0, 0) == 0\n"
            "    assert add(-1, 1) == 0\n"
        )

        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm()

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request="修复 calculator.py 使测试通过",
            project_root=project_root,
        )

        result = await node(state)

        # 验证执行无 Agent 侧错误
        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证修复在 3 次以内完成
        assert result.get("retry_count", 0) <= 3, (
            f"Repair exceeded max retries: {result.get('retry_count')}"
        )

        # 运行 pytest 验证测试通过
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("pytest execution timed out")

        assert proc.returncode == 0, (
            f"Tests still failing after repair:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
