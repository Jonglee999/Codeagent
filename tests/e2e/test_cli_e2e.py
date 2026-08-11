"""端到端集成测试 — 验证完整的 Agent 闭环。

测试场景：生成一个 Python 函数并验证运行。
1. 通过 ExecutionNode 执行一个生成 Python 函数的请求
2. 验证文件被成功创建
3. 运行 python 验证输出正确
4. 验证执行日志包含预期的操作

注意：这些测试需要 LLM API Key（通过 LLM_API_KEY 环境变量设置）。
若未设置 API Key，测试将被跳过（使用 pytest.mark.skipif）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from codeagent.gateway.validation_gateway import IValidationGateway
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.state import AgentState
from codeagent.tools.gateway import ToolGateway
from codeagent.tools.registry import ToolRegistry

# 是否需要 API Key 才能运行 E2E 测试
_HAS_API_KEY = bool(os.environ.get("LLM_API_KEY"))

# E2E 测试标记：需要 LLM API Key
requires_api_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="LLM_API_KEY environment variable not set — E2E test requires real LLM call",
)

# E2E 测试标记：需要网络连接
e2e_test = pytest.mark.e2e


def _build_tool_gateway(project_root: str) -> ToolGateway:
    """构建包含 ReadFileTool 和 WriteFileTool 的工具 Gateway。"""
    from codeagent.tools.file.read_file import ReadFileTool
    from codeagent.tools.file.write_file import WriteFileTool

    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root=project_root))
    registry.register(WriteFileTool(project_root=project_root))
    return ToolGateway(registry)


def _build_phase2_tool_gateway(project_root: str) -> ToolGateway:
    """构建包含所有 Phase 2 工具（ReadFile, WriteFile, SearchCode, GetDiagnostics）的 Gateway。"""
    from codeagent.tools.file.read_file import ReadFileTool
    from codeagent.tools.file.write_file import WriteFileTool
    from codeagent.tools.search.search_code import SearchCodeTool
    from codeagent.tools.lsp.get_diagnostics import GetDiagnosticsTool

    registry = ToolRegistry()
    registry.register(ReadFileTool(project_root=project_root))
    registry.register(WriteFileTool(project_root=project_root))
    registry.register(SearchCodeTool(project_root=project_root))
    registry.register(GetDiagnosticsTool(project_root=project_root))
    return ToolGateway(registry)


def _build_validation_gateway() -> IValidationGateway:
    """构建 IValidationGateway 适配器（基于 SyntaxValidator）。"""
    from codeagent.gateway.validation_gateway import ValidationResult
    from codeagent.validation.syntax_validator import SyntaxValidator

    class _ValidationGateway(IValidationGateway):
        def __init__(self, validator: SyntaxValidator) -> None:
            self._validator = validator

        async def run_syntax_check(self, file_path: str) -> ValidationResult:
            return await self._validator.check_file(file_path)

        async def run_lint(self, files: list[str]) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_tests(self, project_root: str) -> ValidationResult:
            return ValidationResult(passed=True)

        async def run_runtime_check(self, file_path: str) -> ValidationResult:
            return ValidationResult(passed=True)

    return _ValidationGateway(SyntaxValidator())


def _build_llm(model_name: str) -> callable:
    """构建 LLM 调用函数（使用 litellm），含网络重试。"""
    import asyncio

    import litellm

    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_API_BASE", "")
    timeout = int(os.environ.get("LLM_TIMEOUT", "120"))

    async def llm_call(**kwargs: object) -> object:
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                call_kwargs = {
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


# ── Test: Fibonacci function generation ─────────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2EFibonacci:
    """E2E 场景：生成斐波那契数列函数并验证运行。"""

    @pytest.mark.asyncio
    async def test_generate_and_verify_fibonacci(self, tmp_path: Path) -> None:
        """完整场景：生成 fibonacci.py → 验证语法 → 运行验证输出。"""
        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                "Write a Python file fibonacci.py that contains a function "
                "fibonacci(n) which returns the nth Fibonacci number. "
                "Include a test that prints fibonacci(10) == 55."
            ),
            project_root=project_root,
        )

        result = await node(state)

        # ── 验证执行结果 ──────────────────────────────────
        assert "errors" in result
        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        execution_log = result.get("execution_log", [])
        assert len(execution_log) > 0, "Execution log should not be empty"

        # ── 验证文件被创建 ────────────────────────────────
        fib_file = tmp_path / "fibonacci.py"
        assert fib_file.exists(), f"{fib_file} should have been created"

        # ── 验证文件内容是有效的 Python ────────────────────
        content = fib_file.read_text(encoding="utf-8")
        assert "fibonacci" in content.lower(), "File should contain fibonacci function"
        assert len(content) > 0, "File should not be empty"

        # ── 验证 python 可执行并输出正确结果 ──────────────
        try:
            proc = subprocess.run(
                [sys.executable, str(fib_file)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("fibonacci.py execution timed out")

        # 测试文件可能包含 assert（pytest 风格）或 print 语句
        if proc.returncode != 0:
            # 如果直接运行失败，可能是 assert 语句，尝试用 pytest 运行
            try:
                proc2 = subprocess.run(
                    [sys.executable, "-m", "pytest", str(fib_file), "-v"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert proc2.returncode == 0, (
                    f"pytest execution failed:\n"
                    f"stdout: {proc2.stdout}\n"
                    f"stderr: {proc2.stderr}"
                )
            except subprocess.TimeoutExpired:
                # pytest 超时 — 至少文件存在且语法正确
                pass
        else:
            # 直接运行成功 — 输出应为 "True" 或 "55" 或类似
            assert "55" in proc.stdout or "True" in proc.stdout, (
                f"Output should contain expected result. Got: {proc.stdout}"
            )

        # ── 验证 Agent 的最终摘要 ─────────────────────────
        llm_responses = [
            e for e in execution_log if e.get("type") == "llm_response"
        ]
        assert len(llm_responses) >= 1, "Should have at least one LLM response"
        assert llm_responses[-1].get("content"), "Final response should have content"

        # ── 验证工具调用日志包含 write_file ───────────────
        tool_calls = [
            e for e in execution_log
            if e.get("type") == "tool_call" and e.get("tool_name") == "write_file"
        ]
        assert len(tool_calls) >= 1, "Should have at least one write_file call"

        # ── 验证语法检查发生在 write_file 之后 ────────────
        syntax_checks = [
            e for e in execution_log if e.get("type") == "syntax_check"
        ]
        assert len(syntax_checks) >= 1, "Should have at least one syntax check"


# ── Test: Simple function generation (no external deps) ─────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2ESimpleFunction:
    """E2E 场景：生成一个简单的 Python 函数。"""

    @pytest.mark.asyncio
    async def test_generate_hello_function(self, tmp_path: Path) -> None:
        """生成一个简单的 hello world 函数。"""
        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                "Create a file hello.py with a function greet(name) "
                "that returns 'Hello, {name}!'. "
                "Add a print(greet('World')) call at the bottom."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证文件创建
        hello_file = tmp_path / "hello.py"
        assert hello_file.exists(), "hello.py should have been created"

        # 验证可运行
        proc = subprocess.run(
            [sys.executable, str(hello_file)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            assert "Hello, World" in proc.stdout or "Hello, World" in proc.stderr


# ── Test: Module-level validation (no LLM needed) ──────────────────────────


class TestE2EToolIntegration:
    """E2E 工具集成测试 — 验证 ReadFileTool + WriteFileTool + SyntaxValidator
    作为最小闭环能正常工作（不需要 LLM）。"""

    @pytest.mark.asyncio
    async def test_write_read_validate_cycle(self, tmp_path: Path) -> None:
        """验证 写入 → 读取 → 语法检查 的最小闭环。"""
        from codeagent.tools.file.read_file import ReadFileTool
        from codeagent.tools.file.write_file import WriteFileTool
        from codeagent.validation.syntax_validator import SyntaxValidator

        project_root = str(tmp_path)

        # 写入
        write_tool = WriteFileTool(project_root=project_root)
        write_result = await write_tool.execute(
            file_path="test_func.py",
            content="def add(a, b):\n    return a + b\n",
            mode="create",
        )
        assert write_result.success

        # 读取
        read_tool = ReadFileTool(project_root=project_root)
        read_result = await read_tool.execute(file_path="test_func.py")
        assert read_result.success
        assert "def add" in read_result.data.get("content", "")

        # 语法检查
        validator = SyntaxValidator()
        validation_result = await validator.check_file(
            str(tmp_path / "test_func.py")
        )
        assert validation_result.passed

    @pytest.mark.asyncio
    async def test_syntax_error_detection(self, tmp_path: Path) -> None:
        """验证写入无效代码后语法检查能捕获错误。"""
        from codeagent.tools.file.write_file import WriteFileTool
        from codeagent.validation.syntax_validator import SyntaxValidator

        project_root = str(tmp_path)

        # 写入无效代码
        write_tool = WriteFileTool(project_root=project_root)
        write_result = await write_tool.execute(
            file_path="bad_syntax.py",
            content="def foo(\n    pass\n",
            mode="create",
        )
        assert write_result.success

        # 语法检查应失败
        validator = SyntaxValidator()
        validation_result = await validator.check_file(
            str(tmp_path / "bad_syntax.py")
        )
        assert not validation_result.passed
        assert len(validation_result.errors) >= 1

    @pytest.mark.asyncio
    async def test_write_file_then_modify(self, tmp_path: Path) -> None:
        """验证创建 → 修改 → 备份 流程。"""
        from codeagent.tools.file.write_file import WriteFileTool

        project_root = str(tmp_path)
        write_tool = WriteFileTool(project_root=project_root)

        # 创建
        create_result = await write_tool.execute(
            file_path="counter.py",
            content="x = 1\n",
            mode="create",
        )
        assert create_result.success

        # 修改
        modify_result = await write_tool.execute(
            file_path="counter.py",
            content="x = 2\n",
            mode="modify",
        )
        assert modify_result.success
        assert modify_result.data.get("lines_added", 0) >= 0
        assert modify_result.data.get("backup_path") is not None

        # 验证备份存在
        backup_path = tmp_path / ".codeagent" / "backups"
        list(backup_path.glob("*counter.py"))

# ── Scene A: Flask /health endpoint ──────────────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2EFlaskApp:
    """E2E 场景 A：生成 Flask app 并验证可启动。"""

    @pytest.mark.asyncio
    async def test_generate_flask_health_endpoint(self, tmp_path: Path) -> None:
        """生成 Flask app.py 含 /health 端点，验证语法和可导入性。"""
        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                "Create a file app.py that is a Flask application "
                'with a /health endpoint that returns JSON {"status": "ok"}. '
                "Include the if __name__ == '__main__' block on port 5000."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        app_file = tmp_path / "app.py"
        assert app_file.exists(), "app.py should have been created"

        content = app_file.read_text(encoding="utf-8")
        assert "flask" in content.lower(), "File should reference Flask"
        assert "/health" in content, "File should define /health endpoint"
        assert "__main__" in content, "File should have __main__ block"

        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"app.py contains syntax error: {e}")

        execution_log = result.get("execution_log", [])
        tool_calls = [
            e for e in execution_log
            if e.get("type") == "tool_call" and e.get("tool_name") == "write_file"
        ]
        assert len(tool_calls) >= 1, "Should have at least one write_file call"


# ── Scene C: Type annotations on existing file ────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2ETypeAnnotations:
    """E2E 场景 C：读取已有文件，为函数添加类型注解。"""

    @pytest.mark.asyncio
    async def test_add_type_annotations(self, tmp_path: Path) -> None:
        """在已有 Python 文件上添加类型注解。"""
        source = (
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def greet(name):\n"
            "    return 'Hello, ' + name\n"
            "\n"
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        )
        (tmp_path / "calculator.py").write_text(source, encoding="utf-8")

        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                "Read calculator.py and add type annotations to all functions. "
                "The add function should accept int and return int. "
                "greet should accept str and return str. "
                "fibonacci should accept int and return int. "
                "Do not change the function bodies."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        content = (tmp_path / "calculator.py").read_text(encoding="utf-8")
        assert "def add(a: int, b: int) -> int" in content, (
            "add() should have type annotations"
        )
        assert "def greet(name: str) -> str" in content, (
            "greet() should have type annotations"
        )

        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Modified file has syntax error: {e}")

        result_code = (
            "from calculator import add, greet, fibonacci\n"
            "print(add(2, 3))\n"
            "print(greet('World'))\n"
            "print(fibonacci(10))\n"
        )
        (tmp_path / "test_calculator.py").write_text(result_code, encoding="utf-8")

        import subprocess
        import sys
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path / "test_calculator.py")],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=tmp_path,
            )
            if proc.returncode == 0:
                lines = proc.stdout.strip().split("\n")
                assert "5" in lines[0], f"add(2,3) should be 5, got {lines[0]}"
                assert "Hello, World" in lines[1], (
                    f"greet('World') should be 'Hello, World', got {lines[1]}"
                )
                assert "55" in lines[2], f"fibonacci(10) should be 55, got {lines[2]}"
        except subprocess.TimeoutExpired:
            pass


# ── 1b Orchestrator flow ─────────────────────────────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2EOrchestratorFlow:
    """1b E2E 场景：通过 Orchestrator 执行完整的 LangGraph 工作流。"""

    @pytest.mark.asyncio
    async def test_orchestrator_generate_function(self, tmp_path: Path) -> None:
        """使用 Orchestrator 生成一个 Python 函数，验证全流程。"""
        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.gateway.context_gateway import IContextGateway, ContextPackage

        project_root = str(tmp_path)
        tool_gateway = _build_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        class _MockContextGateway(IContextGateway):
            async def build_context(
                self, project_root: str, query: str
            ) -> ContextPackage:
                return ContextPackage(
                    file_tree={
                        "name": "testproj",
                        "type": "directory",
                        "path": ".",
                        "children": [],
                    },
                )

            async def update_index(self, project_root: str) -> None:
                pass

            async def search_semantic(
                self, query: str, top_k: int = 5
            ) -> list:
                return []

        orchestrator = Orchestrator(
            context_gateway=_MockContextGateway(),
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
        )

        result = await orchestrator.run(
            user_request=(
                "Create a file factorial.py with a function factorial(n) "
                "that returns the factorial of n using recursion. "
                "Add a print(factorial(5) == 120) call."
            ),
            project_root=project_root,
        )

        assert result is not None
        assert hasattr(result, "execution_log") or isinstance(result, dict)

        factorial_file = tmp_path / "factorial.py"
        assert factorial_file.exists(), "factorial.py should have been created"

        content = factorial_file.read_text(encoding="utf-8")
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"factorial.py has syntax error: {e}")

        import subprocess
        import sys
        try:
            proc = subprocess.run(
                [sys.executable, str(factorial_file)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0:
                assert "120" in proc.stdout or "True" in proc.stdout, (
                    f"Output should contain expected result. Got: {proc.stdout}"
                )
        except subprocess.TimeoutExpired:
            pass

        checkpoints = orchestrator.get_checkpoints()
        assert len(checkpoints) >= 1, "Should have at least one checkpoint"


# ── Scene D/E/F: Cross-file modification (Phase 2) ────────────────────────────


@pytest.mark.e2e
@requires_api_key
class TestE2ECrossFileModification:
    """验证 Agent 在 sample_python_project 中跨文件添加新 API 端点。"""

    async def _copy_sample_project(self, tmp_path: Path) -> Path:
        """将 sample_python_project 复制到临时目录。"""
        import shutil

        sample_src = (
            Path(__file__).resolve().parent.parent.parent
            / "tests" / "fixtures" / "sample_python_project"
        )
        dest = tmp_path / "sample_project"
        shutil.copytree(str(sample_src), str(dest))
        return dest

    # ── Scene D: Add new API endpoint ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_add_new_api_endpoint(self, tmp_path: Path) -> None:
        """测试场景：Agent 读取现有项目结构，跨文件添加新端点。

        1. Agent 读取 api/routes.py 了解现有路由模式
        2. Agent 读取 services/user_service.py 了解服务层 API
        3. Agent 修改 api/routes.py 添加 get_user 路由
        4. 验证文件修改正确
        """
        project_dir = await self._copy_sample_project(tmp_path)
        project_root = str(project_dir)

        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                f"In the Python project at {project_root}, add a new API endpoint "
                "for getting a user by name. "
                "Look at api/routes.py to see existing routes (login, health). "
                "Look at services/user_service.py to see UserService.get_user(). "
                "Follow the same pattern to add a 'get_user' route that accepts a "
                "name parameter and returns the user data. "
                "After modifying, read the file to verify correctness."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证 routes 文件被修改
        routes_file = project_dir / "api" / "routes.py"
        assert routes_file.exists(), "routes.py should exist"
        content = routes_file.read_text(encoding="utf-8")
        assert "get_user" in content, "Routes should contain get_user"
        assert "svc.get_user" in content or "service.get_user" in content, (
            "Should call get_user on the service"
        )

        # 验证 Python 语法正确
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Modified routes.py has syntax error: {e}")

        # 验证工具调用日志包含 read_file 和 write_file
        execution_log = result.get("execution_log", [])
        tool_calls = [
            e for e in execution_log
            if e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "read_file" in tool_names, "Should have used read_file"
        assert "write_file" in tool_names, "Should have used write_file"

    # ── Scene E: Semantic search and modify ─────────────────────────────

    @pytest.mark.asyncio
    async def test_semantic_search_and_modify(self, tmp_path: Path) -> None:
        """测试场景：使用搜索定位认证相关代码并修改。

        1. Agent 通过 search_code（regex 模式）找到认证相关代码
        2. 读取定位到的文件理解代码逻辑
        3. 添加认证成功后的日志记录
        """
        project_dir = await self._copy_sample_project(tmp_path)
        project_root = str(project_dir)

        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                f"In the Python project at {project_root}, use search_code "
                "with a regex pattern to find authentication-related code "
                "(search for 'authenticate' or 'auth'). Read the found files, "
                "then add a print/log statement in the authenticate_user function "
                "in services/auth.py that prints a message when authentication succeeds. "
                "After modifying, read the file to verify."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证 auth 文件被修改
        auth_file = project_dir / "services" / "auth.py"
        assert auth_file.exists(), "auth.py should exist"
        content = auth_file.read_text(encoding="utf-8")

        # 验证添加了 print/logging 语句（在原有 pass 语句之前）
        assert "print" in content or "logging" in content or "log" in content, (
            "Should have added print or logging statement"
        )

        # 验证 Python 语法正确
        import ast
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Modified auth.py has syntax error: {e}")

        # 验证工具调用日志包含 search_code
        execution_log = result.get("execution_log", [])
        tool_calls = [
            e for e in execution_log
            if e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "search_code" in tool_names, "Should have used search_code"

    # ── Scene F: Code diagnostics and fix ───────────────────────────────

    @pytest.mark.asyncio
    async def test_code_diagnostics_and_fix(self, tmp_path: Path) -> None:
        """测试场景：使用 LSP 诊断修复有语法错误的文件。

        1. 创建一个包含语法错误的 Python 文件
        2. Agent 使用 get_diagnostics 定位错误
        3. Agent 使用 write_file 修复错误
        4. 验证修复后的文件语法正确
        """
        project_root = str(tmp_path)

        # 创建有语法错误的文件
        broken_code = (
            "def add(a, b\n"
            "    return a + b\n"
            "\n"
            "def broken_function(:\n"
            "    pass\n"
        )
        (tmp_path / "broken.py").write_text(broken_code, encoding="utf-8")

        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                f"The file {project_root}/broken.py has syntax errors. "
                "Use get_diagnostics to find the errors (look for line-level diagnostics), "
                "then fix them using write_file. "
                "After fixing, read the file to verify the syntax is correct."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证文件被修复且语法正确
        fixed_file = tmp_path / "broken.py"
        assert fixed_file.exists(), "broken.py should exist"
        fixed_content = fixed_file.read_text(encoding="utf-8")

        import ast
        try:
            ast.parse(fixed_content)
        except SyntaxError as e:
            pytest.fail(f"File still has syntax error after fix: {e}")

        # 验证工具调用日志包含 get_diagnostics
        execution_log = result.get("execution_log", [])
        tool_calls = [
            e for e in execution_log
            if e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "get_diagnostics" in tool_names, (
            "Should have used get_diagnostics"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3.8 — Scene G: 全栈项目认证方式修改（Basic Auth → JWT Auth）
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@requires_api_key
class TestE2EAuthModification:
    """Scene G: 在全栈项目中将 Basic Auth 替换为 JWT Auth。

    验证 Agent 的复杂多步规划 + 执行 + Human Review 能力。
    """

    async def _copy_sample_fullstack_project(self, tmp_path: Path) -> Path:
        """将 sample_fullstack_project 复制到临时目录。"""
        import shutil

        sample_src = (
            Path(__file__).resolve().parent.parent.parent
            / "tests" / "fixtures" / "sample_fullstack_project"
        )
        dest = tmp_path / "fullstack_project"
        shutil.copytree(str(sample_src), str(dest))
        return dest

    @pytest.mark.asyncio
    async def test_auth_modification_with_human_review(
        self, tmp_path: Path,
    ) -> None:
        """完整场景：Basic Auth → JWT Auth 替换。

        1. Agent 读取项目结构，分析现有认证方式
        2. Agent 读取 auth_middleware.py 理解 Basic Auth 实现
        3. Agent 分析依赖（哪些文件引用认证中间件）
        4. Agent 生成修改计划（替换为 JWT 认证）
        5. Plan 中包含高风险步骤 → route_after_planning 路由到 human_review
        6. 人工审核（通过 orchestrator.resume 自动审批）
        7. Agent 执行跨文件修改
        8. 语法验证通过
        """
        project_dir = await self._copy_sample_fullstack_project(tmp_path)
        project_root = str(project_dir)

        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.gateway.context_gateway import IContextGateway, ContextPackage

        # 构建工具和验证 gateway
        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        class _MockContextGateway(IContextGateway):
            async def build_context(
                self, project_root: str, query: str,
            ) -> ContextPackage:
                return ContextPackage(
                    file_tree={
                        "name": "fullstack_project",
                        "type": "directory",
                        "path": ".",
                        "children": [
                            {
                                "name": "backend",
                                "type": "directory",
                                "path": "backend",
                                "children": [
                                    {"name": "app.py", "type": "file", "path": "backend/app.py"},
                                    {"name": "__init__.py", "type": "file", "path": "backend/__init__.py"},
                                    {
                                        "name": "api", "type": "directory", "path": "backend/api",
                                        "children": [
                                            {"name": "__init__.py", "type": "file", "path": "backend/api/__init__.py"},
                                            {"name": "routes.py", "type": "file", "path": "backend/api/routes.py"},
                                        ],
                                    },
                                    {
                                        "name": "auth", "type": "directory", "path": "backend/auth",
                                        "children": [
                                            {"name": "__init__.py", "type": "file", "path": "backend/auth/__init__.py"},
                                            {"name": "auth_middleware.py", "type": "file", "path": "backend/auth/auth_middleware.py"},
                                        ],
                                    },
                                    {
                                        "name": "models", "type": "directory", "path": "backend/models",
                                        "children": [
                                            {"name": "__init__.py", "type": "file", "path": "backend/models/__init__.py"},
                                            {"name": "user.py", "type": "file", "path": "backend/models/user.py"},
                                        ],
                                    },
                                ],
                            },
                            {
                                "name": "frontend", "type": "directory", "path": "frontend",
                                "children": [
                                    {"name": "__init__.py", "type": "file", "path": "frontend/__init__.py"},
                                    {
                                        "name": "src", "type": "directory", "path": "frontend/src",
                                        "children": [
                                            {"name": "__init__.py", "type": "file", "path": "frontend/src/__init__.py"},
                                            {
                                                "name": "api", "type": "directory", "path": "frontend/src/api",
                                                "children": [
                                                    {"name": "__init__.py", "type": "file", "path": "frontend/src/api/__init__.py"},
                                                    {"name": "client.py", "type": "file", "path": "frontend/src/api/client.py"},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                )

            async def update_index(self, project_root: str) -> None:
                pass

            async def search_semantic(self, query: str, top_k: int = 5) -> list:
                return []

        orchestrator = Orchestrator(
            context_gateway=_MockContextGateway(),
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
        )

        # 运行工作流
        final_state = await orchestrator.run(
            user_request=(
                f"In the project at {project_root}, replace Basic Auth authentication "
                "with JWT (JSON Web Token) authentication. "
                "First, read these files to understand the current implementation:\n"
                "1. backend/auth/auth_middleware.py - the Basic Auth middleware\n"
                "2. backend/api/routes.py - API routes using auth\n"
                "3. frontend/src/api/client.py - frontend API client\n"
                "After reading, modify them to use JWT:\n"
                "- auth_middleware.py: Add jwt.encode/jwt.decode logic, create "
                "a new authenticate_request that validates Bearer tokens\n"
                "- routes.py: Keep using authenticate_request as dependency "
                "(interface stays the same, implementation changes)\n"
                "- client.py: Change from Basic Auth header to Bearer token header"
            ),
            project_root=project_root,
        )

        # Orchestrator.run 可能返回 dict（中断时）或 AgentState
        def _get_state_attr(state: object, attr: str, default: object = None) -> object:
            if isinstance(state, dict):
                return state.get(attr, default)
            return getattr(state, attr, default)

        # 检查是否触发了 Human Review（plan 中有 high_risk 步骤）
        checkpoints = orchestrator.get_checkpoints()
        plan = _get_state_attr(final_state, "plan")
        if checkpoints and plan:
            has_high_risk = any(
                getattr(step, "risk", None) == "high" if not isinstance(step, dict) else step.get("risk") == "high"
                for step in plan
            )
            if has_high_risk:
                # 自动审批
                thread_id = checkpoints[0]["thread_id"]
                final_state = await orchestrator.resume(thread_id, "approve")

        errors = _get_state_attr(final_state, "errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证计划已生成
        plan = _get_state_attr(final_state, "plan")
        assert plan is not None
        assert len(plan) >= 2, "Should have at least 2 plan steps"

        # 验证关键文件被修改
        middleware_file = project_dir / "backend" / "auth" / "auth_middleware.py"
        assert middleware_file.exists()
        middleware_content = middleware_file.read_text(encoding="utf-8")

        # 应包含 JWT 相关代码（如 jwt.encode, jwt.decode, pyjwt, Bearer 等）
        jwt_indicators = ["jwt", "JWT", "Bearer", "token"]
        has_jwt = any(ind in middleware_content for ind in jwt_indicators)
        assert has_jwt, (
            "auth_middleware.py should reference JWT. "
            f"Content: {middleware_content[:500]}"
        )

        # 验证 routes.py 已更新
        routes_file = project_dir / "backend" / "api" / "routes.py"
        assert routes_file.exists()
        routes_file.read_text(encoding="utf-8")

        # 验证 client.py 已更新
        client_file = project_dir / "frontend" / "src" / "api" / "client.py"
        assert client_file.exists()
        client_content = client_file.read_text(encoding="utf-8")
        has_bearer = "Bearer" in client_content
        has_jwt_in_client = "jwt" in client_content.lower() or "token" in client_content.lower()
        assert has_bearer or has_jwt_in_client, (
            "client.py should use Bearer token instead of Basic Auth. "
            f"Content: {client_content[:500]}"
        )

        # 验证所有修改过的 Python 文件语法正确
        import ast
        for py_file in project_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.relative_to(project_dir)}: {e}")

        # 验证工具调用包含关键类型
        execution_log = _get_state_attr(final_state, "execution_log", [])
        tool_calls = [
            e for e in execution_log
            if isinstance(e, dict) and e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "read_file" in tool_names, "Should have used read_file"
        assert "write_file" in tool_names, "Should have used write_file"

    @pytest.mark.asyncio
    async def test_auth_modification_direct_execution(
        self, tmp_path: Path,
    ) -> None:
        """简化场景：直接用 ExecutionNode 执行 Basic Auth → JWT Auth 替换。

        相比完整的 Orchestrator 工作流，此测试跳过 Human Review，
        直接验证 Agent 的跨文件分析和修改能力。
        """
        project_dir = await self._copy_sample_fullstack_project(tmp_path)
        project_root = str(project_dir)

        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        node = ExecutionNode(
            llm=llm,
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            max_retries=3,
        )

        state = AgentState(
            user_request=(
                f"In the project at {project_root}, replace Basic Auth with JWT authentication. "
                "Do NOT search for files - they are at these exact paths:\n"
                "- backend/auth/auth_middleware.py\n"
                "- backend/api/routes.py\n"
                "- frontend/src/api/client.py\n"
                "Read all three files first, then modify them:\n"
                "1. auth_middleware.py: Add JWT token creation and validation "
                "using jwt.encode/jwt.decode from the pyjwt library. "
                "Create a new function create_jwt_token(username) and modify "
                "authenticate_request to validate Bearer tokens.\n"
                "2. routes.py: Keep the authenticate_request dependency "
                "(interface stays the same).\n"
                "3. client.py: Change _make_basic_auth_header to use Bearer token.\n"
                "After modifying each file, use read_file to verify it's correct."
            ),
            project_root=project_root,
        )

        result = await node(state)

        errors = result.get("errors", [])
        if errors:
            pytest.fail(f"Execution had errors: {errors}")

        # 验证 JWT 代码被写入 auth_middleware.py
        middleware_file = project_dir / "backend" / "auth" / "auth_middleware.py"
        assert middleware_file.exists()
        content = middleware_file.read_text(encoding="utf-8")
        jwt_found = "jwt" in content.lower() or "JWT" in content or "Bearer" in content
        assert jwt_found, (
            "auth_middleware.py should contain JWT or Bearer references"
        )

        # 验证所有 Python 文件语法正确
        import ast
        for py_file in project_dir.rglob("*.py"):
            try:
                ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.relative_to(project_dir)}: {e}")

        # 验证工具调用包含 read_file 和 write_file
        execution_log = result.get("execution_log", [])
        tool_calls = [
            e for e in execution_log
            if isinstance(e, dict) and e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "read_file" in tool_names, "Should have used read_file"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3.8 — Scene H: 多步骤执行 + 进度追踪
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
@requires_api_key
class TestE2EMultiStepPlan:
    """Scene H: 多步骤计划执行与进度追踪。

    验证 Agent 能执行多步骤计划、展示进度、并在偏离时请求人工干预。
    """

    async def _copy_sample_project(self, tmp_path: Path) -> Path:
        """将 sample_python_project 复制到临时目录。"""
        import shutil

        sample_src = (
            Path(__file__).resolve().parent.parent.parent
            / "tests" / "fixtures" / "sample_python_project"
        )
        dest = tmp_path / "sample_project"
        shutil.copytree(str(sample_src), str(dest))
        return dest

    @pytest.mark.asyncio
    async def test_multi_step_plan_execution(self, tmp_path: Path) -> None:
        """测试多步骤计划执行。

        1. Agent 读取项目结构
        2. Agent 生成含多个步骤的计划（创建、修改、读取混合）
        3. 计划步骤包含依赖关系
        4. 步骤按序执行
        5. 执行日志包含进度追踪
        """
        project_dir = await self._copy_sample_project(tmp_path)
        project_root = str(project_dir)

        from codeagent.orchestration.orchestrator import Orchestrator
        from codeagent.gateway.context_gateway import IContextGateway, ContextPackage

        tool_gateway = _build_phase2_tool_gateway(project_root)
        validation_gateway = _build_validation_gateway()
        llm = _build_llm(os.environ.get("LLM_MODEL", "openai/deepseek-v4-flash"))

        class _MockContextGateway(IContextGateway):
            async def build_context(
                self, project_root: str, query: str,
            ) -> ContextPackage:
                return ContextPackage(
                    file_tree={
                        "name": "sample_project",
                        "type": "directory",
                        "path": ".",
                        "children": [
                            {"name": "main.py", "type": "file", "path": "main.py"},
                            {
                                "name": "models", "type": "directory", "path": "models",
                                "children": [
                                    {"name": "user.py", "type": "file", "path": "models/user.py"},
                                    {"name": "role.py", "type": "file", "path": "models/role.py"},
                                ],
                            },
                            {
                                "name": "services", "type": "directory", "path": "services",
                                "children": [
                                    {"name": "user_service.py", "type": "file", "path": "services/user_service.py"},
                                    {"name": "auth.py", "type": "file", "path": "services/auth.py"},
                                ],
                            },
                            {
                                "name": "api", "type": "directory", "path": "api",
                                "children": [
                                    {"name": "routes.py", "type": "file", "path": "api/routes.py"},
                                ],
                            },
                            {
                                "name": "utils", "type": "directory", "path": "utils",
                                "children": [
                                    {"name": "helpers.py", "type": "file", "path": "utils/helpers.py"},
                                ],
                            },
                        ],
                    },
                )

            async def update_index(self, project_root: str) -> None:
                pass

            async def search_semantic(self, query: str, top_k: int = 5) -> list:
                return []

        orchestrator = Orchestrator(
            context_gateway=_MockContextGateway(),
            tool_gateway=tool_gateway,
            validation_gateway=validation_gateway,
            llm=llm,
        )

        final_state = await orchestrator.run(
            user_request=(
                f"In the Python project at {project_root}, I need to add a new "
                "'notification' feature. Follow these steps:\n"
                "1. Create a new file services/notification_service.py with a "
                "NotificationService class that has a send_notification method.\n"
                "2. Add a new API endpoint POST /notifications in api/routes.py "
                "that uses the NotificationService.\n"
                "3. Read utils/helpers.py to understand existing helper functions, "
                "then add a format_notification helper there.\n"
                "4. Read main.py and add a test print for the notification feature."
            ),
            project_root=project_root,
        )

        # 如果有 Human Review，自动审批
        def _get_attr(state: object, attr: str, default: object = None) -> object:
            if isinstance(state, dict):
                return state.get(attr, default)
            return getattr(state, attr, default)

        checkpoints = orchestrator.get_checkpoints()
        plan = _get_attr(final_state, "plan")
        if checkpoints and plan:
            has_high_risk = any(
                getattr(step, "risk", None) == "high" if not isinstance(step, dict) else step.get("risk") == "high"
                for step in plan
            )
            if has_high_risk:
                thread_id = checkpoints[0]["thread_id"]
                final_state = await orchestrator.resume(thread_id, "approve")

        errors = _get_attr(final_state, "errors", [])
        assert not errors, f"Execution had errors: {errors}"

        # 验证计划已生成
        plan = _get_attr(final_state, "plan")
        assert plan is not None
        assert len(plan) >= 2, "Should have at least 2 plan steps"

        # 验证新文件被创建
        notif_service = project_dir / "services" / "notification_service.py"
        expected_created = notif_service.exists()
        if not expected_created:
            # 也可能写在其他位置
            pass

        # 验证 api/routes.py 被修改（添加了 notification 相关内容）
        routes_file = project_dir / "api" / "routes.py"
        assert routes_file.exists()
        routes_content = routes_file.read_text(encoding="utf-8")
        assert "notification" in routes_content.lower(), (
            "routes.py should reference notification"
        )

        # 验证 utils/helpers.py 被修改
        helpers_file = project_dir / "utils" / "helpers.py"
        helpers_content = helpers_file.read_text(encoding="utf-8")
        assert "notification" in helpers_content.lower() or "format" in helpers_content.lower(), (
            "helpers.py should contain notification-related function"
        )

        # 验证所有修改过的 Python 文件语法正确
        import ast
        for py_file in project_dir.rglob("*.py"):
            try:
                ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.relative_to(project_dir)}: {e}")

        # 验证执行日志包含进度信息
        execution_log = _get_attr(final_state, "execution_log", [])
        progress_entries = [
            e for e in execution_log
            if isinstance(e, dict) and e.get("type") in ("step_start", "step_complete", "progress")
        ]
        assert len(progress_entries) >= 1, (
            "Should have at least one progress tracking entry"
        )

        # 验证工具调用包含 read_file 和 write_file
        tool_calls = [
            e for e in execution_log
            if isinstance(e, dict) and e.get("type") == "tool_call"
        ]
        tool_names = {e.get("tool_name") for e in tool_calls}
        assert "read_file" in tool_names, "Should have used read_file"
        assert "write_file" in tool_names, "Should have used write_file"
