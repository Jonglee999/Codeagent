"""记忆系统 E2E 测试 — 完整 Agent 工作流场景。

Scene K：记忆注入影响 Planning
  - 预置 feedback 记忆 "no-mock-database"
  - 运行 PlanningNode，验证 System Prompt 包含 <relevant_memories>

Scene L：任务完成后自动提取
  - 对话包含显式"记住"指令
  - 运行 ExecutionNode，等待异步提取完成
  - 验证记忆文件存在且下次 recall 可检索

约束：
- 所有测试 Mock LLM 返回固定响应，不发真实 API 请求
- 每个场景独立，不依赖其他场景的副作用
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from codeagent.gateway.memory_gateway import IMemoryGateway
from codeagent.gateway.tool_gateway import IToolGateway, ToolDefinition, ToolResult
from codeagent.gateway.validation_gateway import IValidationGateway, ValidationResult
from codeagent.memory.extractor import MemoryExtractor
from codeagent.memory.manager import MemoryManager
from codeagent.memory.retriever import MemoryRetriever
from codeagent.memory.store import MemoryEntry, MemoryStore, MemoryType
from codeagent.orchestration.nodes.execution_node import ExecutionNode
from codeagent.orchestration.nodes.planning_node import PlanningNode
from codeagent.orchestration.state import AgentState


# ── Mock 工具函数 ────────────────────────────────────────────────────────


def make_mock_llm(response_text: str | None = None):
    """构建模拟 LLM 客户端。

    MemoryExtractor 调用方式：
      await llm_client(model=..., messages=[...])
      response.choices[0].message.content

    PlanningNode 调用方式相同。
    """
    async def mock_llm(**kwargs: Any) -> Any:
        from types import SimpleNamespace
        msg = SimpleNamespace(content=response_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return mock_llm


def make_extractor_llm(response_json: str):
    """构建 MemoryExtractor 专用模拟 LLM。"""
    return make_mock_llm(response_json)


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def memory_roots(tmp_path: Path) -> tuple[Path, Path]:
    """创建全局和项目记忆根目录。"""
    global_root = tmp_path / "global_memory"
    project_root = tmp_path / "project_memory"
    global_root.mkdir(parents=True)
    project_root.mkdir(parents=True)
    return global_root, project_root


def make_manager(
    global_root: Path,
    project_root: Path | None = None,
    llm_client: Any = None,
) -> MemoryManager:
    """构建 MemoryManager 实例。"""
    store = MemoryStore(global_root=global_root, project_root=project_root)
    retriever = MemoryRetriever(store=store)
    if llm_client is None:
        llm_client = make_mock_llm("[]")
    extractor = MemoryExtractor(
        store=store,
        retriever=retriever,
        llm_client=llm_client,
        auto_mode=True,
        min_confidence=0.0,
    )
    return MemoryManager(store=store, retriever=retriever, extractor=extractor)


# ══════════════════════════════════════════════════════════════════════════
# Scene K：记忆注入影响 Planning
# ══════════════════════════════════════════════════════════════════════════


class TestSceneK_MemoryInPlanning:
    """记忆注入应影响 PlanningNode 的 System Prompt。"""

    @pytest.mark.asyncio
    async def test_memory_section_in_planning_prompt(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """预置 feedback 记忆后，PlanningNode 的 LLM 调用应收到含 <relevant_memories> 的 prompt。"""
        global_root, project_root = memory_roots

        # 使用 Mock memory_gateway 确保返回固定 XML
        mock_gateway = AsyncMock(spec=IMemoryGateway)
        mock_gateway.recall.return_value = (
            '<relevant_memories>'
            '  <memory type="feedback" name="no-mock-database" score="0.90" scope="global">'
            '    集成测试必须使用真实数据库，不允许 Mock'
            '  </memory>'
            '</relevant_memories>'
        )

        # 捕获 LLM 收到的 messages
        captured_messages: list[list[dict]] = []

        async def capturing_llm(**kwargs: Any) -> Any:
            captured_messages.append(kwargs.get("messages", []))
            from types import SimpleNamespace
            plan_response = (
                '{"plan": [{"step_id": 1, "description": "Write tests", '
                '"action": "read", "target_file": "test.py", '
                '"risk": "low", "dependencies": []}], '
                '"original_goal_summary": "Write integration tests"}'
            )
            msg = SimpleNamespace(content=plan_response)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = PlanningNode(
            llm=capturing_llm,
            memory_gateway=mock_gateway,
        )

        state = AgentState(
            user_request="为 UserService 写集成测试",
            project_root=str(project_root),
        )

        await node(state)

        # 验证 captured_messages 非空
        assert len(captured_messages) > 0, "LLM should have been called"
        system_msgs = [
            m["content"] for m in captured_messages[0]
            if m["role"] == "system"
        ]
        assert len(system_msgs) > 0, "System prompt should exist"

        full_prompt = system_msgs[0]
        assert "<relevant_memories>" in full_prompt, \
            "System prompt should contain relevant_memories XML"
        assert "no-mock-database" in full_prompt, \
            "System prompt should contain the memory name"

    @pytest.mark.asyncio
    async def test_full_pipeline_memory_in_planning(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """完整管道：预置真实记忆 → PlanningNode System Prompt 包含该记忆。"""
        global_root, project_root = memory_roots

        # 保存一条与 user_request 高度相关的记忆（使用英文确保 tokenizer 正确匹配）
        mgr = make_manager(global_root, project_root)
        mgr.save(MemoryEntry(
            name="no-mock-database", memory_type=MemoryType.FEEDBACK,
            description="Integration test must use real database no mock",
            body="Always use real database in integration tests",
        ), scope="global")

        captured_prompt: list[str] = []

        async def capturing_llm(**kwargs: Any) -> Any:
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if m["role"] == "system":
                    captured_prompt.append(m["content"])
            from types import SimpleNamespace
            plan_response = (
                '{"plan": [{"step_id": 1, "description": "Write test", '
                '"action": "read", "target_file": "test.py", '
                '"risk": "low", "dependencies": []}], '
                '"original_goal_summary": "Write integration tests"}'
            )
            msg = SimpleNamespace(content=plan_response)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = PlanningNode(
            llm=capturing_llm,
            memory_gateway=mgr,
        )
        state = AgentState(
            user_request="write integration test for UserService",
            project_root=str(project_root),
        )
        await node(state)

        prompt = captured_prompt[0] if captured_prompt else ""
        assert "<relevant_memories>" in prompt, \
            f"Full pipeline: expected <relevant_memories> in prompt, got:\n{prompt[:500]}"

    @pytest.mark.asyncio
    async def test_planning_no_memory_no_section(
        self, tmp_path: Path,
    ) -> None:
        """无记忆 gateway 时 System Prompt 不应包含记忆层。"""
        captured_messages: list[list[dict]] = []

        async def capturing_llm(**kwargs: Any) -> Any:
            captured_messages.append(kwargs.get("messages", []))
            from types import SimpleNamespace
            plan_response = (
                '{"plan": [{"step_id": 1, "description": "Do something", '
                '"action": "read", "target_file": "f.py", '
                '"risk": "low", "dependencies": []}], '
                '"original_goal_summary": "Test"}'
            )
            msg = SimpleNamespace(content=plan_response)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = PlanningNode(llm=capturing_llm, memory_gateway=None)

        state = AgentState(
            user_request="Write code",
            project_root=str(tmp_path),
        )
        await node(state)

        system_prompt = captured_messages[0][0]["content"]
        assert "<relevant_memories>" not in system_prompt
        assert "Relevant Memories" not in system_prompt

    @pytest.mark.asyncio
    async def test_planning_memory_xml_format(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """记忆注入的 XML 格式应正确（包含 type/name/score/scope 属性）。"""
        global_root, project_root = memory_roots

        mgr = make_manager(global_root, project_root)
        mgr.save(MemoryEntry(
            name="async-service", memory_type=MemoryType.CODE_PATTERN,
            description="Use async/await", body="All services are async",
        ), scope="project")

        captured_prompt: list[str] = []

        async def capturing_llm(**kwargs: Any) -> Any:
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if m["role"] == "system":
                    captured_prompt.append(m["content"])
            from types import SimpleNamespace
            msg = SimpleNamespace(content='{"plan":[],"original_goal_summary":"Test"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = PlanningNode(llm=capturing_llm, memory_gateway=mgr)
        state = AgentState(
            user_request="Build async service",
            project_root=str(project_root),
        )
        await node(state)

        prompt = captured_prompt[0] if captured_prompt else ""
        assert '<memory type="code_pattern"' in prompt or '<memory type="code_pattern"' in prompt
        assert 'name="async-service"' in prompt
        assert 'scope=' in prompt

    @pytest.mark.asyncio
    async def test_empty_memory_empty_section(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """无相关记忆时不应在 prompt 中添加记忆层。"""
        global_root, project_root = memory_roots
        mgr = make_manager(global_root, project_root)
        # 不保存任何记忆

        captured_prompt: list[str] = []

        async def capturing_llm(**kwargs: Any) -> Any:
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if m["role"] == "system":
                    captured_prompt.append(m["content"])
            from types import SimpleNamespace
            msg = SimpleNamespace(content='{"plan":[],"original_goal_summary":"Empty"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = PlanningNode(llm=capturing_llm, memory_gateway=mgr)
        state = AgentState(user_request="test", project_root=str(project_root))
        await node(state)

        prompt = captured_prompt[0] if captured_prompt else ""
        assert "Relevant Memories" not in prompt


# ══════════════════════════════════════════════════════════════════════════
# Scene L：任务完成后自动提取
# ══════════════════════════════════════════════════════════════════════════


class TestSceneL_PostTaskExtraction:
    """工作流完成后应自动提取记忆。"""

    EXTRACTION_JSON = (
        '[{"type": "feedback", "name": "use-snake-case", '
        '"description": "API returns snake_case", '
        '"body": "All API responses use snake_case\\n\\n**Why:** Project convention", '
        '"confidence": 0.95}]'
    )

    @pytest.mark.asyncio
    async def test_execution_triggers_extraction(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """ExecutionNode 完成后应异步调用 auto_extract。"""
        global_root, project_root = memory_roots
        ext_llm = make_extractor_llm(self.EXTRACTION_JSON)
        mgr = make_manager(global_root, project_root, llm_client=ext_llm)

        # Mock tools
        tool_gw = AsyncMock(spec=IToolGateway)
        tool_gw.list_tools.return_value = []
        tool_gw.execute_tool.return_value = ToolResult(
            success=True, data={}, duration_ms=0,
        )
        val_gw = AsyncMock(spec=IValidationGateway)
        val_gw.run_syntax_check.return_value = ValidationResult(
            passed=True, errors=[], duration_ms=0,
        )

        # LLM 直接返回文本（无 tool_calls → 直连模式完成）
        from types import SimpleNamespace
        async def done_llm(**kwargs: Any) -> Any:
            msg = SimpleNamespace(content="Task completed", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = ExecutionNode(
            llm=done_llm,
            tool_gateway=tool_gw,
            validation_gateway=val_gw,
            memory_gateway=mgr,
        )

        state = AgentState(
            user_request="记住：该项目所有 API 返回 snake_case",
            project_root=str(project_root),
            conversation_history=[
                {"role": "user", "content": "记住：该项目所有 API 返回 snake_case"},
            ],
        )

        result = await node(state)
        assert result.get("memory_extracted") is True

        # 等待异步提取完成
        await asyncio.sleep(0.1)

        # 验证记忆被保存到 store
        memories = mgr.list_memories()
        names = [m.name for m in memories]
        assert "use-snake-case" in names, \
            f"Extracted memory should be in store: {names}"

    @pytest.mark.asyncio
    async def test_extracted_memory_recallable(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """提取后的记忆在下次 recall 时应能检索到。"""
        global_root, project_root = memory_roots
        ext_llm = make_extractor_llm(self.EXTRACTION_JSON)
        mgr = make_manager(global_root, project_root, llm_client=ext_llm)

        tool_gw = AsyncMock(spec=IToolGateway)
        tool_gw.list_tools.return_value = []
        val_gw = AsyncMock(spec=IValidationGateway)
        val_gw.run_syntax_check.return_value = ValidationResult(
            passed=True, errors=[], duration_ms=0,
        )

        from types import SimpleNamespace
        async def done_llm(**kwargs: Any) -> Any:
            msg = SimpleNamespace(content="Done", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        # 第一次执行：触发提取
        node = ExecutionNode(
            llm=done_llm, tool_gateway=tool_gw,
            validation_gateway=val_gw, memory_gateway=mgr,
        )
        state = AgentState(
            user_request="记住 API 返回 snake_case",
            project_root=str(project_root),
            conversation_history=[
                {"role": "user", "content": "记住 API 返回 snake_case"},
            ],
        )
        await node(state)
        await asyncio.sleep(0.1)

        # 随后在新会话中 recall
        mgr2 = make_manager(global_root, project_root, llm_client=ext_llm)
        xml = await mgr2.recall("API response format")
        assert "use-snake-case" in xml
        assert "snake_case" in xml

    @pytest.mark.asyncio
    async def test_no_duplicate_extraction_across_calls(
        self, memory_roots: tuple[Path, Path],
    ) -> None:
        """多次调用 __call__ 不会重复提取（memory_extracted 防护）。"""
        global_root, project_root = memory_roots
        ext_llm = make_extractor_llm(self.EXTRACTION_JSON)
        mgr = make_manager(global_root, project_root, llm_client=ext_llm)

        tool_gw = AsyncMock(spec=IToolGateway)
        tool_gw.list_tools.return_value = []
        val_gw = AsyncMock(spec=IValidationGateway)
        val_gw.run_syntax_check.return_value = ValidationResult(
            passed=True, errors=[], duration_ms=0,
        )

        from types import SimpleNamespace
        async def done_llm(**kwargs: Any) -> Any:
            msg = SimpleNamespace(content="Done", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = ExecutionNode(
            llm=done_llm, tool_gateway=tool_gw,
            validation_gateway=val_gw, memory_gateway=mgr,
        )
        state = AgentState(
            user_request="记住：API snake_case",
            project_root=str(project_root),
            conversation_history=[{"role": "user", "content": "记住：API snake_case"}],
            # 初始 memory_extracted 为 False
        )

        # 第一次调用
        await node(state)
        assert state.memory_extracted is True

        # 此时不会再次触发提取（memory_extracted=True）
        # 验证 store 中只有一条
        memories1 = mgr.list_memories()
        count_after_first = len(memories1)

        # 第二次调用（已被标记，不会触发提取）
        await node(state)
        memories2 = mgr.list_memories()
        assert len(memories2) == count_after_first

    @pytest.mark.asyncio
    async def test_extraction_skipped_without_memory_gateway(
        self, tmp_path: Path,
    ) -> None:
        """无 memory_gateway 时不应触发提取。"""
        tool_gw = AsyncMock(spec=IToolGateway)
        tool_gw.list_tools.return_value = []
        val_gw = AsyncMock(spec=IValidationGateway)
        val_gw.run_syntax_check.return_value = ValidationResult(
            passed=True, errors=[], duration_ms=0,
        )

        from types import SimpleNamespace
        async def done_llm(**kwargs: Any) -> Any:
            msg = SimpleNamespace(content="Done", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        node = ExecutionNode(
            llm=done_llm, tool_gateway=tool_gw,
            validation_gateway=val_gw,
            # memory_gateway=None (default)
        )
        state = AgentState(
            user_request="test",
            project_root=str(tmp_path),
        )
        result = await node(state)

        # memory_extracted 不应出现在 result 中
        assert result.get("memory_extracted") is not True
        assert state.memory_extracted is not True


# ══════════════════════════════════════════════════════════════════════════
# 附加 E2E 场景：MemoryManager.create() 工厂方法
# ══════════════════════════════════════════════════════════════════════════


class TestMemoryManagerFactory:
    """MemoryManager.create() 工厂方法集成。"""

    @pytest.mark.asyncio
    async def test_create_with_project_path(self, tmp_path: Path) -> None:
        """工厂方法应正确创建带项目路径的 MemoryManager。"""
        # 由于 create() 从 config 读取全局路径，我们需要临时覆盖
        import os
        orig = os.environ.get("MEMORY_GLOBAL_ROOT")
        test_global = tmp_path / "factory_global"
        test_global.mkdir(parents=True)
        os.environ["MEMORY_GLOBAL_ROOT"] = str(test_global)

        try:
            mgr = MemoryManager.create(
                project_path=str(tmp_path / "project"),
                llm_client=make_mock_llm("[]"),
                auto_mode=True,
            )
            assert mgr._store is not None
            assert mgr._retriever is not None
            assert mgr._extractor is not None
            assert mgr._store._project_root is not None
        finally:
            if orig is None:
                del os.environ["MEMORY_GLOBAL_ROOT"]
            else:
                os.environ["MEMORY_GLOBAL_ROOT"] = orig
