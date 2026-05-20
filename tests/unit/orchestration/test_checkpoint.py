"""SQLite Checkpoint 会话持久化单元测试。

覆盖：MemorySaver/AsyncSqliteSaver 工厂、路径配置、目录自动创建、graph.py 集成。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestGetCheckpointer:
    """get_checkpointer 同步工厂函数测试。"""

    def test_default_returns_memory_saver(self):
        """默认返回 MemorySaver（向后兼容）。"""
        from codeagent.orchestration.checkpoint import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = get_checkpointer()
        assert isinstance(cp, MemorySaver)

    def test_use_memory_returns_memory_saver(self):
        """use_memory=True 返回 MemorySaver。"""
        from codeagent.orchestration.checkpoint import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = get_checkpointer(use_memory=True)
        assert isinstance(cp, MemorySaver)


class TestGetAsyncCheckpointer:
    """get_async_checkpointer 异步工厂函数测试。"""

    @pytest.mark.asyncio
    async def test_async_default_returns_async_sqlite_saver(self):
        """get_async_checkpointer 默认返回 AsyncSqliteSaver。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            cp = await get_async_checkpointer(db_path=db_path)
            assert isinstance(cp, AsyncSqliteSaver)
            # 关闭连接以免 Windows 文件锁
            await cp.conn.close()

    @pytest.mark.asyncio
    async def test_async_use_memory_returns_memory_saver(self):
        """get_async_checkpointer(use_memory=True) 返回 MemorySaver。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = await get_async_checkpointer(use_memory=True)
        assert isinstance(cp, MemorySaver)

    @pytest.mark.asyncio
    async def test_async_custom_db_path(self):
        """自定义 db_path 应正确创建 SQLite 数据库文件。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "custom" / "async_test.db")
            cp = await get_async_checkpointer(db_path=db_path)
            assert Path(db_path).parent.exists()
            assert Path(db_path).exists()
            await cp.conn.close()

    @pytest.mark.asyncio
    async def test_async_directory_auto_created(self):
        """目标目录不存在时自动创建。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "x" / "y" / "z" / "async_checkpoints.db"
            cp = await get_async_checkpointer(db_path=str(nested))
            assert nested.parent.exists()
            await cp.conn.close()

    @pytest.mark.asyncio
    async def test_async_config_env_var_overrides_default(self):
        """CHECKPOINT_DB_PATH 环境变量应覆盖默认路径。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = str(Path(tmpdir) / "env_override_async.db")
            with patch.dict(
                os.environ,
                {"CHECKPOINT_DB_PATH": custom_path},
            ):
                cp = await get_async_checkpointer()
                assert Path(custom_path).exists()
                await cp.conn.close()

    @pytest.mark.asyncio
    async def test_async_use_memory_takes_precedence(self):
        """use_memory=True 优先于 db_path。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = await get_async_checkpointer(
            db_path="/tmp/should_not_create_async.db", use_memory=True
        )
        assert isinstance(cp, MemorySaver)
        assert not Path("/tmp/should_not_create_async.db").exists()


class TestBuildWorkflowWithCheckpointer:
    """build_workflow 与 checkpointer 集成测试。"""

    @pytest.mark.asyncio
    async def test_build_workflow_accepts_memory_checkpointer(self):
        """build_workflow 应接受 MemorySaver 类型 checkpointer 参数。"""
        from codeagent.orchestration.checkpoint import get_checkpointer
        from codeagent.orchestration.graph import build_workflow

        cp = get_checkpointer(use_memory=True)
        from langgraph.checkpoint.memory import MemorySaver

        assert isinstance(cp, MemorySaver)

        from unittest.mock import MagicMock

        mock_node = MagicMock(return_value={})
        mock_node.__name__ = "mock_node"

        graph = build_workflow(
            context_node=mock_node,
            planning_node=mock_node,
            execution_node=mock_node,
            validation_node=mock_node,
            checkpointer=cp,
        )
        assert graph is not None

    def test_build_workflow_default_checkpointer(self):
        """build_workflow 默认使用 MemorySaver（与旧行为一致）。"""
        from codeagent.orchestration.graph import _build_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = _build_checkpointer()
        assert isinstance(cp, MemorySaver)

    @pytest.mark.asyncio
    async def test_memory_checkpointer_injected_via_get_checkpointer(self):
        """通过 get_checkpointer(use_memory=True) 注入 MemorySaver。"""
        from codeagent.orchestration.checkpoint import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        cp = get_checkpointer(use_memory=True)
        assert isinstance(cp, MemorySaver)

    @pytest.mark.asyncio
    async def test_graph_compiles_with_async_sqlite_checkpointer(self):
        """使用 AsyncSqliteSaver 时 graph.compile 应成功。"""
        from codeagent.orchestration.checkpoint import get_async_checkpointer
        from codeagent.orchestration.graph import build_workflow

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "async_compile.db"
            cp = await get_async_checkpointer(db_path=str(db_path))

            from unittest.mock import MagicMock

            mock_node = MagicMock(return_value={})
            mock_node.__name__ = "mock_node"

            graph = build_workflow(
                context_node=mock_node,
                planning_node=mock_node,
                execution_node=mock_node,
                validation_node=mock_node,
                checkpointer=cp,
            )
            assert graph is not None
            assert db_path.exists()
            await cp.conn.close()

    @pytest.mark.asyncio
    async def test_build_workflow_handles_no_checkpointer(self):
        """不传 checkpointer 参数时 build_workflow 默认使用 MemorySaver。"""
        from codeagent.orchestration.graph import build_workflow

        from unittest.mock import MagicMock

        mock_node = MagicMock(return_value={})
        mock_node.__name__ = "mock_node"

        # 不传 checkpointer 参数
        graph = build_workflow(
            context_node=mock_node,
            planning_node=mock_node,
            execution_node=mock_node,
            validation_node=mock_node,
        )
        assert graph is not None
