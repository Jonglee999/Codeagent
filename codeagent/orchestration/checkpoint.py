"""Checkpointer 工厂 — 管理 LangGraph Checkpointer 生命周期。

生产环境使用 AsyncSqliteSaver（SQLite 持久化，支持 async），
测试/开发环境使用 MemorySaver（内存级）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import aiosqlite

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from codeagent import config


def get_checkpointer(
    db_path: Optional[str] = None,
    use_memory: bool = False,
):
    """获取 LangGraph Checkpointer 实例（同步工厂）。

    参数：
        db_path: SQLite 数据库文件路径（默认从 CHECKPOINT_DB_PATH 环境变量读取）
        use_memory: 强制使用 MemorySaver（测试环境使用）

    生产环境使用 MemorySaver（保持同步兼容），
    如需持久化请使用 get_async_checkpointer()。
    """
    if use_memory:
        return MemorySaver()

    path = db_path or config.get_checkpoint_db_path()
    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    return MemorySaver()


async def get_async_checkpointer(
    db_path: Optional[str] = None,
    use_memory: bool = False,
) -> AsyncSqliteSaver | MemorySaver:
    """获取 LangGraph Checkpointer 实例（异步工厂）。

    生产环境使用 AsyncSqliteSaver 实现持久化。
    测试环境使用 MemorySaver。

    参数：
        db_path: SQLite 数据库文件路径（默认从 CHECKPOINT_DB_PATH 环境变量读取）
        use_memory: 强制使用 MemorySaver（测试环境使用）
    """
    if use_memory:
        return MemorySaver()

    path = db_path or config.get_checkpoint_db_path()
    db_file = Path(path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(db_file), check_same_thread=False)
    return AsyncSqliteSaver(conn)
