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

CHECKPOINT_SCHEMA_VERSION = 1


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
    saver = AsyncSqliteSaver(conn)
    await _initialize_metadata(conn)
    await cleanup_expired_checkpoints(conn, ttl_days=config.get_checkpoint_ttl_days())
    return saver


async def _initialize_metadata(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS codeagent_checkpoint_schema "
        "(version INTEGER NOT NULL, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    cursor = await conn.execute("SELECT MAX(version) FROM codeagent_checkpoint_schema")
    row = await cursor.fetchone()
    if row is None or row[0] is None:
        await conn.execute(
            "INSERT INTO codeagent_checkpoint_schema(version) VALUES (?)",
            (CHECKPOINT_SCHEMA_VERSION,),
        )
    elif int(row[0]) > CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Checkpoint database schema is newer than this CodeAgent build")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS codeagent_checkpoint_threads ("
        "thread_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, status TEXT NOT NULL, "
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    await conn.commit()


async def register_checkpoint_thread(
    checkpointer: object,
    *,
    thread_id: str,
    task_id: str,
    status: str,
) -> None:
    """Persist the stable task/thread mapping without depending on LangGraph internals."""
    conn = getattr(checkpointer, "conn", None)
    if conn is None:
        return
    await conn.execute(
        "INSERT INTO codeagent_checkpoint_threads(thread_id, task_id, status, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(thread_id) DO UPDATE SET task_id=excluded.task_id, "
        "status=excluded.status, updated_at=CURRENT_TIMESTAMP",
        (thread_id, task_id, status),
    )
    await conn.commit()


async def cleanup_expired_checkpoints(
    conn: aiosqlite.Connection,
    *,
    ttl_days: int,
) -> int:
    """Remove only expired terminal metadata; active/review threads are retained."""
    cursor = await conn.execute(
        "DELETE FROM codeagent_checkpoint_threads "
        "WHERE status IN ('completed', 'failed', 'cancelled') "
        "AND updated_at < datetime('now', ?)",
        (f"-{max(1, ttl_days)} days",),
    )
    await conn.commit()
    return max(0, int(cursor.rowcount or 0))
