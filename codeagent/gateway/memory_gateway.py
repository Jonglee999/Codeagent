"""Memory Gateway — 记忆系统的抽象接口契约。

定义 IMemoryGateway 供上层模块（Context Engine、Orchestration Node）调用，
同时方便单元测试时注入 Mock 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from codeagent.memory.store import MemoryEntry, MemoryType


class IMemoryGateway(ABC):
    """记忆系统对外接口契约，供测试 Mock 使用。"""

    @abstractmethod
    async def recall(self, query: str, token_budget: int = 800) -> str:
        """检索相关记忆，返回 XML 字符串。

        Args:
            query: 查询字符串
            token_budget: token 预算上限

        Returns:
            XML 格式的记忆字符串，无结果时返回空字符串
        """
        ...

    @abstractmethod
    def save(self, entry: MemoryEntry, scope: str = "auto") -> None:
        """保存一条记忆。

        Args:
            entry: 记忆条目
            scope: "auto"、"global" 或 "project"
        """
        ...

    @abstractmethod
    async def auto_extract(
        self,
        conversation_history: list[dict],
        trigger: str = "task_complete",
    ) -> list[MemoryEntry]:
        """从对话历史自动提取并保存记忆。

        Args:
            conversation_history: 对话历史 [{"role":..., "content":...}]
            trigger: 触发原因

        Returns:
            已保存的 MemoryEntry 列表
        """
        ...

    @abstractmethod
    def forget(self, name: str) -> bool:
        """删除指定记忆。

        Args:
            name: 记忆条目的 name

        Returns:
            bool: 是否成功删除
        """
        ...

    @abstractmethod
    def list_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        scope: str = "both",
    ) -> list[MemoryEntry]:
        """列出记忆条目。

        Args:
            memory_type: 筛选特定类型，None 表示全部
            scope: "global"、"project" 或 "both"

        Returns:
            list[MemoryEntry]: 记忆条目列表
        """
        ...
