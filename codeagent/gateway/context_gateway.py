"""Context Gateway — 上下文引擎的抽象接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeSnippet:
    """代码片段，语义搜索的返回结果。

    Attributes:
        file_path: 文件路径（相对项目根目录）
        start_line: 起始行号（1-indexed）
        end_line: 结束行号（含）
        code: 代码内容
        score: 相关性评分（0.0 ~ 1.0）
    """

    file_path: str
    start_line: int
    end_line: int
    code: str
    score: float = 0.0


@dataclass
class ContextPackage:
    """上下文数据包，经裁剪后供 LLM 消费。

    Attributes:
        file_tree: 项目文件树结构
        related_code: 与查询相关的代码片段列表
        dependency_info: 依赖信息
        symbol_table: 符号表摘要列表（Phase 2 新增）
    """

    file_tree: Any = None
    related_code: list[CodeSnippet] = field(default_factory=list)
    dependency_info: dict = field(default_factory=dict)
    symbol_table: list[dict] = field(default_factory=list)


class IContextGateway(ABC):
    """上下文引擎 Gateway 接口。

    编排核心层通过此接口获取项目上下文，不直接依赖上下文引擎实现。
    """

    @abstractmethod
    async def build_context(
        self, project_root: str, query: str
    ) -> ContextPackage:
        """构建完整上下文数据包。

        根据项目根目录和用户查询，收集文件树、相关代码、
        依赖信息并返回裁剪后的上下文包。

        Args:
            project_root: 项目根目录路径
            query: 用户的查询/需求描述

        Returns:
            ContextPackage: 上下文数据包
        """
        ...

    @abstractmethod
    async def update_index(self, project_root: str) -> None:
        """更新项目索引（增量或全量）。

        Args:
            project_root: 项目根目录路径
        """
        ...

    @abstractmethod
    async def search_semantic(
        self, query: str, top_k: int = 5
    ) -> list[CodeSnippet]:
        """语义搜索相关代码片段。

        Args:
            query: 搜索查询
            top_k: 返回结果数量上限

        Returns:
            list[CodeSnippet]: 相关代码片段列表，按相关性降序排列
        """
        ...
