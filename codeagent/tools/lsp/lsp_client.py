"""LspClient — LSP 协议客户端（Phase 2 简化版：subprocess 模式）。

设计要点：
1. 按语言维护客户端实例（Python / TypeScript / JavaScript）
2. Phase 2 简化：用 subprocess 调用 ruff/pyright，不维护长连接
3. 支持延迟初始化、自动重连（语言服务器崩溃时自动重启）
4. 客户端池空闲超时关闭（默认 10 分钟）
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Diagnostic:
    """LSP 诊断信息。"""

    file_path: str
    line: int
    column: int
    message: str
    severity: str  # "error" | "warning" | "info"
    source: str  # "ruff" | "pyright" | "ast"
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
        }
        if self.code is not None:
            d["code"] = self.code
        return d


# ── Severity mapping ──────────────────────────────────────────────────────────

# ruff code prefix → severity mapping
_RUFF_SEVERITY_MAP: dict[str, str] = {
    "E": "error",
    "F": "error",
    "W": "warning",
    "D": "info",
    "I": "info",
    "N": "info",
    "UP": "info",
    "ANN": "info",
    "ARG": "info",
    "BLE": "error",
    "B": "error",
    "COM": "info",
    "C4": "info",
    "C90": "warning",
    "DTZ": "info",
    "EM": "info",
    "EXE": "info",
    "FA": "info",
    "ISC": "info",
    "ICN": "info",
    "G": "info",
    "YTT": "info",
    "INP": "info",
    "PIE": "info",
    "PL": "warning",
    "PT": "info",
    "PTH": "info",
    "PYI": "info",
    "Q": "info",
    "RSE": "info",
    "RET": "info",
    "SLF": "info",
    "SLOT": "info",
    "SIM": "info",
    "S": "warning",
    "T10": "info",
    "T20": "info",
    "TCH": "info",
    "TD": "info",
    "TID": "info",
    "TRY": "info",
    "RUF": "info",
}

_VALID_SEVERITIES = frozenset({"error", "warning", "info"})


def _map_ruff_severity(code: str) -> str:
    """根据 ruff 代码前缀映射到严重级别。"""
    if not code:
        return "warning"
    for prefix, severity in _RUFF_SEVERITY_MAP.items():
        if code.startswith(prefix):
            return severity
    return "warning"


def _detect_language(file_path: str) -> str:
    """根据文件扩展名检测语言。"""
    lower = file_path.lower()
    if lower.endswith(".py"):
        return "python"
    if lower.endswith((".ts", ".tsx")):
        return "typescript"
    if lower.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    return "text"


# ── LspClient ─────────────────────────────────────────────────────────────────


class LspClient:
    """LSP 协议客户端——按语言区分，Phase 2 简化使用 subprocess 模式。

    Phase 2 简化版本通过调用 ruff/pyright CLI 获取诊断，后续可升级为
    基于 pygls 的长连接模式。
    """

    def __init__(self, language: str) -> None:
        """初始化 LspClient。

        Args:
            language: 编程语言（"python", "typescript", "javascript"）。
        """
        self.language = language
        self._last_used: float = time.monotonic()
        self._closed: bool = False

    # ── 公开接口 ──────────────────────────────────────────────────────────

    async def open_file(self, file_path: str) -> None:
        """通知语言服务器文件已打开。

        Phase 2 简化版为空操作。
        """
        self._touch()

    async def change_file(self, file_path: str, content: str) -> None:
        """通知语言服务器文件已变更。

        Phase 2 简化版为空操作。
        """
        self._touch()

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """获取文件的诊断信息。

        Args:
            file_path: 文件绝对路径。

        Returns:
            诊断信息列表。
        """
        self._touch()
        fp = Path(file_path)

        if not fp.exists():
            return [
                Diagnostic(
                    file_path=str(fp),
                    line=0,
                    column=0,
                    message=f"File not found: {file_path}",
                    severity="error",
                    source=self.language,
                ),
            ]

        if self.language == "python":
            return await self._get_python_diagnostics(str(fp.resolve()))
        elif self.language in ("typescript", "javascript"):
            # Phase 2: TS/JS 诊断暂返回空，后续可通过 tsc --noEmit 支持
            return []
        else:
            return []

    async def close_file(self, file_path: str) -> None:
        """通知语言服务器文件已关闭。

        Phase 2 简化版为空操作。
        """
        self._touch()

    async def shutdown(self) -> None:
        """关闭连接，释放资源。"""
        self._closed = True

    @property
    def is_idle(self) -> bool:
        """检查是否超过空闲超时（默认 10 分钟 = 600 秒）。"""
        return time.monotonic() - self._last_used > 600

    @property
    def is_closed(self) -> bool:
        """检查是否已关闭。"""
        return self._closed

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _touch(self) -> None:
        """更新最后使用时间。"""
        self._last_used = time.monotonic()

    async def _get_python_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """获取 Python 文件的诊断信息。

        优先级：ruff CLI > ast.parse 语法检查。
        """
        # 1. 尝试 ruff
        diagnostics = await self._run_ruff(file_path)
        if diagnostics is not None:
            return diagnostics

        # 2. 降级到 ast.parse 语法检查
        return self._run_ast_check(file_path)

    async def _run_ruff(self, file_path: str) -> list[Diagnostic] | None:
        """调用 ruff check --output-format json 获取诊断。

        Returns:
            None: ruff 不可用或执行失败。
            list[Diagnostic]: 解析后的诊断列表。
        """
        cmd = ["ruff", "check", "--output-format", "json", file_path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError:
            return None

        if proc.returncode not in (0, 1):
            # returncode 0 = no issues, 1 = issues found, other = error
            return None

        try:
            data = json.loads(stdout.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(data, list):
            return None

        result: list[Diagnostic] = []
        for item in data:
            location = item.get("location", {})
            result.append(Diagnostic(
                file_path=item.get("filename", file_path),
                line=location.get("row", 0),
                column=location.get("column", 0),
                message=item.get("message", ""),
                severity=_map_ruff_severity(item.get("code", "")),
                source="ruff",
                code=item.get("code"),
            ))

        return result

    def _run_ast_check(self, file_path: str) -> list[Diagnostic]:
        """使用 ast.parse 进行 Python 语法检查。"""
        import ast

        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            ast.parse(source)
            return []
        except SyntaxError as e:
            return [
                Diagnostic(
                    file_path=file_path,
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    message=e.msg,
                    severity="error",
                    source="ast",
                    code="syntax-error",
                ),
            ]
        except (OSError, UnicodeDecodeError) as e:
            return [
                Diagnostic(
                    file_path=file_path,
                    line=0,
                    column=0,
                    message=str(e),
                    severity="error",
                    source="ast",
                ),
            ]


# ── LspClientPool ─────────────────────────────────────────────────────────────


class LspClientPool:
    """LSP 客户端池——按语言管理客户端生命周期。

    功能：
    - 延迟初始化：仅在首次使用时创建客户端
    - 空闲超时回收：超过 idle_timeout 秒未使用的客户端自动关闭
    - 自动重连：客户端崩溃后自动创建新实例
    """

    def __init__(self, idle_timeout: int = 600) -> None:
        """初始化客户端池。

        Args:
            idle_timeout: 空闲超时秒数，默认 600（10 分钟）。
        """
        self._idle_timeout = idle_timeout
        self._clients: dict[str, LspClient] = {}

    def get_client(self, language: str) -> LspClient:
        """获取或创建指定语言的客户端。

        Args:
            language: 编程语言。

        Returns:
            LspClient: 客户端实例。
        """
        # 如果已有且未关闭的客户端，直接返回
        existing = self._clients.get(language)
        if existing is not None and not existing.is_closed:
            existing._touch()  # noqa: SLF001
            return existing

        # 创建新客户端
        client = LspClient(language)
        self._clients[language] = client
        return client

    async def shutdown_all(self) -> None:
        """关闭所有客户端。"""
        for client in self._clients.values():
            if not client.is_closed:
                await client.shutdown()
        self._clients.clear()

    def cleanup_idle(self) -> int:
        """清理所有空闲超时的客户端。

        Returns:
            清理的客户端数量。
        """
        to_remove: list[str] = []
        for lang, client in self._clients.items():
            if client.is_idle and not client.is_closed:
                to_remove.append(lang)

        for lang in to_remove:
            del self._clients[lang]

        return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """返回客户端池统计信息。"""
        return {
            "total_clients": len(self._clients),
            "languages": list(self._clients.keys()),
            "idle_timeout": self._idle_timeout,
        }
