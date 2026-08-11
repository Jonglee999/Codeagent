"""Progressive tool exposure for coding tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from codeagent.gateway.tool_gateway import ToolDefinition
from codeagent.orchestration.policy import RunProfile


_CORE_EXPLORATION_TOOL_NAMES = {
    "read_file",
    "list_files",
    "search_code",
}
_CORE_MUTATION_TOOL_NAMES = {
    "write_file",
    "apply_patch",
}
# run_terminal is intentionally NOT in the base set: simple/direct tasks that
# only read or edit a file should not carry its (large) schema. It is exposed
# on demand when the request references command execution, and always for the
# planned workflow whose "command" steps require it.
_TERMINAL_PATTERN = re.compile(
    r"运行|执行|命令行|命令|终端|跑(一)?下|测试|编译|构建|安装|依赖|调试|环境检查|"
    r"启动|部署|检查语法|运行结果|"
    r"\b(run|exec|execute|terminal|shell|command|test|compile|build|install|npm|pip|"
    r"gradle|maven|pytest|uv|go test|cargo)\b",
    re.I,
)
_DELETE_PATTERN = re.compile(r"删除|移除|清理文件|\b(delete|remove)\b", re.I)
_GIT_PATTERN = re.compile(r"\bgit\b|提交|分支|版本历史|\b(commit|branch|diff|blame)\b", re.I)
_MUTATION_PATTERN = re.compile(
    r"修复|创建|新建|新增|修改|重构|实现|优化|生成|编写|制作|开发|搭建|写|改|"
    r"\b(fix|add|create|change|edit|write|update|refactor|implement|optimize|generate|build)\b",
    re.I,
)
_DIAGNOSTIC_PATTERN = re.compile(
    r"诊断|静态检查|类型检查|语法检查|\b(diagnostic|lint|type.?check|mypy)\b",
    re.I,
)
_EXTERNAL_HINTS: dict[str, re.Pattern[str]] = {
    "github": re.compile(r"github|pull request|\bpr\b|issue|代码仓库|拉取请求", re.I),
    "playwright": re.compile(r"playwright|浏览器|页面交互|点击|截图|browser|screenshot", re.I),
    "fetch": re.compile(r"联网|网上|网页资料|网址|\b(fetch|web|url|https?)\b", re.I),
}
_MCP_EXPLICIT_PATTERN = re.compile(r"\bmcp\b|MCP工具|外部工具", re.I)


def select_mcp_server_names(query: str, server_names: set[str]) -> set[str]:
    """Select configured MCP servers without starting any transport."""
    if not query.strip() or not server_names:
        return set()
    if _MCP_EXPLICIT_PATTERN.search(query):
        return set(server_names)

    selected: set[str] = set()
    lowered_query = query.lower()
    for name in server_names:
        lowered_name = name.lower()
        if len(lowered_name) >= 3 and lowered_name in lowered_query:
            selected.add(name)
            continue
        for capability, pattern in _EXTERNAL_HINTS.items():
            if capability in lowered_name and pattern.search(query):
                selected.add(name)
                break
    return selected


@dataclass(frozen=True)
class CapabilitySelection:
    selected: tuple[ToolDefinition, ...]
    deferred: tuple[ToolDefinition, ...]
    reasons: dict[str, str]

    def public_metadata(self) -> dict[str, Any]:
        return {
            "selected_names": [tool.name for tool in self.selected],
            "deferred_names": [tool.name for tool in self.deferred],
            "selected": [tool.public_metadata() for tool in self.selected],
            "deferred": [tool.public_metadata() for tool in self.deferred],
            "reasons": dict(self.reasons),
        }


def _external_relevant(tool: ToolDefinition, query: str) -> bool:
    lowered_source = tool.source.lower()
    lowered_name = tool.name.lower()
    for server, pattern in _EXTERNAL_HINTS.items():
        if server in lowered_source or server in lowered_name:
            return bool(pattern.search(query))

    # Custom MCP servers are selected by an explicit server/tool-name mention.
    identifiers = re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", f"{tool.source} {tool.name}")
    return any(len(identifier) >= 4 and identifier.lower() in query.lower() for identifier in identifiers)


def select_tool_capabilities(
    query: str,
    tools: list[ToolDefinition],
    run_profile: RunProfile,
) -> CapabilitySelection:
    """Expose the small core set and defer unrelated privileged tools."""

    selected: list[ToolDefinition] = []
    deferred: list[ToolDefinition] = []
    reasons: dict[str, str] = {}
    terminal_intent = bool(_TERMINAL_PATTERN.search(query))
    mutation_intent = bool(_MUTATION_PATTERN.search(query))
    # A command-only follow-up should carry the command tool schema only. The
    # previous conversation commonly already names the file/command to rerun.
    command_only = (
        run_profile.workflow == "direct" and terminal_intent and not mutation_intent
    )

    for tool in tools:
        reason: str | None = None
        if tool.name in _CORE_EXPLORATION_TOOL_NAMES and not command_only:
            reason = "just-in-time code discovery"
        elif tool.name in _CORE_MUTATION_TOOL_NAMES and (
            mutation_intent or run_profile.workflow == "planned" or run_profile.benchmark
        ):
            reason = (
                "benchmark contract requires a candidate file mutation"
                if run_profile.benchmark and not mutation_intent
                else "planned workflow may require a file mutation"
                if run_profile.workflow == "planned" and not mutation_intent
                else "request asks for a file change"
            )
        elif tool.name == "get_diagnostics" and _DIAGNOSTIC_PATTERN.search(query):
            reason = "request explicitly asks for diagnostics"
        elif tool.name == "run_terminal" and (
            terminal_intent or run_profile.workflow == "planned" or run_profile.benchmark
        ):
            reason = (
                "benchmark contract requires executed test evidence"
                if run_profile.benchmark and not terminal_intent
                else "request references command execution or a planned step requires it"
            )
        elif tool.name == "delete_file" and _DELETE_PATTERN.search(query):
            reason = "request explicitly asks to remove content"
        elif tool.name == "git" and _GIT_PATTERN.search(query):
            reason = "request explicitly references version-control work"
        elif tool.external and not run_profile.benchmark and _external_relevant(tool, query):
            reason = "request matches this external capability"
        elif tool.name.lower() in query.lower():
            reason = "request explicitly names the tool"

        if reason is not None:
            selected.append(tool)
            reasons[tool.name] = reason
        else:
            deferred.append(tool)

    # Planned tasks retain the same bounded core set. Planning affects the
    # workflow, not the privilege surface.
    reasons["__policy__"] = f"{run_profile.workflow} workflow with progressive discovery"
    return CapabilitySelection(tuple(selected), tuple(deferred), reasons)
