"""SearchCodeTool 单元测试。

Mock rg 子进程和 ContextEngine，测试两种搜索模式、边界情况。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.gateway.context_gateway import CodeSnippet
from codeagent.tools.search.search_code import SearchCodeTool


class _MockProcess:
    """Mock asyncio subprocess 返回值。"""

    def __init__(
        self,
        stdout_bytes: bytes = b"",
        stderr_bytes: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        self.returncode = returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        return (self.stdout_bytes, self.stderr_bytes)


def _make_match(
    file_path: str,
    line_number: int,
    line_content: str,
    column: int = 1,
    submatch_text: str = "",
) -> str:
    """构造一条 ripgrep JSON match 行。"""
    import json

    data = {
        "type": "match",
        "data": {
            "path": {"text": file_path},
            "lines": {"text": line_content},
            "line_number": line_number,
            "absolute_offset": 0,
            "submatches": [
                {
                    "match": {"text": submatch_text or line_content.strip()},
                    "start": column - 1,
                    "end": column - 1 + len(submatch_text or line_content.strip()),
                }
            ],
        },
    }
    return json.dumps(data)


def _make_context(file_path: str, line_number: int, line_content: str) -> str:
    """构造一条 ripgrep JSON context 行。"""
    import json

    data = {
        "type": "context",
        "data": {
            "path": {"text": file_path},
            "lines": {"text": line_content},
            "line_number": line_number,
        },
    }
    return json.dumps(data)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tool(tmp_path) -> SearchCodeTool:
    """在临时目录创建 SearchCodeTool 实例。"""
    return SearchCodeTool(project_root=tmp_path)


@pytest.fixture
def mock_context_engine() -> MagicMock:
    """创建 mock ContextEngine 实例。"""
    engine = MagicMock()
    engine.search_semantic = AsyncMock()
    return engine


@pytest.fixture
def tool_with_engine(
    tmp_path, mock_context_engine: MagicMock,
) -> SearchCodeTool:
    """创建带 ContextEngine 的 SearchCodeTool 实例。"""
    return SearchCodeTool(
        project_root=tmp_path,
        context_engine=mock_context_engine,
    )


# ── Regex 模式测试 ────────────────────────────────────────────────────────────


class TestRegexSearch:
    """正则搜索模式测试。"""

    @pytest.mark.asyncio
    async def test_basic_regex_search(self, tool: SearchCodeTool) -> None:
        """基本正则搜索应返回匹配结果。"""
        "\n".join([
            _make_match("src/main.py", 10, "def hello():\n", column=1, submatch_text="hello"),
            _make_match("src/main.py", 20, "    return hello\n", column=5, submatch_text="hello"),
            '{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":1234},"stats":{"matches":2}}}',
        ])

        with patch.object(
            type(tool), "_execute_regex", new=AsyncMock(),
        ) as mock_regex:
            mock_regex.return_value = type("TR", (), {
                "success": True,
                "data": {
                    "results": [
                        {
                            "file_path": "src/main.py",
                            "line": 10,
                            "column": 1,
                            "line_content": 'def hello():',
                        },
                        {
                            "file_path": "src/main.py",
                            "line": 20,
                            "column": 5,
                            "line_content": "    return hello",
                        },
                    ],
                    "total_results": 2,
                    "search_type": "regex",
                },
                "error_message": None,
                "error_code": None,
                "duration_ms": 1.0,
            })()

            result = await tool.execute(query="hello", search_type="regex")
            assert result.success is True
            assert result.data["total_results"] == 2
            assert result.data["search_type"] == "regex"
            assert len(result.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_regex_no_matches(self, tool: SearchCodeTool) -> None:
        """无匹配时应返回空结果列表。"""
        ndjson = '{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":500},"stats":{"matches":0}}}'

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.return_value = _MockProcess(
                stdout_bytes=ndjson.encode(),
                returncode=1,  # rg returncode 1 = no matches
            )

            result = await tool.execute(query="nonexistent_pattern_xyz")
            assert result.success is True
            assert result.data["total_results"] == 0
            assert result.data["results"] == []

    @pytest.mark.asyncio
    async def test_regex_with_file_pattern(self, tool: SearchCodeTool) -> None:
        """file_pattern 应过滤搜索范围。"""
        ndjson = "\n".join([
            _make_match("src/app.py", 5, "import os\n", column=1, submatch_text="os"),
            '{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":100},"stats":{"matches":1}}}',
        ])

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.return_value = _MockProcess(
                stdout_bytes=ndjson.encode(),
                returncode=0,
            )

            result = await tool.execute(
                query="import",
                file_pattern="*.py",
            )
            assert result.success is True
            assert result.data["total_results"] == 1

    @pytest.mark.asyncio
    async def test_path_qualified_glob_matches_under_absolute_search_root(
        self, tool: SearchCodeTool, tmp_path
    ) -> None:
        target = tmp_path / "src" / "_pytest" / "mark" / "structures.py"
        target.parent.mkdir(parents=True)
        target.write_text("pytestmark = []\n", encoding="utf-8")

        result = await tool.execute(
            query="pytestmark",
            file_pattern="src/_pytest/mark/*.py",
        )

        assert result.success is True
        assert result.data["total_results"] == 1
        assert result.data["results"][0]["file_path"] == (
            "src/_pytest/mark/structures.py"
        )

    @pytest.mark.asyncio
    async def test_regex_with_context_lines(self, tool: SearchCodeTool) -> None:
        """context_lines 参数应传递到 rg。"""
        ndjson = "\n".join([
            _make_context("src/app.py", 9, "line before\n"),
            _make_match("src/app.py", 10, "def target():\n", column=1, submatch_text="target"),
            _make_context("src/app.py", 11, "line after\n"),
            '{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":100},"stats":{"matches":1}}}',
        ])

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.return_value = _MockProcess(
                stdout_bytes=ndjson.encode(),
                returncode=0,
            )

            result = await tool.execute(
                query="target",
                context_lines=1,
            )
            assert result.success is True
            assert result.data["total_results"] == 1

    @pytest.mark.asyncio
    async def test_regex_deduplication(self, tool: SearchCodeTool) -> None:
        """同一文件同一行的重复匹配应去重。"""
        ndjson = "\n".join([
            _make_match("src/app.py", 10, "foo bar foo\n", column=1, submatch_text="foo"),
            _make_match("src/app.py", 10, "foo bar foo\n", column=9, submatch_text="foo"),
            '{"type":"summary","data":{"elapsed_total":{"secs":0,"nanos":100},"stats":{"matches":2}}}',
        ])

        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.return_value = _MockProcess(
                stdout_bytes=ndjson.encode(),
                returncode=0,
            )

            result = await tool.execute(query="foo")
            assert result.success is True
            assert result.data["total_results"] == 1  # deduplicated


# ── Semantic 模式测试 ─────────────────────────────────────────────────────────


class TestSemanticSearch:
    """语义搜索模式测试。"""

    @pytest.mark.asyncio
    async def test_semantic_search_basic(
        self,
        tool_with_engine: SearchCodeTool,
        mock_context_engine: MagicMock,
    ) -> None:
        """语义搜索应返回 ContextEngine 的结果。"""
        mock_context_engine.search_semantic.return_value = [
            CodeSnippet(
                file_path="services/auth.py",
                start_line=42,
                end_line=50,
                code="def login_user():\n    pass\n",
                score=0.95,
            ),
            CodeSnippet(
                file_path="services/auth.py",
                start_line=10,
                end_line=20,
                code="class AuthService:\n    pass\n",
                score=0.85,
            ),
        ]

        result = await tool_with_engine.execute(
            query="user login functionality",
            search_type="semantic",
        )
        assert result.success is True
        assert result.data["total_results"] == 2
        assert result.data["search_type"] == "semantic"
        assert result.data["results"][0]["file_path"] == "services/auth.py"
        assert result.data["results"][0]["line"] == 42
        assert result.data["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_semantic_no_engine(self, tool: SearchCodeTool) -> None:
        """没有配置 ContextEngine 时应返回错误。"""
        result = await tool.execute(
            query="login",
            search_type="semantic",
        )
        assert result.success is False
        assert result.error_code == "CONTEXT_ENGINE_NOT_CONFIGURED"

    @pytest.mark.asyncio
    async def test_semantic_search_error(
        self,
        tool_with_engine: SearchCodeTool,
        mock_context_engine: MagicMock,
    ) -> None:
        """ContextEngine 异常时应返回错误。"""
        mock_context_engine.search_semantic.side_effect = RuntimeError(
            "Index not built",
        )

        result = await tool_with_engine.execute(
            query="login",
            search_type="semantic",
        )
        assert result.success is False
        assert result.error_code == "SEMANTIC_SEARCH_ERROR"

    @pytest.mark.asyncio
    async def test_semantic_empty_results(
        self,
        tool_with_engine: SearchCodeTool,
        mock_context_engine: MagicMock,
    ) -> None:
        """语义搜索无结果时返回空列表。"""
        mock_context_engine.search_semantic.return_value = []

        result = await tool_with_engine.execute(
            query="xyznonexistent",
            search_type="semantic",
        )
        assert result.success is True
        assert result.data["total_results"] == 0


# ── 边界情况测试 ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_empty_query(self, tool: SearchCodeTool) -> None:
        """空查询应返回错误。"""
        result = await tool.execute(query="")
        assert result.success is False
        assert result.error_code == "EMPTY_QUERY"

    @pytest.mark.asyncio
    async def test_whitespace_query(self, tool: SearchCodeTool) -> None:
        """纯空格查询应返回错误。"""
        result = await tool.execute(query="   ")
        assert result.success is False
        assert result.error_code == "EMPTY_QUERY"

    @pytest.mark.asyncio
    async def test_invalid_search_type(self, tool: SearchCodeTool) -> None:
        """无效 search_type 应返回错误。"""
        result = await tool.execute(query="test", search_type="fuzzy")
        assert result.success is False
        assert result.error_code == "INVALID_SEARCH_TYPE"

    @pytest.mark.asyncio
    async def test_rg_not_found_uses_python_fallback(
        self, tool: SearchCodeTool, tmp_path
    ) -> None:
        """rg 未安装时应返回友好错误。"""
        (tmp_path / "example.py").write_text(
            "first line\nneedle = True\n", encoding="utf-8"
        )
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError(
                "No such file or directory: 'rg'",
            )

            result = await tool.execute(query="needle", file_pattern="*.py")

            assert result.success is True
            assert result.data["engine"] == "python-fallback"
            assert result.data["results"] == [{
                "file_path": "example.py",
                "line": 2,
                "column": 1,
                "line_content": "needle = True",
            }]

    @pytest.mark.asyncio
    async def test_rg_error(self, tool: SearchCodeTool) -> None:
        """rg 执行错误应返回错误信息。"""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_subprocess.return_value = _MockProcess(
                stdout_bytes=b"",
                stderr_bytes=b"error: invalid glob pattern",
                returncode=2,
            )

            result = await tool.execute(query="test")
            assert result.success is False
            assert result.error_code == "RG_ERROR"

    @pytest.mark.asyncio
    async def test_search_path_cannot_escape_project(self, tool: SearchCodeTool) -> None:
        result = await tool.execute(query="anything", paths=["../outside"])
        assert result.success is False
        assert result.error_code == "PATH_OUTSIDE_PROJECT"


@pytest.mark.asyncio
async def test_hybrid_search_fuses_regex_ast_and_semantic(tmp_path) -> None:
    engine = MagicMock()
    engine.search_semantic = AsyncMock(return_value=[CodeSnippet(
        file_path="auth.py", start_line=10, end_line=12,
        code="def authenticate(): pass", score=0.9,
    )])
    engine.search_structural = AsyncMock(return_value=[CodeSnippet(
        file_path="auth.py", start_line=10, end_line=12,
        code="def authenticate(): pass", score=1.0,
    )])
    tool = SearchCodeTool(tmp_path, context_engine=engine)
    lexical = type("TR", (), {
        "success": True,
        "data": {"results": [{
            "file_path": "auth.py", "line": 10, "column": 1,
            "line_content": "def authenticate(): pass",
        }]},
    })()
    with patch.object(tool, "_execute_regex", new=AsyncMock(return_value=lexical)):
        result = await tool.execute("authenticate", search_type="hybrid")

    assert result.success is True
    assert result.data["results"][0]["sources"] == ["ast", "regex", "semantic"]


@pytest.mark.asyncio
async def test_explore_search_returns_compact_file_digest(tmp_path) -> None:
    (tmp_path / "auth.py").write_text(
        "def authentication_handler(request):\n    return request.user\n",
        encoding="utf-8",
    )
    (tmp_path / "routes.py").write_text(
        "from auth import authentication_handler\n",
        encoding="utf-8",
    )
    tool = SearchCodeTool(tmp_path)

    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await tool.execute(
            "authentication handler",
            search_type="explore",
            max_results=100,
        )

    assert result.success is True
    assert result.data["search_type"] == "explore"
    assert result.data["raw_results_returned"] is False
    assert result.data["context_firewall"]["max_files"] == 5
    assert len(result.data["files"]) <= 5
    assert result.data["files"][0]["file_path"] in {"auth.py", "routes.py"}
    assert "authentication_handler" in result.data["files"][0]["excerpt"]
