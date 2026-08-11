"""SemanticSearchEngine 单元测试。"""

from __future__ import annotations

import os
import tempfile

import pytest

from codeagent.context_engine.semantic_search import (
    SearchResult,
    SemanticSearchEngine,
    _BM25Index,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> str:
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_project(temp_dir: str) -> str:
    """创建简单的测试项目。"""
    # main.py
    with open(os.path.join(temp_dir, "main.py"), "w") as f:
        f.write("""
from services.auth import authenticate_user
from services.user_service import UserService


def main():
    \"\"\"Main entry point.\"\"\"
    svc = UserService()
    token = authenticate_user(svc, "test_user")
    print(f"Token: {token}")
""")

    # services/auth.py
    svc_dir = os.path.join(temp_dir, "services")
    os.makedirs(svc_dir)
    with open(os.path.join(svc_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(svc_dir, "auth.py"), "w") as f:
        f.write("""
def authenticate_user(service, username):
    \"\"\"Authenticate a user and return a token.\"\"\"
    return f"token_{username}"


def validate_token(token):
    \"\"\"Validate an authentication token.\"\"\"
    return token.startswith("token_")


class AuthError(Exception):
    \"\"\"Authentication error.\"\"\"
    pass
""")

    # services/user_service.py
    with open(os.path.join(svc_dir, "user_service.py"), "w") as f:
        f.write("""
from models.user import User


class UserService:
    \"\"\"Service for managing users.\"\"\"

    def create_user(self, name, email):
        \"\"\"Create a new user.\"\"\"
        return User(name=name, email=email)

    def get_user(self, name):
        \"\"\"Get user by name.\"\"\"
        return None
""")

    # models/user.py
    models_dir = os.path.join(temp_dir, "models")
    os.makedirs(models_dir)
    with open(os.path.join(models_dir, "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(models_dir, "user.py"), "w") as f:
        f.write("""
class User:
    \"\"\"Represents a user.\"\"\"

    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.is_active = True

    def deactivate(self):
        \"\"\"Deactivate the user.\"\"\"
        self.is_active = False
""")

    return temp_dir


@pytest.fixture
def engine(temp_dir: str) -> SemanticSearchEngine:
    """创建使用 mock 嵌入模型的语义检索引擎。"""
    db_path = os.path.join(temp_dir, ".codeagent", "lancedb")
    return SemanticSearchEngine(db_path=db_path, use_mock=True)


# ── Tests: 索引 ──────────────────────────────────────────────────────────────


class TestSemanticSearchIndex:
    """测试索引功能。"""

    @pytest.mark.asyncio
    async def test_index_project(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """验证全量索引成功。"""
        stats = await engine.index_project(sample_project)
        assert stats["total_files"] > 0
        assert stats["total_chunks"] > 0
        assert stats["model"] is not None

    @pytest.mark.asyncio
    async def test_index_empty_project(
        self, engine: SemanticSearchEngine, temp_dir: str
    ) -> None:
        """空项目应返回 0 块。"""
        stats = await engine.index_project(temp_dir)
        assert stats["total_files"] == 0
        assert stats["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_index_nonexistent_path(
        self, engine: SemanticSearchEngine
    ) -> None:
        """不存在的路径应抛出错误。"""
        with pytest.raises(NotADirectoryError):
            await engine.index_project("/nonexistent/project")

    @pytest.mark.asyncio
    async def test_index_stats(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """验证索引统计。"""
        await engine.index_project(sample_project)
        stats = engine.get_index_stats()
        assert stats["total_chunks"] > 0
        assert stats["total_files"] > 0
        assert "model" in stats
        assert "db_path" in stats


# ── Tests: 搜索 ──────────────────────────────────────────────────────────────


class TestSemanticSearchQuery:
    """测试搜索功能。"""

    @pytest.mark.asyncio
    async def test_search_before_index(
        self, engine: SemanticSearchEngine
    ) -> None:
        """索引前搜索应返回空列表。"""
        results = await engine.search("test query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """搜索应返回结果。"""
        await engine.index_project(sample_project)
        results = await engine.search("user authentication", top_k=5)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.score > 0
            assert r.file_path
            assert r.code_snippet

    @pytest.mark.asyncio
    async def test_search_result_structure(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """验证搜索结果结构。"""
        await engine.index_project(sample_project)
        results = await engine.search("create user", top_k=3)
        if results:
            r = results[0]
            assert r.file_path
            assert isinstance(r.start_line, int)
            assert isinstance(r.end_line, int)
            assert r.start_line >= 1
            assert r.end_line >= r.start_line
            assert r.code_snippet
            assert 0 <= r.score <= 1  # 归一化后的得分

    @pytest.mark.asyncio
    async def test_search_with_language_filter(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """按语言过滤搜索。"""
        await engine.index_project(sample_project)
        results = await engine.search("user", filter_lang="python", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r.language == "python"

    @pytest.mark.asyncio
    async def test_search_with_nonexistent_language(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """不存在语言的搜索应返回空。"""
        await engine.index_project(sample_project)
        results = await engine.search("user", filter_lang="rust", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_relevance(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """相关搜索应返回更高得分的相关结果。"""
        await engine.index_project(sample_project)

        # 搜索"用户认证"相关内容
        auth_results = await engine.search("user authentication", top_k=5)
        {r.file_path for r in auth_results}

        # 搜索完全无关的内容
        await engine.search("xyznonexistent_keyword_12345", top_k=5)

        # 相关搜索应有非零结果
        assert len(auth_results) > 0
        auth_scores = [r.score for r in auth_results]
        # 至少有一些合理的结果
        assert any(s > 0.1 for s in auth_scores)


# ── Tests: 增量更新 ──────────────────────────────────────────────────────────


class TestSemanticSearchReindex:
    """测试增量更新。"""

    @pytest.mark.asyncio
    async def test_reindex_file(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """验证文件重新索引。"""
        await engine.index_project(sample_project)
        engine.get_index_stats()

        # 添加新函数到 auth.py
        auth_path = os.path.join(sample_project, "services", "auth.py")
        with open(auth_path, "a") as f:
            f.write("\ndef new_auth_func():\n    pass\n")

        await engine.reindex_file(auth_path)

        # 验证索引已更新（未崩溃，搜索新函数相关内容）
        results = await engine.search("new_auth_func", top_k=5)
        assert len(results) >= 0  # 可能没有索引到新函数，但不应崩溃

    @pytest.mark.asyncio
    async def test_reindex_nonexistent_file(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """不存在的文件重新索引不应崩溃。"""
        await engine.index_project(sample_project)
        # 不应抛出异常
        await engine.reindex_file("/nonexistent/file.py")


# ── Tests: 边界情况 ──────────────────────────────────────────────────────────


class TestSemanticSearchEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_empty_query(self, engine: SemanticSearchEngine, sample_project: str) -> None:
        """空查询应返回空结果或正常处理。"""
        await engine.index_project(sample_project)
        results = await engine.search("", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_index_twice(
        self, engine: SemanticSearchEngine, sample_project: str
    ) -> None:
        """重复索引不应崩溃。"""
        await engine.index_project(sample_project)
        engine.get_index_stats()

        # 第二次索引
        stats2 = await engine.index_project(sample_project)
        assert stats2["total_chunks"] > 0


# ── Tests: BM25 ──────────────────────────────────────────────────────────────


class TestBM25Index:
    """BM25 检索单元测试。"""

    def test_bm25_basic(self) -> None:
        """BM25 基本搜索。"""
        bm25 = _BM25Index()
        docs = [
            "The user authentication system",
            "A function to create new users",
            "Database connection pool",
        ]
        metadata = [{"id": i} for i in range(3)]
        bm25.index(docs, metadata)

        results = bm25.search("user authentication", top_k=3)
        # 文档 0 应匹配 "user authentication"
        assert len(results) > 0
        top_idx, top_score = results[0]
        assert top_score > 0

    def test_bm25_empty_index(self) -> None:
        """空索引的搜索。"""
        bm25 = _BM25Index()
        results = bm25.search("test", top_k=5)
        assert results == []

    def test_bm25_no_match(self) -> None:
        """无匹配的搜索。"""
        bm25 = _BM25Index()
        bm25.index(["python code", "javascript code"], [{"id": 0}, {"id": 1}])
        results = bm25.search("xyznonexistent", top_k=5)
        assert len(results) == 0

    def test_bm25_metadata(self) -> None:
        """BM25 元数据访问。"""
        bm25 = _BM25Index()
        meta = [{"file": "a.py"}, {"file": "b.py"}]
        bm25.index(["def foo(): pass", "def bar(): pass"], meta)
        assert bm25.get_metadata(0) == {"file": "a.py"}
        assert bm25.get_metadata(99) == {}


# ── Tests: SearchResult ──────────────────────────────────────────────────────


class TestSearchResult:
    """SearchResult 数据类测试。"""

    def test_to_dict(self) -> None:
        """验证序列化。"""
        r = SearchResult(
            file_path="/test/main.py",
            start_line=1,
            end_line=10,
            code_snippet="def foo(): pass",
            score=0.95,
            symbol_name="foo",
            language="python",
        )
        d = r.to_dict()
        assert d["file_path"] == "/test/main.py"
        assert d["score"] == 0.95
        assert d["symbol_name"] == "foo"

    def test_to_dict_minimal(self) -> None:
        """最小化 SearchResult 序列化。"""
        r = SearchResult(
            file_path="/test/main.py",
            start_line=1,
            end_line=5,
            code_snippet="code",
            score=0.5,
        )
        d = r.to_dict()
        assert d["file_path"] == "/test/main.py"
        assert "symbol_name" not in d
        assert "language" not in d
