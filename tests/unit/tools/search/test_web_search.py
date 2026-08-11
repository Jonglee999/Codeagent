"""WebSearchTool 单元测试。

Mock 搜索后端，测试搜索逻辑、缓存、域名过滤、错误处理和边界情况。
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeagent.tools.search.web_search import (
    _extract_domain,
    _parse_bing_html,
    _parse_ddg_html,
    WebSearchTool,
)


# =============================================================================
# 测试辅助
# =============================================================================


def _make_html_results(count: int = 3, domain: str = "example.com") -> str:
    """构造模拟 DDG HTML 搜索结果页面。"""
    links = ""
    snippets = ""
    for i in range(count):
        url = f"https://{domain}/page{i}"
        title = f"Result {i} Title"
        snippet = f"This is the snippet for result {i}."

        links += (
            f'<a class="result__a" href="{url}">{title}</a>\n'
        )
        snippets += (
            f'<a class="result__snippet">{snippet}</a>\n'
        )

    return f"<html><body>{links}{snippets}</body></html>"


def _make_mock_httpx_client(
    html: str | None = None,
    status: int = 200,
    side_effect: Exception | None = None,
    api_json: dict | None = None,
) -> MagicMock:
    """创建模拟的 httpx.AsyncClient（在上下文管理器中）。

    支持 post（DDG HTML）和 get（DDG API）两种调用方式。
    """
    html_response = MagicMock()
    html_response.text = html or _make_html_results()
    html_response.status_code = status

    api_response = MagicMock()
    api_response.json.return_value = api_json or {"RelatedTopics": []}

    mock_client = AsyncMock()

    if side_effect:
        mock_client.post.side_effect = side_effect
        mock_client.get.side_effect = side_effect
    else:
        mock_client.post.return_value = html_response
        mock_client.get.return_value = api_response

    # 支持 async with
    mock_cm = MagicMock()
    mock_cm.__aenter__.return_value = mock_client
    mock_cm.__aexit__.return_value = None

    return mock_cm


def _make_bing_html_results(count: int = 3, domain: str = "example.com") -> str:
    """构造模拟 Bing HTML 搜索结果页面。"""

    items = ""
    for i in range(count):
        url = f"https://{domain}/page{i}"
        title = f"Result {i} Title"
        snippet = f"This is the snippet for result {i}."

        # 构造 Bing 追踪 URL（base64 编码真实 URL）
        encoded = base64.b64encode(url.encode()).decode()
        bing_tracking_url = (
            f"https://www.bing.com/ck/a?!&"
            f"u=a1{encoded}"
            f"&ntb=1"
        )

        items += (
            f'<li class="b_algo" data-id="SERP.{i}">'
            f'<h2><a href="{bing_tracking_url}">{title}</a></h2>'
            f'<div class="b_caption"><p>{snippet}</p></div>'
            f'</li>\n'
        )

    return f'<ol id="b_results">{items}</ol>'


# 禁用 DDGS 和 primp，强制走 httpx 回退以便 mock
DDGS_PATCH = patch("codeagent.tools.search.web_search.HAS_DDGS", False)
PRIMP_PATCH = patch("codeagent.tools.search.web_search.HAS_PRIMP", False)


def apply_patches(func):
    """装饰器：同时应用 DDGS 和 primp 禁用补丁。"""
    return PRIMP_PATCH(DDGS_PATCH(func))


# =============================================================================
# 测试：_extract_domain 辅助函数
# =============================================================================


class TestExtractDomain:
    """验证域名提取函数。"""

    def test_normal_url(self) -> None:
        assert _extract_domain("https://docs.python.org/3/library") == "docs.python.org"

    def test_with_www(self) -> None:
        assert _extract_domain("https://www.github.com/user/repo") == "github.com"

    def test_no_scheme(self) -> None:
        assert _extract_domain("docs.python.org/guide") == "docs.python.org"


# =============================================================================
# 测试：_parse_ddg_html 辅助函数
# =============================================================================


class TestParseDdgHtml:
    """验证 DDG HTML 解析函数。"""

    def test_parse_normal(self) -> None:
        html = _make_html_results(count=2, domain="docs.python.org")
        results = _parse_ddg_html(html, max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Result 0 Title"
        assert results[1]["source"] == "docs.python.org"

    def test_parse_empty(self) -> None:
        results = _parse_ddg_html("<html></html>", max_results=5)
        assert results == []

    def test_parse_max_results(self) -> None:
        html = _make_html_results(count=10, domain="github.com")
        results = _parse_ddg_html(html, max_results=3)
        assert len(results) == 3


# =============================================================================
# 测试：_parse_bing_html 辅助函数
# =============================================================================


class TestParseBingHtml:
    """验证 Bing HTML 解析函数。"""

    def test_parse_normal(self) -> None:
        html = _make_bing_html_results(count=2, domain="docs.python.org")
        results = _parse_bing_html(html, max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Result 0 Title"
        assert results[1]["source"] == "docs.python.org"
        assert results[0]["url"] == "https://docs.python.org/page0"

    def test_parse_empty(self) -> None:
        results = _parse_bing_html("<html></html>", max_results=5)
        assert results == []

    def test_parse_max_results(self) -> None:
        html = _make_bing_html_results(count=10, domain="github.com")
        results = _parse_bing_html(html, max_results=3)
        assert len(results) == 3

    def test_parse_url_decoded(self) -> None:
        """验证 Bing 追踪 URL 被正确解码为真实 URL。"""
        html = _make_bing_html_results(count=1, domain="docs.python.org")
        results = _parse_bing_html(html, max_results=5)
        assert len(results) == 1
        assert "bing.com" not in results[0]["url"]
        assert results[0]["url"] == "https://docs.python.org/page0"

    def test_parse_www_stripped(self) -> None:
        """验证 www 前缀被正确剥离。"""
        html = _make_bing_html_results(count=1, domain="www.example.com")
        results = _parse_bing_html(html, max_results=5)
        assert results[0]["source"] == "example.com"


# =============================================================================
# 测试：基本搜索
# =============================================================================


@pytest.mark.asyncio
class TestBasicSearch:
    """基本搜索功能。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_normal_search(self, mock_httpx: MagicMock) -> None:
        """正常搜索应返回正确格式的结果。"""
        html = _make_html_results(count=3, domain="docs.python.org")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="python logging")

        assert result.success is True
        data = result.data
        assert data["cached"] is False
        assert len(data["results"]) == 3
        assert data["total_results"] == 3

        r = data["results"][0]
        assert "title" in r
        assert "url" in r
        assert "snippet" in r
        assert "source" in r
        assert r["source"] == "docs.python.org"

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_empty_results(self, mock_httpx: MagicMock) -> None:
        """搜索无结果时应返回空列表。"""
        html = _make_html_results(count=0)
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="xyznonexistent12345")

        assert result.success is True
        assert result.data["total_results"] == 0
        assert result.data["results"] == []
        assert result.data["cached"] is False

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_result_count_respected(self, mock_httpx: MagicMock) -> None:
        """max_results 参数应限制返回结果数。"""
        html = _make_html_results(count=10, domain="github.com")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="test", max_results=3)

        assert result.success is True
        assert len(result.data["results"]) == 3
        assert result.data["total_results"] == 3


# =============================================================================
# 测试：缓存
# =============================================================================


@pytest.mark.asyncio
class TestCache:
    """缓存机制。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_cache_hit(self, mock_httpx: MagicMock) -> None:
        """相同查询在 TTL 内应返回缓存。"""
        html = _make_html_results(count=2, domain="docs.python.org")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()

        result1 = await tool.execute(query="python logging")
        assert result1.data["cached"] is False

        result2 = await tool.execute(query="python logging")
        assert result2.data["cached"] is True
        assert len(result2.data["results"]) == 2

        # 验证 httpx.AsyncClient.post 只被调用一次
        mock_client = mock_cm.__aenter__.return_value
        mock_client.post.assert_called_once()

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_cache_expiry(self, mock_httpx: MagicMock) -> None:
        """缓存超过 TTL 后应失效。"""
        html1 = _make_html_results(count=1, domain="github.com")
        mock_cm = _make_mock_httpx_client(html=html1)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        tool.CACHE_TTL = 0.1

        result1 = await tool.execute(query="flask")
        assert result1.data["cached"] is False

        import asyncio
        await asyncio.sleep(0.15)

        # 第二次搜索，重新 mock
        html2 = _make_html_results(count=2, domain="github.com")
        mock_cm2 = _make_mock_httpx_client(html=html2)
        mock_httpx.return_value = mock_cm2

        result2 = await tool.execute(query="flask")
        assert result2.data["cached"] is False
        assert len(result2.data["results"]) == 2

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_clear_cache(self, mock_httpx: MagicMock) -> None:
        """clear_cache() 应清空所有缓存。"""
        html = _make_html_results(count=1)
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()

        await tool.execute(query="python")
        assert tool.cache_size == 1

        tool.clear_cache()
        assert tool.cache_size == 0

        result2 = await tool.execute(query="python")
        assert result2.data["cached"] is False


# =============================================================================
# 测试：域名过滤
# =============================================================================


@pytest.mark.asyncio
class TestDomainFilter:
    """域名过滤。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_source_docs_python(self, mock_httpx: MagicMock) -> None:
        """source=docs.python.org 应过滤出该域名结果。"""
        html = _make_html_results(count=3, domain="docs.python.org")
        html += '<a class="result__a" href="https://github.com/tool">GitHub</a>\n'
        html += '<a class="result__snippet">GitHub snippet</a>\n'

        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="python", source="docs.python.org")

        assert result.success is True
        for r in result.data["results"]:
            assert "python.org" in r["source"]

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_source_web_returns_all(self, mock_httpx: MagicMock) -> None:
        """source=web 应返回所有不限制域名。"""
        html = (
            '<a class="result__a" href="https://docs.python.org/3">Python</a>\n'
            '<a class="result__a" href="https://github.com/test">GitHub</a>\n'
            '<a class="result__snippet">Python snippet</a>\n'
            '<a class="result__snippet">GitHub snippet</a>\n'
        )
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="test", source="web")

        assert result.success is True
        assert len(result.data["results"]) == 2


# =============================================================================
# 测试：错误处理
# =============================================================================


@pytest.mark.asyncio
class TestErrorHandling:
    """错误处理。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_network_timeout(self, mock_httpx: MagicMock) -> None:
        """网络超时时应返回错误结果。"""
        import httpx

        mock_cm = _make_mock_httpx_client(
            side_effect=httpx.TimeoutException("Connection timed out"),
        )
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="python")

        assert result.success is False
        assert "超时" in result.error_message or "timed out" in result.error_message
        assert result.error_code == "SEARCH_ERROR"

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_search_exception(self, mock_httpx: MagicMock) -> None:
        """搜索异常时应返回错误结果。"""
        mock_cm = _make_mock_httpx_client(
            side_effect=RuntimeError("Unexpected search error"),
        )
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="python")

        assert result.success is False
        assert result.error_code == "SEARCH_ERROR"

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_http_error(self, mock_httpx: MagicMock) -> None:
        """HTTP 错误应返回错误。"""
        mock_cm = _make_mock_httpx_client(
            html="", status=503, side_effect=Exception("HTTP 503"),
        )
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="python")

        assert result.success is False


# =============================================================================
# 测试：边界情况
# =============================================================================


@pytest.mark.asyncio
class TestEdgeCases:
    """边界情况。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_empty_query(self, mock_httpx: MagicMock) -> None:
        """空查询应返回错误。"""
        tool = WebSearchTool()

        result = await tool.execute(query="")
        assert result.success is False
        assert result.error_code == "EMPTY_QUERY"

        result2 = await tool.execute(query="   ")
        assert result2.success is False
        assert result2.error_code == "EMPTY_QUERY"

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_special_characters(self, mock_httpx: MagicMock) -> None:
        """特殊字符查询应正常处理。"""
        html = _make_html_results(count=2, domain="stackoverflow.com")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        result = await tool.execute(query="C++ template<typename T>")

        assert result.success is True
        assert len(result.data["results"]) == 2

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_max_results_clamping(self, mock_httpx: MagicMock) -> None:
        """max_results 应在 [1, 20] 范围。"""
        html = _make_html_results(count=30, domain="github.com")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()

        result = await tool.execute(query="test", max_results=100)
        assert len(result.data["results"]) <= 20

        result2 = await tool.execute(query="test", max_results=0)
        assert len(result2.data["results"]) == 1


# =============================================================================
# 测试：缓存键隔离
# =============================================================================


@pytest.mark.asyncio
class TestCacheKeyIsolation:
    """缓存键隔离性。"""

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_different_queries_separate_cache(self, mock_httpx: MagicMock) -> None:
        """不同查询使用独立缓存条目。"""
        html = _make_html_results(count=1, domain="docs.python.org")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        await tool.execute(query="python logging")
        await tool.execute(query="flask tutorial")

        assert tool.cache_size == 2

    @apply_patches
    @patch("httpx.AsyncClient")
    async def test_same_query_different_source_separate_cache(self, mock_httpx: MagicMock) -> None:
        """相同查询不同 source 应使用独立缓存。"""
        html = _make_html_results(count=1, domain="docs.python.org")
        mock_cm = _make_mock_httpx_client(html=html)
        mock_httpx.return_value = mock_cm

        tool = WebSearchTool()
        await tool.execute(query="python", source="web")
        await tool.execute(query="python", source="docs.python.org")

        assert tool.cache_size == 2
