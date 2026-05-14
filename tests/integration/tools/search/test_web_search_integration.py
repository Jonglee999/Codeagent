"""WebSearchTool 集成测试 — 需要网络环境。

覆盖场景：
- 真实搜索（验证搜索结果格式和完整性）
- 域名过滤（验证 source 参数行为）

注意：这些测试需要网络访问 DuckDuckGo。
默认跳过，设置环境变量 CODAGENT_TEST_NETWORK=1 启用：
  CODAGENT_TEST_NETWORK=1 pytest tests/integration/tools/search/ -v
"""

from __future__ import annotations

import os

import pytest

requires_network = pytest.mark.skipif(
    not os.environ.get("CODAGENT_TEST_NETWORK"),
    reason="Set CODAGENT_TEST_NETWORK=1 to enable network-dependent integration tests",
)

from codeagent.tools.search.web_search import WebSearchTool


@pytest.fixture
def tool() -> WebSearchTool:
    """创建一个 WebSearchTool 实例（使用真实 HTTP 请求）。"""
    return WebSearchTool()


# =============================================================================
# 测试：真实搜索
# =============================================================================


@pytest.mark.asyncio
@requires_network
class TestRealSearch:
    """真实网络搜索测试。"""

    async def test_basic_search(self, tool: WebSearchTool) -> None:
        """基本搜索应返回结果。"""
        result = await tool.execute(query="python logging")
        assert result.success is True
        data = result.data

        assert "results" in data
        assert len(data["results"]) > 0

        # 验证结果格式
        r = data["results"][0]
        assert "title" in r
        assert "url" in r
        assert "snippet" in r
        assert r["source"]

    async def test_result_completeness(self, tool: WebSearchTool) -> None:
        """搜索返回的结果应包含必要的字段。"""
        result = await tool.execute(query="fastapi tutorial", max_results=3)
        assert result.success is True
        data = result.data

        assert data["total_results"] <= 3
        assert data["cached"] is False

        for r in data["results"]:
            assert isinstance(r["title"], str) and len(r["title"]) > 0
            assert r["url"].startswith("http")
            assert isinstance(r["snippet"], str)
            assert isinstance(r["source"], str)


# =============================================================================
# 测试：域名过滤
# =============================================================================


@pytest.mark.asyncio
@requires_network
class TestDomainFilterIntegration:
    """域名过滤集成测试。"""

    async def test_restricted_domain_search(self, tool: WebSearchTool) -> None:
        """限定域名搜索应返回该域名的结果。"""
        result = await tool.execute(
            query="python",
            source="docs.python.org",
            max_results=3,
        )
        assert result.success is True
        data = result.data

        if data["results"]:
            for r in data["results"]:
                assert "python.org" in r["source"], (
                    f"Expected domain to contain python.org, got {r['source']}"
                )

    async def test_domain_filter_behavior(self, tool: WebSearchTool) -> None:
        """domain 过滤参数的行为验证。"""
        # 不限域名应返回多种来源
        result_all = await tool.execute(query="python package", source="web", max_results=5)
        assert result_all.success is True

        # 限定 MDN 域名
        result_mdn = await tool.execute(
            query="javascript array methods",
            source="developer.mozilla.org",
            max_results=3,
        )
        assert result_mdn.success is True
        if result_mdn.data["results"]:
            for r in result_mdn.data["results"]:
                assert "mozilla" in r["source"], (
                    f"Expected domain to contain mozilla, got {r['source']}"
                )
