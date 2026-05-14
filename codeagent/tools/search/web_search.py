"""WebSearchTool — 网络搜索工具。

使用 DuckDuckGo 搜索（免费，无需 API Key）。
限制搜索域名白名单（技术文档站）。
缓存搜索结果（相同查询 5 分钟内返回缓存）。

搜索后端优先级：
1. duckduckgo_search (DDGS) — 标准库，优先使用
2. primp — DDGS v8+ 的底层 HTTP 客户端，可绕过挑战页面
3. httpx — 通用回退
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlparse

from codeagent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ── 可选依赖探测 ─────────────────────────────────────────────────────────


def _import_ddgs() -> bool:
    """尝试导入 duckduckgo_search。"""
    try:
        global DDGS  # type: ignore[name-defined]
        from duckduckgo_search import DDGS  # type: ignore[no-redef]

        return True
    except ImportError:
        return False


def _import_primp() -> bool:
    """尝试导入 primp（DDGS v8+ 底层客户端）。"""
    try:
        global primp_client  # type: ignore[name-defined]
        import primp as primp_client  # type: ignore[no-redef]

        return True
    except ImportError:
        return False


HAS_DDGS = _import_ddgs()
HAS_PRIMP = _import_primp()

# ── 常量 ─────────────────────────────────────────────────────────────────

# 搜索域名白名单（仅技术文档站）
ALLOWED_DOMAINS: list[str] = [
    "docs.python.org",
    "developer.mozilla.org",
    "stackoverflow.com",
    "github.com",
    "pypi.org",
    "packaging.python.org",
    "pip.pypa.io",
    "pytest.org",
    "numpy.org",
    "pandas.pydata.org",
    "fastapi.tiangolo.com",
    "flask.palletsprojects.com",
    "docs.djangoproject.com",
    "react.dev",
    "vuejs.org",
    "angular.io",
    "kubernetes.io",
    "docker.com",
    "docker.io",
    "docs.docker.com",
]

# DDG HTML 搜索端点
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"


def _extract_domain(url: str) -> str:
    """从 URL 中提取域名。"""
    try:
        if "://" not in url:
            url = "https://" + url
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""


def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """解析 DDG HTML 搜索结果页面。

    从 <a class="result__a"> 和 <a class="result__snippet"> 中提取
    标题、URL 和摘要。不依赖外部解析库。
    """
    import html as html_mod
    import re

    results: list[dict] = []

    link_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )

    links = link_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    for i, (url, title_html) in enumerate(links):
        if i >= max_results:
            break

        title = re.sub(r"<[^>]+>", "", title_html).strip()
        title = html_mod.unescape(title)

        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url

        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
            snippet = html_mod.unescape(snippet)

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": _extract_domain(url),
        })

    return results


def _parse_bing_html(html: str, max_results: int) -> list[dict]:
    """解析 Bing HTML 搜索结果页面。

    从 <li class="b_algo"> 中提取标题、URL 和摘要。
    Bing 使用追踪 URL（u 参数 base64 编码），需解码获取真实 URL。
    """
    import base64
    import html as html_mod
    import re
    import urllib.parse

    algo_re = re.compile(
        r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>',
        re.DOTALL,
    )
    h2_re = re.compile(
        r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a></h2>',
        re.DOTALL,
    )
    p_re = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL)

    def _decode_bing_url(url: str) -> str:
        """从 Bing 追踪 URL 中提取真实 URL（base64 解码 u 参数）。"""
        url = html_mod.unescape(url)
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        u_param = params.get("u", [None])[0]
        if u_param:
            try:
                if u_param.startswith("a1"):
                    u_param = u_param[2:]
                padding = 4 - len(u_param) % 4
                if padding != 4:
                    u_param += "=" * padding
                return base64.b64decode(u_param).decode("utf-8")
            except Exception:
                pass
        return url

    results: list[dict] = []
    for algo in algo_re.findall(html):
        if len(results) >= max_results:
            break

        h2 = h2_re.search(algo)
        if not h2:
            continue

        raw_url = h2.group(1)
        url = _decode_bing_url(raw_url)
        title_html = h2.group(2)
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        title = html_mod.unescape(title)

        ps = p_re.findall(algo)
        snippet = ""
        for p in ps:
            text = re.sub(r"<[^>]+>", "", p).strip()
            text = html_mod.unescape(text)
            if text and len(text) > len(snippet):
                snippet = text

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": _extract_domain(url),
        })

    return results


class WebSearchTool(BaseTool):
    """网络搜索工具——搜索技术文档和参考资料。

    使用 DuckDuckGo 搜索（免费，无需 API Key）。
    限制搜索域名白名单（仅技术文档站）。
    缓存搜索结果（相同查询 5 分钟内返回缓存）。
    """

    name = "web_search"
    description = "搜索技术文档和参考资料（DuckDuckGo）"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "source": {
                "type": "string",
                "enum": ["web", "docs.python.org", "developer.mozilla.org"],
                "default": "web",
                "description": "搜索源或限定域名",
            },
            "max_results": {
                "type": "integer",
                "default": 5,
                "description": "最大返回结果数，默认 5",
            },
        },
        "required": ["query"],
    }

    requires_sandbox: bool = False

    CACHE_TTL: int = 300  # 5 分钟

    def __init__(self) -> None:
        """初始化 WebSearchTool。"""
        super().__init__()
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._primp_client: Any = None

    async def execute(  # type: ignore[override]
        self,
        query: str,
        source: str = "web",
        max_results: int = 5,
    ) -> ToolResult:
        """执行网络搜索。

        Args:
            query: 搜索关键词。
            source: "web"（不限域名）或限定的域名（如 "docs.python.org"）。
            max_results: 最大返回结果数。

        Returns:
            ToolResult: 搜索结果。
        """
        start_time = time.monotonic()

        if not query or not query.strip():
            return ToolResult(
                success=False,
                error_message="Query cannot be empty",
                error_code="EMPTY_QUERY",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        max_results = min(max(max_results, 1), 20)
        query = query.strip()

        # ── 缓存检查 ─────────────────────────────────────────────
        cache_key = f"{query}:::{source}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            results = cached[:max_results]
            return ToolResult(
                success=True,
                data={
                    "results": results,
                    "total_results": len(results),
                    "cached": True,
                },
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 执行搜索 ─────────────────────────────────────────────
        try:
            results = await self._search(query, max_results)
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return ToolResult(
                success=False,
                error_message=f"搜索失败: {exc}",
                error_code="SEARCH_ERROR",
                duration_ms=(time.monotonic() - start_time) * 1000,
            )

        # ── 域名过滤 ─────────────────────────────────────────────
        if source != "web":
            results = self._filter_by_domain(results, source)

        # ── 写入缓存 ─────────────────────────────────────────────
        self._set_cache(cache_key, results)
        results = results[:max_results]

        return ToolResult(
            success=True,
            data={
                "results": results,
                "total_results": len(results),
                "cached": False,
            },
            duration_ms=(time.monotonic() - start_time) * 1000,
        )

    async def _search(self, query: str, max_results: int) -> list[dict]:
        """按优先级尝试搜索后端。

        primp 优先：直接调用 DDG HTML 端点，能绕过挑战页面。
        DDGS 次之：duckduckgo_search 标准库。
        httpx 第三：通用 HTTP 回退（DDG HTML）。
        bing 第四：Bing HTML 搜索（DDG 不可用时可靠回退）。
        api 最后：DDG Instant Answer API（永不屏蔽，但只返回摘要式结果）。
        """
        errors: list[str] = []

        if HAS_PRIMP:
            try:
                results = await self._search_with_primp(query, max_results)
                if results:
                    return results
                errors.append("primp: empty results")
            except Exception as e:
                errors.append(f"primp: {e}")

        if HAS_DDGS:
            try:
                results = await self._search_with_ddgs(query, max_results)
                if results:
                    return results
                errors.append("ddgs: empty results")
            except Exception as e:
                errors.append(f"ddgs: {e}")

        try:
            results = await self._search_with_httpx(query, max_results)
            if results:
                return results
            errors.append("httpx: empty results (likely blocked)")
        except Exception as e:
            errors.append(f"httpx: {e}")

        # 回退：Bing HTML 搜索（在中国等区域 DDG 不可用时可工作）
        try:
            return await self._search_with_bing(query, max_results)
        except Exception as e:
            errors.append(f"bing: {e}")

        # 最终回退：DDG Instant Answer API（永不屏蔽）
        try:
            return await self._search_with_api(query, max_results)
        except Exception as e:
            errors.append(f"api: {e}")

        raise RuntimeError(f"All search backends failed: {'; '.join(errors)}")

    async def _search_with_ddgs(self, query: str, max_results: int) -> list[dict]:
        """使用 duckduckgo_search 库进行搜索。"""
        import asyncio

        loop = asyncio.get_event_loop()

        def _search() -> list[dict]:
            results: list[dict] = []
            with DDGS() as ddgs:
                for i, r in enumerate(
                    ddgs.text(query, region="wt-wt", safesearch="moderate")
                ):
                    if i >= max_results:
                        break
                    title = r.get("title", "")
                    href = r.get("href", "") or r.get("link", "")
                    body = r.get("body", "") or r.get("snippet", "")
                    results.append({
                        "title": title,
                        "url": href,
                        "snippet": body,
                        "source": _extract_domain(href),
                    })
            return results

        return await loop.run_in_executor(None, _search)

    async def _search_with_primp(self, query: str, max_results: int) -> list[dict]:
        """使用 primp 直接调用 DDG HTML 搜索。

        primp 是 duckduckgo_search v8+ 的底层 HTTP 客户端，
        能自动处理 DDG 的挑战页面。
        """
        if self._primp_client is None:
            self._primp_client = primp_client.Client(
                follow_redirects=True,
                timeout=10.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )

        import asyncio

        loop = asyncio.get_event_loop()

        def _search() -> list[dict]:
            resp = self._primp_client.get(
                _DDG_HTML_URL,
                params={"q": query},
            )
            return _parse_ddg_html(resp.text, max_results)

        return await loop.run_in_executor(None, _search)

    async def _search_with_httpx(self, query: str, max_results: int) -> list[dict]:
        """使用 httpx 直接调用 DDG HTML 搜索作为回退。"""
        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        ) as client:
            resp = await client.post(
                _DDG_HTML_URL,
                data={"q": query},
            )

        return _parse_ddg_html(resp.text, max_results)

    async def _search_with_bing(self, query: str, max_results: int) -> list[dict]:
        """使用 Bing HTML 搜索作为回退（在 DDG 不可用的区域可工作）。"""
        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        ) as client:
            resp = await client.get(
                "https://www.bing.com/search",
                params={"q": query, "count": min(max_results, 50)},
            )

        return _parse_bing_html(resp.text, max_results)

    async def _search_with_api(self, query: str, max_results: int) -> list[dict]:
        """使用 DDG Instant Answer API 作为最终回退。

        此 API 永不屏蔽请求，但仅返回摘要式结果（非完整网页搜索结果）。
        """
        import asyncio
        import json

        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
        ) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[dict] = []
        seen_urls: set[str] = set()

        # 从 RelatedTopics 提取结果
        topics: list[dict] = data.get("RelatedTopics", [])

        def extract(item: dict) -> None:
            if "Topics" in item:
                for sub in item["Topics"]:
                    extract(sub)
                return
            url = item.get("FirstURL", "")
            text = item.get("Text", "")
            # Text 格式通常是 "Title Description"，尝试拆分
            title = text.split("  ")[0] if "  " in text else text
            snippet = text[len(title):].strip() if "  " in text else ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({
                    "title": title[:200],
                    "url": url,
                    "snippet": snippet[:300],
                    "source": _extract_domain(url),
                })

        for topic in topics:
            if len(results) >= max_results:
                break
            extract(topic)

        return results

    def _filter_by_domain(self, results: list[dict], source: str) -> list[dict]:
        """按域名过滤搜索结果。"""
        if not results or source == "web":
            return results

        target_domain = source.lower()
        filtered = []
        for r in results:
            domain = r.get("source", "").lower()
            if domain == target_domain or target_domain in domain:
                filtered.append(r)

        return filtered

    # ── 缓存管理 ──────────────────────────────────────────────────────

    def _check_cache(self, key: str) -> list[dict] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        timestamp, results = entry
        if time.monotonic() - timestamp > self.CACHE_TTL:
            del self._cache[key]
            return None
        return results

    def _set_cache(self, key: str, results: list[dict]) -> None:
        self._cache[key] = (time.monotonic(), list(results))

    def clear_cache(self) -> None:
        """清空缓存。"""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
