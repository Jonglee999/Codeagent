"""E2E 测试 conftest — live server + Playwright fixtures。

E2E 测试需要：
1. 运行中的 Redis 实例（默认 redis://127.0.0.1:6379）
2. LLM API Key（环境变量 LLM_API_KEY）
3. Playwright 浏览器（pip install pytest-playwright && playwright install chromium）
4. 前端构建产物（frontend/dist/）

所有 E2E 测试使用 @pytest.mark.e2e 标记，默认跳过。
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

# ── 环境检查 ──────────────────────────────────────────────────────────

_HAS_PLAYWRIGHT = False
try:
    from playwright.sync_api import Page, expect  # noqa: F401

    _HAS_PLAYWRIGHT = True
except ImportError:
    pass

_HAS_API_KEY = bool(os.environ.get("LLM_API_KEY"))
_HAS_FRONTEND_DIST = (Path(__file__).resolve().parent.parent.parent / "frontend" / "dist").exists()


# ── pytest hooks ──────────────────────────────────────────────────────


def pytest_configure(config):
    """Register e2e marker."""
    config.addinivalue_line("markers", "e2e: Playwright E2E browser tests (requires full stack)")


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests if prerequisites are not met."""
    skip_e2e = pytest.mark.skip(
        reason="E2E tests require: pytest-playwright, LLM_API_KEY, "
        "running Redis, and frontend/dist/. "
        "Install: pip install pytest-playwright && playwright install chromium"
    )
    for item in items:
        if "e2e" in item.keywords:
            if not _HAS_PLAYWRIGHT or not _HAS_API_KEY or not _HAS_FRONTEND_DIST:
                item.add_marker(skip_e2e)


# ── Live server fixture ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def live_server(request) -> str:
    """Start FastAPI backend + frontend static server for E2E testing.

    Returns the frontend URL (e.g. http://localhost:4173).

    Requires:
    - Redis running on REDIS_URL (default redis://127.0.0.1:6379)
    - Frontend built at frontend/dist/
    """
    import socket

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    project_root = Path(__file__).resolve().parent.parent.parent
    frontend_dist = project_root / "frontend" / "dist"

    api_port = _find_free_port()
    frontend_port = _find_free_port()

    api_url = f"http://127.0.0.1:{api_port}"
    frontend_url = f"http://127.0.0.1:{frontend_port}"

    processes = []

    try:
        # Start FastAPI backend
        api_proc = subprocess.Popen(
            [
                "uvicorn",
                "codeagent.interaction.api.main:app",
                "--host", "127.0.0.1",
                "--port", str(api_port),
                "--log-level", "warning",
            ],
            cwd=str(project_root),
            env={
                **os.environ,
                "REDIS_URL": os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"),
            },
        )
        processes.append(api_proc)

        # Start frontend static file server
        frontend_proc = subprocess.Popen(
            [
                "python", "-m", "http.server",
                str(frontend_port),
                "--directory", str(frontend_dist),
            ],
            cwd=str(frontend_dist),
        )
        processes.append(frontend_proc)

        # Wait for servers to be ready
        _wait_for_server(api_url, timeout=30)
        _wait_for_server(frontend_url, timeout=10)

        yield frontend_url

    finally:
        for proc in processes:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _wait_for_server(url: str, timeout: int = 30) -> None:
    """Wait until the server responds or timeout."""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    logger.warning("Server at %s did not respond within %ds", url, timeout)


# ── Playwright fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="session")
def browser_context_args():
    """Playwright browser context configuration."""
    return {
        "viewport": {"width": 1440, "height": 900},
        "ignore_https_errors": True,
    }


@pytest.fixture
def app_url(live_server: str) -> str:
    """Alias for live_server to match test expectations."""
    return live_server
