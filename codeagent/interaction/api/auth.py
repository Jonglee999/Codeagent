"""API Key 认证 — FastAPI Security Dependency。

从 API_KEYS 环境变量加载合法 key 列表，支持多 key 轮换。
开发模式（未配置 API_KEYS）时放行所有请求。
"""

from __future__ import annotations

import logging
import os

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer(auto_error=False)


def _load_api_keys() -> set[str]:
    """从环境变量加载合法 API Key 列表。

    API_KEYS 格式：逗号分隔的 key 列表
    例如：API_KEYS=sk-key1,sk-key2,sk-key3
    """
    keys_str = os.environ.get("API_KEYS", "")
    if not keys_str:
        return set()
    return set(k.strip() for k in keys_str.split(",") if k.strip())


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security_scheme),
) -> str:
    """验证 API Key。

    返回 api_key（成功时）或抛出 HTTPException。

    - 未配置 API_KEYS 时返回 "dev-mode"（开发模式放行）
    - 无认证信息时返回 401
    - API Key 无效时返回 403
    """
    api_keys = _load_api_keys()

    # 开发模式：未配置 key 时放行
    if not api_keys:
        return "dev-mode"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials not in api_keys:
        # 记录认证失败（含 IP，不含 key 值）
        logger.warning(
            "API Key authentication failed (IP: %s)",
            getattr(credentials, "_request_ip", "unknown"),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )

    return credentials.credentials
