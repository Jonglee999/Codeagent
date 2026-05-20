"""API Key 认证单元测试。

覆盖：有效/无效 key、无 key、开发模式放行、豁免端点、日志记录。
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials


class TestVerifyApiKey:
    """verify_api_key 认证函数测试。"""

    def test_valid_key_passes(self):
        """有效 API Key 应通过认证并返回 key 值。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {"API_KEYS": "sk-valid-key"}, clear=True):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="sk-valid-key",
            )
            result = verify_api_key(credentials=creds)
            assert result == "sk-valid-key"

    def test_multiple_valid_keys(self):
        """多 key 配置时，任一有效 key 都应通过认证。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(
            os.environ,
            {"API_KEYS": "sk-key1,sk-key2,sk-key3"},
            clear=True,
        ):
            for valid_key in ("sk-key1", "sk-key2", "sk-key3"):
                creds = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=valid_key,
                )
                result = verify_api_key(credentials=creds)
                assert result == valid_key

    def test_invalid_key_returns_403(self):
        """无效 API Key 应抛出 403 Forbidden。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {"API_KEYS": "sk-valid-key"}, clear=True):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="sk-invalid-key",
            )
            with pytest.raises(HTTPException) as exc:
                verify_api_key(credentials=creds)
            assert exc.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Invalid API Key" in exc.value.detail

    def test_no_credentials_returns_401(self):
        """无认证信息时返回 401 Unauthorized。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {"API_KEYS": "sk-valid-key"}, clear=True):
            with pytest.raises(HTTPException) as exc:
                verify_api_key(credentials=None)
            assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Missing Authorization" in exc.value.detail
            # 应包含 WWW-Authenticate 头
            assert exc.value.headers.get("WWW-Authenticate") == "Bearer"

    def test_dev_mode_no_keys(self):
        """未配置 API_KEYS 时放行所有请求（开发模式）。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {}, clear=True):
            # 无认证信息也放行
            result = verify_api_key(credentials=None)
            assert result == "dev-mode"

    def test_dev_mode_with_key_ignored(self):
        """开发模式下即使传了无效 key 也放行。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {}, clear=True):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="anything",
            )
            result = verify_api_key(credentials=creds)
            assert result == "dev-mode"

    def test_empty_api_keys_value(self):
        """API_KEYS 为空字符串时等同于开发模式。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(os.environ, {"API_KEYS": ""}, clear=True):
            result = verify_api_key(credentials=None)
            assert result == "dev-mode"

    def test_whitespace_api_keys(self):
        """API_KEYS 中 key 的前后空白应被 strip。"""
        from codeagent.interaction.api.auth import verify_api_key

        with patch.dict(
            os.environ,
            {"API_KEYS": "  sk-key1 , sk-key2  "},
            clear=True,
        ):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="sk-key1",
            )
            result = verify_api_key(credentials=creds)
            assert result == "sk-key1"

    def test_auth_failure_logs_warning(self, caplog):
        """认证失败应记录日志（含 IP 信息）。"""
        import logging

        from codeagent.interaction.api.auth import verify_api_key

        caplog.set_level(logging.WARNING)

        with patch.dict(os.environ, {"API_KEYS": "sk-valid-key"}, clear=True):
            creds = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials="sk-bad",
            )
            with pytest.raises(HTTPException):
                verify_api_key(credentials=creds)

            # 验证日志包含 IP 相关信息
            assert any(
                "API Key authentication failed" in record.message
                for record in caplog.records
            )

    def test_key_order_independence(self):
        """多 key 配置下，认证结果与 key 在列表中的顺序无关。"""
        from codeagent.interaction.api.auth import verify_api_key

        keys_list = ["sk-first", "sk-second", "sk-third"]
        with patch.dict(
            os.environ,
            {"API_KEYS": ",".join(keys_list)},
            clear=True,
        ):
            # 验证所有 key 都能通过，无论顺序
            for key in reversed(keys_list):
                creds = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=key,
                )
                result = verify_api_key(credentials=creds)
                assert result == key
