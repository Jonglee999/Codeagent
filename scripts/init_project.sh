#!/usr/bin/env bash
# ============================================================
# CodeAgent 项目初始化脚本 (Mac / Linux)
# 功能：创建完整目录结构 + 虚拟环境 + 安装依赖 + Git 初始化
# 用法：cd codeagent/ && bash scripts/init_project.sh
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  CodeAgent 项目初始化"
echo "  路径: $PROJECT_ROOT"
echo "============================================"

# ─── 1. 创建目录结构 (§10.6) ──────────────────────────────
echo ""
echo "[1/8] 创建项目目录结构 ..."

# 主源码包
mkdir -p codeagent/gateway
mkdir -p codeagent/context_engine
mkdir -p codeagent/orchestration/nodes
mkdir -p codeagent/tools/file
mkdir -p codeagent/tools/terminal
mkdir -p codeagent/tools/lsp
mkdir -p codeagent/tools/search
mkdir -p codeagent/tools/git
mkdir -p codeagent/validation
mkdir -p codeagent/interaction/cli
mkdir -p codeagent/interaction/api

# 测试目录
mkdir -p tests/unit/gateway
mkdir -p tests/unit/context_engine
mkdir -p tests/unit/orchestration/nodes
mkdir -p tests/unit/tools/file
mkdir -p tests/unit/tools/terminal
mkdir -p tests/unit/tools/lsp
mkdir -p tests/unit/validation
mkdir -p tests/unit/interaction/cli
mkdir -p tests/unit/interaction/api
mkdir -p tests/integration/gateway
mkdir -p tests/integration/context_engine
mkdir -p tests/integration/orchestration/nodes
mkdir -p tests/integration/tools
mkdir -p tests/integration/validation
mkdir -p tests/integration/interaction/api
mkdir -p tests/e2e
mkdir -p tests/fixtures/sample_python_project
mkdir -p tests/fixtures/sample_typescript_project
mkdir -p tests/fixtures/sample_broken_project
mkdir -p tests/fixtures/sample_empty_project

# CI / 工作目录
mkdir -p .github/workflows
mkdir -p .codeagent/cache
mkdir -p .codeagent/backups

echo "   ✓ 目录结构创建完成"

# ─── 2. 创建 __init__.py ──────────────────────────────────
echo ""
echo "[2/8] 创建 __init__.py 文件 ..."

cat > codeagent/__init__.py << 'EOF'
"""CodeAgent — AI-powered coding assistant."""
EOF

cat > codeagent/gateway/__init__.py << 'EOF'
"""Gateway 抽象接口层 — 定义核心模块契约。"""
EOF

cat > codeagent/context_engine/__init__.py << 'EOF'
"""上下文引擎 — 项目扫描、语义搜索、上下文组装。"""
EOF

cat > codeagent/orchestration/__init__.py << 'EOF'
"""编排核心 — Agent 工作流定义与状态管理。"""
EOF

cat > codeagent/orchestration/nodes/__init__.py << 'EOF'
"""工作流节点 — 各步骤的独立执行单元。"""
EOF

cat > codeagent/tools/__init__.py << 'EOF'
"""工具系统 — 工具注册中心、基类、Gateway。"""
EOF

cat > codeagent/tools/file/__init__.py << 'EOF'
"""文件读写工具。"""
EOF

cat > codeagent/tools/terminal/__init__.py << 'EOF'
"""终端执行工具（含 Docker 沙箱）。"""
EOF

cat > codeagent/tools/lsp/__init__.py << 'EOF'
"""LSP 语言服务协议客户端。"""
EOF

cat > codeagent/tools/search/__init__.py << 'EOF'
"""代码搜索与 Web 搜索工具。"""
EOF

cat > codeagent/tools/git/__init__.py << 'EOF'
"""Git 操作工具。"""
EOF

cat > codeagent/validation/__init__.py << 'EOF'
"""验证闭环 — 语法检查、静态分析、运行时验证。"""
EOF

cat > codeagent/interaction/__init__.py << 'EOF'
"""用户交互层 — CLI、API、Web UI。"""
EOF

cat > codeagent/interaction/cli/__init__.py << 'EOF'
"""CLI 命令行界面。"""
EOF

cat > codeagent/interaction/api/__init__.py << 'EOF'
"""REST API 与 WebSocket 服务。"""
EOF

cat > tests/__init__.py << 'EOF'
"""CodeAgent 测试套件。"""
EOF

cat > tests/conftest.py << 'EOF'
"""pytest 共享 fixtures。"""
import pytest
EOF

echo "   ✓ __init__.py 创建完成"

# ─── 3. 创建 pyproject.toml ────────────────────────────────
echo ""
echo "[3/8] 创建 pyproject.toml ..."

cat > pyproject.toml << 'PYEOF'
[project]
name = "codeagent"
version = "0.1.0"
description = "AI-powered coding assistant with multi-agent orchestration"
requires-python = ">=3.12"
dependencies = [
    "langgraph>=1.1.0",
    "litellm>=1.83.0",
    "lancedb>=0.10.0",
    "sentence-transformers>=5.4.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "tiktoken>=0.7.0",
    "networkx>=3.3",
    "structlog>=24.0.0",
    "pydantic>=2.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=1.3.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.15.0",
    "mypy>=2.0.0",
]
sandbox = [
    "tree-sitter>=0.25.0",
    "docker>=7.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short --strict-markers"
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true
PYEOF

echo "   ✓ pyproject.toml 创建完成"

# ─── 4. 创建 README.md ─────────────────────────────────────
echo ""
echo "[4/8] 创建 README.md ..."

cat > README.md << 'EOF'
# CodeAgent

AI-powered coding assistant with multi-agent orchestration.

## Quick Start

```bash
# Create virtual environment and install dependencies
uv venv
uv sync

# Install dev dependencies
uv sync --group dev

# Verify setup
python -m pytest tests/ -x -q
```

## Project Structure

See [SRS §10.6](devdoc/SRS.html) for full directory layout.
EOF

echo "   ✓ README.md 创建完成"

# ─── 5. 创建 .env.example ──────────────────────────────────
echo ""
echo "[5/8] 创建 .env.example ..."

cat > .env.example << 'EOF'
# === LLM 配置 ===
LLM_API_KEY=your_api_key_here
LLM_MODEL=deepseek/deepseek-v4-flash
LLM_TIMEOUT=60

# === 嵌入模型配置 ===
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# === Agent 行为配置 ===
MAX_RETRIES=3
CONTEXT_BUDGET_TOKENS=8000
AUTO_MODE=false

# === Docker 沙箱配置（Phase 1b）===
SANDBOX_MEMORY_MB=512
SANDBOX_TIMEOUT_SECONDS=60

# === 日志配置 ===
LOG_LEVEL=INFO
EOF

echo "   ✓ .env.example 创建完成"

# ─── 6. 创建 .gitignore ────────────────────────────────────
echo ""
echo "[6/8] 创建 .gitignore ..."

cat > .gitignore << 'EOF'
# ─── Python ────────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg
.venv/
venv/
*.so

# ─── Node / JavaScript ─────────────────────────────────
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# ─── Docker ────────────────────────────────────────────
.docker/
*.container
docker-compose.override.yml

# ─── IDE / Editor ──────────────────────────────────────
.vscode/settings.json
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# ─── Environment ───────────────────────────────────────
.env
.env.local
*.key
*.pem

# ─── Project-specific ──────────────────────────────────
.codeagent/cache/
.codeagent/backups/
dist/

# ─── Coverage / Reports ───────────────────────────────
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover
EOF

echo "   ✓ .gitignore 创建完成"

# ─── 7. 虚拟环境 + 依赖安装 ──────────────────────────────
echo ""
echo "[7/8] 初始化 Python 虚拟环境并安装依赖 ..."

echo "   → uv venv"
uv venv

echo "   → uv sync（运行依赖）"
uv sync

echo "   → uv sync --group dev（开发依赖）"
uv sync --group dev

echo "   ✓ 依赖安装完成"

# ─── 8. Git 初始化 ────────────────────────────────────────
echo ""
echo "[8/8] 初始化 Git 仓库 ..."

if [ ! -d ".git" ]; then
    git init
    echo "   ✓ git init 完成"
else
    echo "   - Git 仓库已存在，跳过"
fi

# ─── 验证清单 ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ✅ 项目初始化完成！验证清单"
echo "============================================"
echo ""

# Python 版本
if command -v python &> /dev/null; then
    echo "  [1] Python 版本: $(python --version 2>&1)"
else
    echo "  [1] Python 版本: ❌ 未找到 python 命令"
fi

# uv 版本
if command -v uv &> /dev/null; then
    echo "  [2] uv 版本:     $(uv --version 2>&1)"
else
    echo "  [2] uv 版本:     ❌ 未安装 uv"
fi

# 虚拟环境
if [ -d ".venv" ]; then
    VENV_PYTHON=".venv/bin/python"
    if [ ! -f "$VENV_PYTHON" ]; then
        VENV_PYTHON=".venv/Scripts/python"
    fi
    if [ -f "$VENV_PYTHON" ]; then
        echo "  [3] 虚拟环境:   ✓ $("$VENV_PYTHON" --version 2>&1)"
    else
        echo "  [3] 虚拟环境:   ⚠ .venv 目录存在但 python 未找到"
    fi
else
    echo "  [3] 虚拟环境:   ❌ 未找到 .venv"
fi

# 依赖安装状态
if [ -f ".venv/bin/python" ]; then
    VENV_PIP=".venv/bin/python -m pip"
elif [ -f ".venv/Scripts/python" ]; then
    VENV_PIP=".venv/Scripts/python -m pip"
else
    VENV_PIP=""
fi

if [ -n "$VENV_PIP" ]; then
    INSTALLED=$($VENV_PIP list --format=columns 2>/dev/null | grep -c -E "^(pytest|click|rich|pydantic|litellm|langgraph|structlog)" || true)
    echo "  [4] 核心依赖:   ✓ $INSTALLED 个关键包已安装"
else
    echo "  [4] 核心依赖:   ⚠ 跳过检查"
fi

# 目录结构
DIR_COUNT=$(find codeagent -type d 2>/dev/null | wc -l)
echo "  [5] 源码目录:   $DIR_COUNT 个子目录"

# .codeagent 工作目录
if [ -d ".codeagent" ]; then
    echo "  [6] .codeagent:  ✓ 工作目录已创建"
else
    echo "  [6] .codeagent:  ❌ 未创建"
fi

# Git 状态
if [ -d ".git" ]; then
    echo "  [7] Git 仓库:    ✓ $(git rev-parse --git-dir 2>&1)"
    echo "  [8] Git 状态:    $(git status --short 2>/dev/null | head -5 | wc -l) 个未跟踪文件"
else
    echo "  [7] Git 仓库:    ❌ 未初始化"
fi

# .env.example
if [ -f ".env.example" ]; then
    echo "  [9] .env.example: ✓ 已创建"
else
    echo "  [9] .env.example: ❌ 未创建"
fi

# .gitignore
if [ -f ".gitignore" ]; then
    GITIGNORE_LINES=$(wc -l < .gitignore)
    echo "  [10] .gitignore:  ✓ $GITIGNORE_LINES 行规则"
else
    echo "  [10] .gitignore:  ❌ 未创建"
fi

echo ""
echo "============================================"
echo "  下一步: 配置环境变量"
echo ""
echo "  $ cp .env.example .env"
echo "  # 编辑 .env，填入 LLM_API_KEY"
echo "  $ pytest -x -q    # 验证环境"
echo "============================================"
echo ""
