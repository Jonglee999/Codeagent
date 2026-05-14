#Requires -Version 7.0
# ============================================================
# CodeAgent 项目初始化脚本 (Windows PowerShell 7+)
# 功能：创建完整目录结构 + 虚拟环境 + 安装依赖 + Git 初始化
# 用法：cd codeagent\ && .\scripts\init_project.ps1
# ============================================================

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $ProjectRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  CodeAgent 项目初始化" -ForegroundColor Cyan
Write-Host "  路径: $ProjectRoot" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# ─── 1. 创建目录结构 (§10.6) ──────────────────────────────
Write-Host "`n[1/8] 创建项目目录结构 ..." -ForegroundColor Yellow

@(
    "codeagent\gateway",
    "codeagent\context_engine",
    "codeagent\orchestration\nodes",
    "codeagent\tools\file",
    "codeagent\tools\terminal",
    "codeagent\tools\lsp",
    "codeagent\tools\search",
    "codeagent\tools\git",
    "codeagent\validation",
    "codeagent\interaction\cli",
    "codeagent\interaction\api",
    "tests\unit\gateway",
    "tests\unit\context_engine",
    "tests\unit\orchestration\nodes",
    "tests\unit\tools\file",
    "tests\unit\tools\terminal",
    "tests\unit\tools\lsp",
    "tests\unit\validation",
    "tests\unit\interaction\cli",
    "tests\unit\interaction\api",
    "tests\integration\gateway",
    "tests\integration\context_engine",
    "tests\integration\orchestration\nodes",
    "tests\integration\tools",
    "tests\integration\validation",
    "tests\integration\interaction\api",
    "tests\e2e",
    "tests\fixtures\sample_python_project",
    "tests\fixtures\sample_typescript_project",
    "tests\fixtures\sample_broken_project",
    "tests\fixtures\sample_empty_project",
    ".github\workflows",
    ".codeagent\cache",
    ".codeagent\backups"
) | ForEach-Object {
    New-Item -ItemType Directory -Path $_ -Force | Out-Null
}

Write-Host "   ✓ 目录结构创建完成" -ForegroundColor Green

# ─── 2. 创建 __init__.py ──────────────────────────────────
Write-Host "`n[2/8] 创建 __init__.py 文件 ..." -ForegroundColor Yellow

$initFiles = @{
    "codeagent\__init__.py"               = '"""CodeAgent — AI-powered coding assistant."""'
    "codeagent\gateway\__init__.py"       = '"""Gateway 抽象接口层 — 定义核心模块契约。"""'
    "codeagent\context_engine\__init__.py" = '"""上下文引擎 — 项目扫描、语义搜索、上下文组装。"""'
    "codeagent\orchestration\__init__.py" = '"""编排核心 — Agent 工作流定义与状态管理。"""'
    "codeagent\orchestration\nodes\__init__.py" = '"""工作流节点 — 各步骤的独立执行单元。"""'
    "codeagent\tools\__init__.py"         = '"""工具系统 — 工具注册中心、基类、Gateway。"""'
    "codeagent\tools\file\__init__.py"    = '"""文件读写工具。"""'
    "codeagent\tools\terminal\__init__.py"= '"""终端执行工具（含 Docker 沙箱）。"""'
    "codeagent\tools\lsp\__init__.py"     = '"""LSP 语言服务协议客户端。"""'
    "codeagent\tools\search\__init__.py"  = '"""代码搜索与 Web 搜索工具。"""'
    "codeagent\tools\git\__init__.py"     = '"""Git 操作工具。"""'
    "codeagent\validation\__init__.py"    = '"""验证闭环 — 语法检查、静态分析、运行时验证。"""'
    "codeagent\interaction\__init__.py"   = '"""用户交互层 — CLI、API、Web UI。"""'
    "codeagent\interaction\cli\__init__.py" = '"""CLI 命令行界面。"""'
    "codeagent\interaction\api\__init__.py" = '"""REST API 与 WebSocket 服务。"""'
    "tests\__init__.py"                   = '"""CodeAgent 测试套件。"""'
}

foreach ($path in $initFiles.Keys) {
    $content = $initFiles[$path]
    Set-Content -Path $path -Value $content -Encoding UTF8 -NoNewline
}

# conftest.py
@'
"""pytest 共享 fixtures。"""
import pytest
'@ | Set-Content -Path "tests\conftest.py" -Encoding UTF8

Write-Host "   ✓ __init__.py 创建完成" -ForegroundColor Green

# ─── 3. 创建 pyproject.toml ────────────────────────────────
Write-Host "`n[3/8] 创建 pyproject.toml ..." -ForegroundColor Yellow

@'
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
'@ | Set-Content -Path "pyproject.toml" -Encoding UTF8

Write-Host "   ✓ pyproject.toml 创建完成" -ForegroundColor Green

# ─── 4. 创建 README.md ─────────────────────────────────────
Write-Host "`n[4/8] 创建 README.md ..." -ForegroundColor Yellow

@'
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
'@ | Set-Content -Path "README.md" -Encoding UTF8

Write-Host "   ✓ README.md 创建完成" -ForegroundColor Green

# ─── 5. 创建 .env.example ──────────────────────────────────
Write-Host "`n[5/8] 创建 .env.example ..." -ForegroundColor Yellow

@'
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
'@ | Set-Content -Path ".env.example" -Encoding UTF8

Write-Host "   ✓ .env.example 创建完成" -ForegroundColor Green

# ─── 6. 创建 .gitignore ────────────────────────────────────
Write-Host "`n[6/8] 创建 .gitignore ..." -ForegroundColor Yellow

@'
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
'@ | Set-Content -Path ".gitignore" -Encoding UTF8

Write-Host "   ✓ .gitignore 创建完成" -ForegroundColor Green

# ─── 7. 虚拟环境 + 依赖安装 ──────────────────────────────
Write-Host "`n[7/8] 初始化 Python 虚拟环境并安装依赖 ..." -ForegroundColor Yellow

Write-Host "   → uv venv"
uv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ uv venv 失败，请确保已安装 uv (https://docs.astral.sh/uv/)" -ForegroundColor Red
    exit 1
}

Write-Host "   → uv sync (运行依赖)"
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ uv sync 失败" -ForegroundColor Red
    exit 1
}

Write-Host "   → uv sync --group dev (开发依赖)"
uv sync --group dev
if ($LASTEXITCODE -ne 0) {
    Write-Host "   ❌ uv sync --group dev 失败" -ForegroundColor Red
    exit 1
}

Write-Host "   ✓ 依赖安装完成" -ForegroundColor Green

# ─── 8. Git 初始化 ────────────────────────────────────────
Write-Host "`n[8/8] 初始化 Git 仓库 ..." -ForegroundColor Yellow

if (-not (Test-Path ".git")) {
    git init
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ git init 完成" -ForegroundColor Green
    } else {
        Write-Host "   ❌ git init 失败" -ForegroundColor Red
    }
} else {
    Write-Host "   - Git 仓库已存在，跳过" -ForegroundColor Gray
}

# ─── 验证清单 ──────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ✅ 项目初始化完成！验证清单" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Python 版本
try {
    $pyVer = python --version 2>&1
    Write-Host "  [1] Python 版本:   $pyVer" -ForegroundColor Green
} catch {
    Write-Host "  [1] Python 版本:   ❌ 未找到 python 命令" -ForegroundColor Red
}

# uv 版本
try {
    $uvVer = uv --version 2>&1
    Write-Host "  [2] uv 版本:       $uvVer" -ForegroundColor Green
} catch {
    Write-Host "  [2] uv 版本:       ❌ 未安装 uv" -ForegroundColor Red
}

# 虚拟环境
if (Test-Path ".venv") {
    $venvPython = ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $vpyVer = & $venvPython --version 2>&1
        Write-Host "  [3] 虚拟环境:     ✓ $vpyVer" -ForegroundColor Green
    } else {
        Write-Host "  [3] 虚拟环境:     ⚠ .venv 目录存在但 python 未找到" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [3] 虚拟环境:     ❌ 未找到 .venv" -ForegroundColor Red
}

# 依赖安装状态
$venvPip = ".venv\Scripts\python.exe"
if (Test-Path $venvPip) {
    $corePkgs = @("pytest", "click", "rich", "pydantic", "litellm", "langgraph", "structlog")
    $installed = 0
    foreach ($pkg in $corePkgs) {
        $result = & $venvPip -m pip list --format=columns 2>$null | Select-String -Pattern "^$pkg\s" -Quiet
        if ($result) { $installed++ }
    }
    Write-Host "  [4] 核心依赖:     ✓ $installed/$($corePkgs.Count) 个关键包已安装" -ForegroundColor Green
} else {
    Write-Host "  [4] 核心依赖:     ⚠ 跳过检查" -ForegroundColor Yellow
}

# 目录结构
$dirCount = (Get-ChildItem -Path "codeagent" -Recurse -Directory).Count
Write-Host "  [5] 源码目录:     $dirCount 个子目录" -ForegroundColor Green

# .codeagent 工作目录
if (Test-Path ".codeagent") {
    Write-Host "  [6] .codeagent:    ✓ 工作目录已创建" -ForegroundColor Green
} else {
    Write-Host "  [6] .codeagent:    ❌ 未创建" -ForegroundColor Red
}

# Git 状态
if (Test-Path ".git") {
    $gitDir = git rev-parse --git-dir 2>&1
    Write-Host "  [7] Git 仓库:      ✓ $gitDir" -ForegroundColor Green
    $untracked = (git status --short 2>$null | Measure-Object | Select-Object -ExpandProperty Count)
    Write-Host "  [8] Git 状态:      $untracked 个未跟踪文件" -ForegroundColor Green
} else {
    Write-Host "  [7] Git 仓库:      ❌ 未初始化" -ForegroundColor Red
}

# .env.example
if (Test-Path ".env.example") {
    Write-Host "  [9] .env.example:  ✓ 已创建" -ForegroundColor Green
} else {
    Write-Host "  [9] .env.example:  ❌ 未创建" -ForegroundColor Red
}

# .gitignore
if (Test-Path ".gitignore") {
    $lines = (Get-Content ".gitignore" | Measure-Object -Line).Lines
    Write-Host "  [10] .gitignore:   ✓ $lines 行规则" -ForegroundColor Green
} else {
    Write-Host "  [10] .gitignore:   ❌ 未创建" -ForegroundColor Red
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  下一步: 配置环境变量" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Copy-Item .env.example .env"
Write-Host "  # 编辑 .env，填入 LLM_API_KEY"
Write-Host "  pytest -x -q    # 验证环境"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
