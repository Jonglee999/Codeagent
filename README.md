# CodeAgent

AI-powered coding assistant with multi-agent orchestration.

## Prerequisites

- **Python 3.12+**
- **Docker Desktop**（可选，用于沙箱模式）
- **LLM API Key**（DeepSeek / OpenAI / Anthropic 等）

## Installation

```bash
# 创建虚拟环境并安装依赖
uv venv
uv sync

# 安装开发依赖（测试、lint、沙箱）
uv sync --group dev

# 安装 checkpoint 持久化支持（可选）
uv sync --group persistence

# 验证安装
python -m pytest tests/ -x -q
```

## Configuration

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env，填入 API Key 和模型配置
# LLM_API_KEY=your_api_key_here
# LLM_MODEL=deepseek/deepseek-v4-flash
```

所有配置项说明见 [.env.example](.env.example)。

## Quick Start

```bash
# 初始化 CodeAgent 工作目录
codeagent init

# 执行任务：让 Agent 完成一个编码任务
codeagent ask "Create a hello.py that prints 'Hello, World!'"

# 自动模式（跳过人工审核）
codeagent ask "Refactor the project structure" --auto

# 指定模型
codeagent ask "Add unit tests for the utils module" --model openai/gpt-4o
```

## Supported Models

通过 [litellm](https://docs.litellm.ai/docs/providers) 支持以下 Provider：

| Provider | 模型示例 | LLM_MODEL 格式 |
|----------|---------|---------------|
| DeepSeek | deepseek-v4-flash | `deepseek/deepseek-v4-flash` |
| OpenAI | GPT-4o, GPT-4o-mini | `openai/gpt-4o` |
| Anthropic | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |
| Google | Gemini 2.0 Flash | `gemini/gemini-2.0-flash` |

更多 Provider 见 [litellm 文档](https://docs.litellm.ai/docs/providers)。

## Commands

| 命令 | 说明 |
|------|------|
| `codeagent ask <request>` | 执行编码任务（完整 Orchestrator 流程） |
| `codeagent init` | 初始化 CodeAgent 工作目录 |
| `codeagent config [key] [value]` | 查看或设置配置项 |
| `codeagent history [limit]` | 查看历史任务记录 |

### Ask 命令选项

| 选项 | 说明 |
|------|------|
| `--auto` | 自动模式，跳过人工审核步骤 |
| `--model TEXT` | 指定 LLM 模型（覆盖 LLM_MODEL） |
| `--max-retries INT` | 最大重试次数（覆盖 MAX_RETRIES） |
| `--verbose` | 输出详细日志 |
| `--json` | 以 JSON 格式输出结果 |
| `--no-color` | 禁用彩色输出 |

## Docker Sandbox

启用沙箱模式后，代码执行将在隔离的 Docker 容器中运行：

```bash
# 确保 Docker Desktop 已启动
# 在 .env 中启用：
SANDBOX_ENABLED=true
SANDBOX_TIMEOUT=60
SANDBOX_MEMORY_MB=512
```

未安装 Docker 时自动降级为本地执行，不会阻断工作流。

## Troubleshooting

| 问题 | 解决方法 |
|------|---------|
| `LLM_API_KEY not configured` | 复制 `.env.example` 到 `.env` 并填入 API Key |
| `Docker is not available` | 安装 Docker Desktop 或将 `SANDBOX_ENABLED` 设为 `false` |
| litellm 连接超时 | 检查网络连接，增大 `LLM_TIMEOUT`（默认 60s） |
| `ModuleNotFoundError` | 运行 `uv sync` 确保所有依赖已安装 |
| 测试集合错误 | 确保使用 Python 3.12+ 并运行 `uv sync --group dev` |

## Project Structure

See [SRS §10.6](devdoc/SRS.html) for full directory layout.
