<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/github/actions/workflow/status/user/codeagent/ci.yml?label=CI" alt="CI">
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-1.1-FF6F00" alt="LangGraph">
</p>

<h1 align="center">CodeAgent</h1>

<p align="center">
  <strong>AI-powered coding assistant with multi-agent orchestration, self-evolution, and a modern Web UI.</strong>
</p>

<p align="center">
  CodeAgent is an intelligent coding assistant that understands your project context, plans and executes multi-step coding tasks, validates results, and learns from its own experience — all through a CLI terminal, a REST API, or a real-time Web dashboard.
</p>

<br>

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🧠 Multi-Agent Orchestration** | LangGraph-based pipeline: Context → Planning → Execution → Validation → Human Review, with conditional routing and rollback support |
| **📋 Plan-Aware Execution** | Automatically decomposes complex requests into step-by-step plans with dependency tracking and progress monitoring |
| **🔧 Rich Tool System** | File I/O, code search, Git operations, terminal commands, LSP diagnostics — extensible via a plugin-like registry |
| **✅ Automated Validation** | Three-layer validation (syntax → static analysis → runtime) with auto-repair loop when errors are detected |
| **🧪 Docker Sandbox** | Isolated code execution in Docker containers with configurable resource limits; auto-degrades to local execution when Docker is unavailable |
| **🧠 Memory System** | Persistent multi-type memory (user, project, session, feedback) with vector-based semantic retrieval, auto-extraction, and confidence decay |
| **🔄 Self-Evolution** | Trajectory recording, strategy extraction, and strategy application — the agent learns from past tasks and improves over time |
| **🕵️ Deviation Detection** | Monitors tool calls against the original plan, escalates repeated deviations through warnings to human review |
| **💬 Human-in-the-Loop** | Configurable human review gates for high-risk operations, deviation escalations, and cost-limit approvals |
| **🌐 Modern Web UI** | Real-time React dashboard with WebSocket streaming, progress visualization, and interactive task management |
| **⚡ Async Architecture** | FastAPI + Celery + Redis for non-blocking task execution; supports both Celery and Inline modes |
| **🔁 Rollback Support** | Tracks file modifications and supports partial or full rollback to previous states |
| **☁️ Docker Deployment** | Multi-stage Dockerfile + docker-compose for one-command production deployment with Nginx reverse proxy |
| **📊 Cost Control** | Configurable limits on LLM calls per task and token consumption to prevent runaway costs |
| **🔌 Multi-Provider LLM** | Supports 200+ LLM providers via [litellm](https://litellm.ai) — DeepSeek, OpenAI, Anthropic, Google, and more |

<br>

## 🏗️ Architecture

CodeAgent uses a **gateway-based microkernel architecture** built on LangGraph's state machine:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                              │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   CLI    │  │  REST API / WS   │  │       Web UI (React)     │  │
│  └────┬─────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
└───────┼─────────────────┼──────────────────────────┼────────────────┘
        │                 │                          │
┌───────┴─────────────────┴──────────────────────────┴────────────────┐
│                    Orchestration Layer                               │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    LangGraph StateGraph                        │  │
│  │                                                               │  │
│  │  Context ──▶ Planning ──▶ Execution ──▶ Validation ──▶ Done  │  │
│  │    ▲              │              │              │              │  │
│  │    │              ▼              ▼              ▼              │  │
│  │    └────────── Re-plan ──── Repair Loop ── Human Review        │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Context  │  Tool    │Validation│ Memory   │Evolution │ Sandbox  │
│ Engine   │ Gateway  │ Gateway  │ Gateway  │ Manager  │  (Docker)│
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ FileTree │ Registry │ Syntax   │ Store    │Trajectory│ Container│
│ CodeAna  │  File    │ Static   │Retriever │ Strategy │ Executor │
│ Semantic │  Search  │ Runtime  │Extractor │ Applier  │ Resource │
│ Assembly │  Git/LSP │ Auto-    │ Confidence│ Decay   │  Limits  │
│          │  Terminal│ repair   │          │          │          │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### Orchestrator Workflow

The core workflow is a LangGraph `StateGraph` with these nodes:

1. **Context Node** — Scans the project file tree, analyzes code structure, builds dependency graphs, and performs semantic search to gather relevant context into a `ContextPackage`
2. **Planning Node** — Analyzes the user request with the gathered context, decomposes it into a sequence of `PlanStep` items (create, modify, delete, read, command)
3. **Execution Node** — Executes plan steps via LLM tool-calling loops; supports **plan-aware execution**, **direct mode** (Phase 1a compatible), and **auto-repair mode** when validation fails
4. **Validation Node** — Runs three validation layers (syntax → static analysis → runtime), detects test files, and analyzes errors with structured repair context
5. **Human Review Node** — Pauses execution for human decisions: approve, abort, or modify; triggered by high-risk plans, deviation escalations, or cost limits

The graph includes **conditional routing** with loops (re-plan, repair) and **interrupt-based human review gates**.

<br>

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker Desktop** (optional, for sandbox mode)
- **LLM API Key** (DeepSeek / OpenAI / Anthropic / etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/codeagent.git
cd codeagent

# Create virtual environment and install dependencies
uv venv
uv sync

# Install dev dependencies (testing, linting, sandbox)
uv sync --group dev

# Install checkpoint persistence support (optional)
uv sync --group persistence

# Verify installation
python -m pytest tests/ -x -q
```

### Configuration

```bash
# Copy the example configuration file
cp .env.example .env

# Edit .env with your API key and model settings
# LLM_API_KEY=your_api_key_here
# LLM_MODEL=deepseek/deepseek-v4-flash
```

All configuration options are documented in [.env.example](.env.example).

### Run Your First Task

```bash
# Initialize CodeAgent working directory
codeagent init

# Execute a task — the agent plans, executes, validates, and reports
codeagent ask "Create a hello.py that prints 'Hello, World!'"

# Auto mode (skip human review)
codeagent ask "Refactor the project structure" --auto

# Specify a different model
codeagent ask "Add unit tests for the utils module" --model openai/gpt-4o

# Verbose mode (show every tool call)
codeagent ask "Create a Flask app with /health endpoint" --verbose

# JSON output (for script consumption)
codeagent ask "Analyze the codebase" --json
```

<br>

## 📖 Usage

### CLI Commands

| Command | Description |
|---------|-------------|
| `codeagent ask <request>` | Execute a coding task through the full Orchestrator pipeline |
| `codeagent init` | Initialize `.codeagent/` working directory |
| `codeagent config [--show] [--set KEY VALUE]` | View or modify configuration |
| `codeagent history [--limit N]` | View recent task history |

#### Ask Command Options

| Option | Description |
|--------|-------------|
| `--auto` | Auto mode, skip human review steps |
| `--model TEXT` | Specify LLM model (overrides `LLM_MODEL`) |
| `--max-retries INT` | Maximum retry count (overrides `MAX_RETRIES`) |
| `--max-llm-calls INT` | Maximum LLM calls per task (overrides `MAX_LLM_CALLS_PER_TASK`) |
| `--verbose` | Show detailed execution process |
| `--json` | Output results in JSON format |
| `--no-color` | Disable colored output |
| `--project PATH` | Project root directory (default: current directory) |

### Web UI

CodeAgent includes a modern React dashboard with real-time WebSocket streaming:

```bash
# Start the API server (Inline mode — no Celery Worker needed)
USE_INLINE_RUNNER=true uvicorn codeagent.interaction.api.main:app --host 0.0.0.0 --port 8000

# Start the frontend dev server (separate terminal)
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

The Web UI provides:
- **Task Input** — Submit new coding tasks with real-time progress
- **Execution Log** — Live streaming of tool calls, LLM responses, and validation results
- **Human Review Modal** — Interactive approve/abort/modify decisions
- **Progress Stepper** — Visual step-by-step progress through the plan
- **Report Panel** — Detailed task completion reports
- **Task History** — Browse and re-run previous tasks

### REST API

When the API server is running:

```bash
# Create a task
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"query": "Create a Python fibonacci function", "project_root": "/path/to/project"}'

# Get task status
curl http://localhost:8000/api/v1/tasks/{task_id}

# Get task report
curl http://localhost:8000/api/v1/tasks/{task_id}/report

# Submit human review decision
curl -X POST http://localhost:8000/api/v1/tasks/{task_id}/decision \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve"}'

# Cancel a task
curl -X DELETE http://localhost:8000/api/v1/tasks/{task_id}

# Health check
curl http://localhost:8000/health
```

#### WebSocket Streaming

```javascript
// Connect to the task event stream
const ws = new WebSocket("ws://localhost:8000/api/v1/tasks/{task_id}/stream");

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data);
  // Events: node_start, tool_call, tool_result, task_complete, task_error, keepalive
};
```

<br>

## ⚙️ Configuration

CodeAgent is configured through environment variables (`.env` file). Key configuration groups:

### LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | API key for LLM provider (required) |
| `LLM_API_BASE` | `https://api.deepseek.com/v1` | API base URL |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash` | Model name (format: `provider/model`) |
| `LLM_TIMEOUT` | `60` | LLM request timeout in seconds |

### Supported LLM Providers

CodeAgent supports 200+ providers via [litellm](https://docs.litellm.ai/docs/providers):

| Provider | Model Examples | `LLM_MODEL` Format |
|----------|---------------|-------------------|
| **DeepSeek** | deepseek-v4-flash | `deepseek/deepseek-v4-flash` |
| **OpenAI** | GPT-4o, GPT-4o-mini | `openai/gpt-4o` |
| **Anthropic** | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |
| **Google** | Gemini 2.0 Flash | `gemini/gemini-2.0-flash` |

### Cost Control

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_LLM_CALLS_PER_TASK` | `50` | Maximum LLM calls per task (prevents infinite loops) |
| `MAX_TOKENS_PER_TASK` | `100000` | Estimated maximum token consumption per task |

### Docker Sandbox

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_ENABLED` | `false` | Enable isolated Docker container execution |
| `SANDBOX_TIMEOUT` | `60` | Container command execution timeout (seconds) |
| `SANDBOX_MEMORY_MB` | `512` | Container memory limit (MB) |

### Memory System

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_GLOBAL_ROOT` | `~/.codeagent/memory` | Global memory storage root directory |
| `MEMORY_TOKEN_BUDGET` | `800` | Token budget for memory injection into system prompt |
| `MEMORY_SESSION_TTL_DAYS` | `7` | Session-type memory expiration in days |

### Task Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RETRIES` | `3` | Maximum retry count for syntax error recovery |
| `CONTEXT_BUDGET_TOKENS` | `8000` | Context budget for the context assembler |
| `AUTO_MODE` | `false` | Default auto mode (skip human review) |

### Deployment

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `CELERY_TASK_TIMEOUT` | `600` | Celery task timeout (seconds) |
| `USE_INLINE_RUNNER` | `false` | Run tasks inline without Celery Worker (for development) |

<br>

## 🐳 Docker Deployment

CodeAgent ships with a production-grade Docker setup:

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Rebuild and start
docker compose up -d --build

# Stop and clean up
docker compose down
```

### Service Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│  Nginx   │───▶│  FastAPI │───▶│    Redis      │◀───│  Celery  │
│  :80     │    │  :8000   │    │  (queue+pubsub)│   │  Worker  │
│          │    │          │    │               │    │          │
│ SPA +    │    │ REST API │    │ Message Queue │    │ Agent    │
│ Proxy    │    │ WebSocket│    │ Event Bus     │    │ Tasks    │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
```

- **Nginx** — Reverse proxy serving the React SPA (`/`) and proxying API/WebSocket requests
- **FastAPI** — REST API + WebSocket endpoint for real-time task streaming
- **Celery Worker** — Asynchronously executes Agent tasks (can be scaled horizontally)
- **Redis** — Message broker (Celery) and event bus (WebSocket Pub/Sub)

The multi-stage `Dockerfile` builds the frontend (Node), installs Python dependencies (uv), and packages everything into an Alpine-based Nginx image. One command deploys the entire stack.

<br>

## 📁 Project Structure

```
codeagent/
├── codeagent/                          # Python backend package
│   ├── config.py                       # Centralized env/flag access
│   ├── worker.py                       # Celery Worker entry point
│   ├── context_engine/                 # Code understanding subsystem
│   │   ├── engine.py                   #   Unified facade (IContextGateway)
│   │   ├── code_analyzer.py            #   AST-based code analysis
│   │   ├── code_chunker.py             #   Code chunking for context windows
│   │   ├── context_assembler.py        #   Token-budgeted XML assembly
│   │   ├── dependency_graph.py         #   Module dependency graph (NetworkX)
│   │   ├── file_tree_indexer.py        #   Project file tree scanner
│   │   ├── semantic_search.py          #   Vector-based semantic search
│   │   ├── symbol_table.py             #   Symbol table extraction
│   │   └── evolution/                  # Self-evolution subsystem
│   │       ├── manager.py              #   SelfEvolutionManager facade
│   │       ├── trajectory_recorder.py  #   Task execution trajectory recording
│   │       ├── strategy_extractor.py   #   Strategy extraction from trajectories
│   │       ├── strategy_store.py       #   Strategy persistence + decay
│   │       └── strategy_applier.py     #   Strategy retrieval + prompt injection
│   ├── gateway/                        # Interface abstractions (ports)
│   │   ├── context_gateway.py          #   IContextGateway interface
│   │   ├── memory_gateway.py           #   IMemoryGateway interface
│   │   ├── tool_gateway.py             #   IToolGateway + ToolResult DTOs
│   │   ├── validation_gateway.py       #   IValidationGateway + ValidationResult
│   │   ├── validation_gateway_impl.py  #   Validation implementation
│   │   ├── orchestration_gateway.py    #   IOrchestrationGateway contracts
│   │   └── orchestration_gateway_impl.py # Celery+Redis implementation
│   ├── interaction/                    # User-facing interfaces
│   │   ├── cli/                        #   CLI using Click + Rich
│   │   │   ├── main.py                 #     Command definitions
│   │   │   └── formatters.py           #     Rich terminal formatting
│   │   └── api/                        #   REST API (FastAPI)
│   │       ├── main.py                 #     App entry + inline runner
│   │       ├── models.py               #     Pydantic request/response models
│   │       ├── routes.py               #     Task CRUD endpoints
│   │       └── websocket.py            #     WebSocket event streaming
│   ├── memory/                         # Persistent memory subsystem
│   │   ├── manager.py                  #   MemoryManager facade (IMemoryGateway)
│   │   ├── store.py                    #   File-based MemoryEntry store
│   │   ├── retriever.py                #   Keyword + vector retrieval
│   │   └── extractor.py                #   LLM-based memory extraction
│   ├── orchestration/                  # LangGraph workflow orchestration
│   │   ├── graph.py                    #   StateGraph assembly + compilation
│   │   ├── orchestrator.py             #   Orchestrator (run/resume API)
│   │   ├── state.py                    #   AgentState + PlanStep definitions
│   │   ├── routing.py                  #   Conditional edge routing functions
│   │   ├── rollback.py                 #   File rollback manager
│   │   └── nodes/                      #   Workflow nodes
│   │       ├── context_node.py         #     Context gathering
│   │       ├── planning_node.py        #     Plan decomposition
│   │       ├── execution_node.py       #     Tool-calling execution loop
│   │       ├── validation_node.py      #     Multi-layer validation
│   │       └── human_review_node.py    #     Human-in-the-loop gate
│   ├── sandbox/                        # Docker sandbox execution
│   │   └── docker_executor.py          #   Container lifecycle management
│   ├── tools/                          # Extensible tool system
│   │   ├── base.py                     #   BaseTool abstract class
│   │   ├── gateway.py                  #   ToolGateway adapter
│   │   ├── registry.py                 #   ToolRegistry (thread-safe)
│   │   ├── file/                       #   File I/O tools
│   │   ├── search/                     #   Code + web search tools
│   │   ├── terminal/                   #   Terminal execution + safety
│   │   ├── git/                        #   Git operation tools
│   │   └── lsp/                        #   LSP diagnostic tools
│   └── validation/                     # Multi-layer code validation
│       ├── syntax_validator.py         #   Tree-sitter based syntax checks
│       ├── static_analyzer.py          #   AST/pattern rule checks
│       ├── runtime_validator.py        #   Test execution validation
│       ├── error_analyzer.py           #   Error classification + ranking
│       └── test_detector.py            #   Test file discovery
├── frontend/                           # React SPA (Vite + TailwindCSS)
│   ├── src/
│   │   ├── App.tsx                     #   Main application component
│   │   ├── api/client.ts               #   API + WebSocket client
│   │   ├── hooks/                      #   React hooks (useTask, useWebSocket)
│   │   ├── components/                 #   UI components
│   │   │   ├── TaskInput.tsx           #     Task submission form
│   │   │   ├── ProgressStepper.tsx     #     Step-by-step progress
│   │   │   ├── ExecutionLog.tsx        #     Live execution log
│   │   │   ├── HumanReviewModal.tsx    #     Review decision dialog
│   │   │   ├── ReportPanel.tsx         #     Task result report
│   │   │   └── TaskHistory.tsx         #     Previous tasks list
│   │   └── types/index.ts              #   TypeScript type definitions
│   └── ...                             #   Vite, PostCSS, Tailwind config
├── tests/                              # Comprehensive test suite
│   ├── unit/                           #   Unit tests (pytest)
│   ├── integration/                    #   Integration tests
│   └── e2e/                            #   End-to-end tests
├── scripts/                            # Development scripts
│   ├── start_dev.bat                   #   Windows dev startup
│   ├── start_dev.sh                    #   Unix dev startup
│   └── cleanup.ps1                     #   Cleanup utility
├── Dockerfile                          # Multi-stage build
├── docker-compose.yml                  # Production deployment
├── nginx.conf                          # Nginx reverse proxy config
└── pyproject.toml                      # Project metadata + dependencies
```

<br>

## 🧪 Testing

CodeAgent has a comprehensive test suite across three levels:

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run only unit tests
python -m pytest tests/unit/ -q --tb=short

# Run unit tests with coverage
python -m pytest tests/unit/ -q --tb=short --cov=codeagent --cov-report=html

# Run integration tests (network-dependent tests excluded)
python -m pytest tests/integration/ -q --tb=short -m "not requires_network and not docker"

# Run integration tests with Docker
python -m pytest tests/integration/ -q --tb=short -m "docker"

# Run end-to-end tests (requires LLM API key)
python -m pytest tests/e2e/ -q --tb=short -m "e2e"

# Run linter and type checker
uv run ruff check codeagent/
uv run mypy codeagent/ --ignore-missing-imports
```

### Test Organization

| Directory | Tests | Markers |
|-----------|-------|---------|
| `tests/unit/` | Isolated component tests (mocked dependencies) | — |
| `tests/integration/` | Cross-component tests with real deps | `docker`, `requires_network` |
| `tests/e2e/` | Full pipeline tests (requires LLM API) | `e2e`, `slow` |

<br>

## 🔬 Key Technical Details

### Gateway Architecture

CodeAgent uses a **hexagonal architecture** (ports & adapters) pattern. All major subsystems define interfaces (`I*Gateway`) that decouple the orchestration core from implementations:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Orchestration   │────▶│   IContextGateway │◀────│  ContextEngine   │
│     Core          │────▶│  IToolGateway     │◀────│  ToolGateway     │
│   (LanguageGraph) │────▶│ IValidationGateway│◀────│ ValidationGateway│
│                   │────▶│  IMemoryGateway   │◀────│  MemoryManager   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

This design allows each subsystem to be tested, replaced, or upgraded independently.

### Memory System

The memory subsystem stores four types of memories:

- **User** — User role, goals, preferences, knowledge
- **Feedback** — Guidance on approach, corrections, validated patterns
- **Project** — Ongoing work context, milestones, constraints
- **Reference** — External system pointers, documentation locations

Memories are stored as Markdown files with YAML frontmatter, indexed via `MEMORY.md`, and retrieved using both keyword matching and vector embeddings (sentence-transformers). A confidence decay mechanism prunes low-value memories over time.

### Self-Evolution

The evolution subsystem records execution trajectories, extracts reusable strategies from successful task completions, and applies them to future tasks via system prompt injection. Strategy extraction runs asynchronously to avoid blocking the main workflow.

<br>

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes and ensure tests pass: `python -m pytest tests/ -x -q`
4. Run the linter: `uv run ruff check codeagent/`
5. Run the type checker: `uv run mypy codeagent/ --ignore-missing-imports`
6. Commit your changes: `git commit -am 'feat: add my feature'`
7. Push to the branch: `git push origin feat/my-feature`
8. Open a Pull Request

### Development Conventions

- **Code style**: Ruff (line length 100), with strict mypy type checking
- **Commit style**: Conventional commits (`feat:`, `fix:`, `refactor:`, etc.)
- **Python**: 3.12+ with modern type annotations (`list[str]`, `| None`)
- **Testing**: pytest with `asyncio_mode = auto`

<br>

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `LLM_API_KEY not configured` | Copy `.env.example` to `.env` and fill in your API key |
| `Docker is not available` | Install Docker Desktop or set `SANDBOX_ENABLED=false` |
| litellm connection timeout | Check network; increase `LLM_TIMEOUT` (default 60s) |
| `ModuleNotFoundError` | Run `uv sync` to ensure all dependencies are installed |
| Test collection errors | Ensure Python 3.12+ and run `uv sync --group dev` |
| WebSocket disconnects | Check Redis connectivity; verify `REDIS_URL` setting |

<br>

## 📄 License

MIT License. See `LICENSE` for more information.

---

<p align="center">
  Built with ❤️ using Python, LangGraph, React, and FastAPI.
</p>
