# CodeAgent

CodeAgent is a Web-first software-engineering agent. It accepts an issue and a local
repository, builds a plan, inspects and edits code through bounded tools, runs validation,
streams progress over WebSocket, and stores project-scoped experience for later runs.

The repository includes a 30-instance SWE-bench Lite smoke catalog. The catalog is for
repeatable product testing and inference experiments; only the upstream SWE-bench Docker
harness can establish an official resolved/not-resolved result.

## Quick start on Windows

Requirements: Python 3.12+, Node.js 20+, Git, and Docker Desktop.

```powershell
Copy-Item .env.example .env
# Edit .env locally and set LLM_API_KEY, LLM_MODEL, and LLM_API_BASE as needed.

powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 setup
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 doctor
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 up
```

Open `http://127.0.0.1:5173`. API documentation is available at
`http://127.0.0.1:8000/docs`.

Do not commit `.env`. Real agent and SWE inference runs consume model quota.

## Product flow

1. Enter a query in the AGENT4CODE conversation box; no project path is required.
2. A normal query receives a writable project-local workspace automatically. Optionally create a
   durable project to share uploaded and generated files across its conversations. A selected
   **SWE Smoke** item prepares and uses its pinned repository behind the UI.
3. Start the run. The timeline shows planning, tool calls, validation, review requests,
   and completion events.
4. Inspect the report and repository diff. A model response alone is not a successful run;
   validation evidence is required.

The default local profile uses the inline runner and Redis. Celery is only needed for the
distributed deployment profile. Context runs in `auto` mode: ripgrep and tree-sitter stay
available immediately, while large repositories build their LanceDB index in the background.
The agent can continue narrowing the task with bounded search/read tools while indexing.

## Harness

The Windows Harness is the single local entry point:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 check
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 test
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 eval
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 security
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 verify
```

Use `status`, `logs`, and `down` to manage local services. `eval` is deterministic and does
not call an LLM. `lint`, `typecheck`, and `integration` are strict standalone gates.

SWE inference is an explicit quota-consuming gate. Inspect state and export completed real runs
without calling an LLM via `swe-status` and `swe-export`. To start inference, acknowledge cost:

```powershell
$env:CONFIRM_LLM_API_COST='true'
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 swe-infer smoke-5
```

## Main capabilities

- LangGraph context, planning, execution, validation, review, and retry workflow.
- Hybrid code discovery through bounded ripgrep, tree-sitter symbols/dependencies, and
  progressively available LanceDB semantic search.
- Confined file listing/read/write/delete, conflict-aware patching, regex code search, Git,
  diagnostics, terminal, and MCP tools.
- Project memory, trajectory recording, strategy extraction, and strategy reuse.
- Project instructions from `AGENTS.md` and `.codeagent/skills/*/SKILL.md`.
- REST task lifecycle, Redis-backed event history, live WebSocket delivery, and reports.
- Responsive React UI with environment readiness, benchmark catalog, recent runs, durable projects,
  file upload/download, Markdown conversation rendering, collapsible real-event thinking cards,
  reconnect feedback, review controls, and final evidence.

MCP stdio and Streamable HTTP servers are health-checked at run start and their allow-listed tools
are registered independently. Copy `.codeagent/mcp.example.json` to `.codeagent/mcp.json`; keep
credentials in the ignored `mcp.secrets.json` or allow-listed process environment only. The UI
reports real server health. Use `codeagent mcp doctor --project .` for a real, LLM-free handshake
and discovery report.

Model calls share classified retry, Retry-After handling, provider/model circuits, and an optional
explicit fallback. Redis and tool recovery evidence is streamed and persisted in task reports.

## Documentation

- [REST and WebSocket contract](docs/api.md)
- [Deployment guide](docs/deployment.md)
- [Current capability improvement plan](docs/current/agent-swe-capability-improvement-plan-2026-08-10.md)
- [Latest benchmark canary record](docs/current/benchmark-canary-followup-2026-08-10.md)
- [Harness commands](docs/harness/README.md)
- [SWE Smoke methodology](docs/evals/swe-smoke.md)

## Repository map

```text
codeagent/interaction/api   FastAPI and WebSocket boundary
codeagent/orchestration     workflow and agent nodes
codeagent/context_engine    tree, analysis, semantic context, memory, evolution
codeagent/tools             file, search, Git, diagnostics, terminal tools
codeagent/benchmarks        SWE task catalog and pinned workspace preparation
frontend/src                React/Vite dashboard
evals/swe_smoke             curated task manifest
scripts/harness.ps1         setup, lifecycle, checks, and deterministic gates
```

## Security

Tools resolve paths beneath the selected project root. Terminal execution applies command
safety checks, a timeout, a workspace-bound working directory, and bounded output. For
untrusted repositories, use Docker isolation and review the selected task before execution.
Never expose provider keys through task text, logs, or committed files.
