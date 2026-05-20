# Contributing to CodeAgent

Thank you for your interest in contributing to CodeAgent! This guide will help you set up your development environment and understand our contribution workflow.

## Table of Contents

- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Project Structure](#project-structure)

---

## Development Environment

### Prerequisites

- **Python 3.12+**
- **uv** — Fast Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Node.js 20+** (for frontend development)
- **Docker Desktop** (optional, for sandbox mode and some integration tests)
- **LLM API Key** (for end-to-end testing)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/codeagent.git
cd codeagent

# Create virtual environment and install all dependencies
uv venv
uv sync

# Install dev, sandbox, and persistence extras
uv sync --group dev

# Install pre-commit hooks (optional but recommended)
# uv run pre-commit install

# Configure environment
cp .env.example .env
# Edit .env with your LLM_API_KEY

# Verify setup
python -m pytest tests/unit/ -x -q --tb=short
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## Coding Standards

### Python

We enforce strict code quality through automated tooling:

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Linting & formatting | Line length 100, all rules enabled |
| **Mypy** | Static type checking | Strict mode, Python 3.12 |
| **Bandit** | Security linting | Default ruleset |

### Style Guidelines

- **Type annotations**: Required for all function signatures. Use modern syntax (`list[str]` not `List[str]`, `X | None` not `Optional[X]`).
- **Imports**: Use `from __future__ import annotations` in all files. Order: stdlib → third-party → local. Use blank-line separation.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Docstrings**: Optional — prefer self-documenting code with well-named identifiers. Only add docstrings for non-obvious behavior, complex algorithms, or public API surfaces.
- **Comments**: Do not explain what the code does (the code should speak for itself). Use comments only for *why* — hidden constraints, invariants, or workarounds.

### Running Linters

```bash
# Lint all source code
uv run ruff check codeagent/

# Auto-fix fixable issues
uv run ruff check --fix codeagent/

# Type check
uv run mypy codeagent/ --ignore-missing-imports

# Security scan
uv run bandit -r codeagent/
```

---

## Testing Requirements

All new features must include tests. We maintain a baseline of **1600+ passing tests** and expect contributions to maintain or improve this.

### Test Structure

| Directory | Scope | Requirements |
|-----------|-------|-------------|
| `tests/unit/` | Isolated component tests | Mock all external dependencies. Fast, <1s per test. |
| `tests/integration/` | Cross-component tests | Real Redis, Docker, or filesystem. Mark with `@pytest.mark.docker` or `@pytest.mark.requires_network`. |
| `tests/e2e/` | Full pipeline tests | Requires valid `LLM_API_KEY`. Mark with `@pytest.mark.e2e`. |
| `tests/benchmarks/` | Performance benchmarks | Mark with `@pytest.mark.benchmark`. Threshold-based assertions. |

### Running Tests

```bash
# All unit tests
python -m pytest tests/unit/ -q --tb=short

# With coverage
python -m pytest tests/unit/ --cov=codeagent --cov-report=html

# Integration tests (Docker required)
python -m pytest tests/integration/ -m docker -q --tb=short

# End-to-end tests (LLM API key required)
python -m pytest tests/e2e/ -m e2e -q --tb=short

# Benchmarks
python -m pytest tests/benchmarks/ -m benchmark --tb=short
```

### Writing Test Guidelines

- **Prefer `unittest.mock` over real dependencies** for unit tests.
- **Use `pytest.mark.asyncio`** for async test functions (framework uses `asyncio_mode=auto`).
- **Use `pytest.fixture`** for shared setup; prefer `scope="function"` unless the fixture is expensive.
- **Test edge cases**: empty inputs, error conditions, timeout scenarios.
- **Avoid mocking what you don't own** — mock your adapters, not third-party SDKs.

---

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** following our coding standards.

3. **Write tests** covering your changes (unit tests for logic, integration for cross-component behavior).

4. **Run the full test suite** and ensure all tests pass:
   ```bash
   python -m pytest tests/unit/ -q --tb=short
   uv run ruff check codeagent/
   uv run mypy codeagent/ --ignore-missing-imports
   ```

5. **Commit your changes** using conventional commits (see below).

6. **Push and open a PR**:
   ```bash
   git push origin feat/my-feature
   ```
   Then open a pull request on GitHub. Fill out the PR template with:
   - **Summary**: What and why (1-3 bullet points)
   - **Test plan**: How reviewers should verify the changes
   - **Related issues**: Links to any related issues

7. **Address review feedback**. Ensure CI checks pass (Ruff, mypy, pytest).

### PR Title Conventions

Use the same conventional commit format for PR titles:
- `feat: add support for X`
- `fix: resolve Y edge case`
- `refactor: simplify Z implementation`
- `docs: update README with API examples`

### What to Expect

- Reviews typically within 1-2 business days.
- CI must be green before merging.
- Squash merge is preferred to keep history clean.

---

## Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

### Types

| Type | Usage |
|------|-------|
| `feat` | New feature or enhancement |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behavior change) |
| `test` | Adding or modifying tests |
| `docs` | Documentation changes |
| `chore` | Build/config changes, dependencies |
| `perf` | Performance improvement |
| `ci` | CI/CD configuration changes |

### Examples

```
feat(orchestrator): add SQLite checkpoint persistence
fix(execution): handle empty plan fallback correctly
refactor(context): extract code chunker from engine
test(validation): add runtime validator timeout tests
docs: update README with benchmark commands
```

### Scope

Optional but encouraged. Common scopes:
`orchestrator`, `execution`, `planning`, `validation`, `context`, `memory`, `tools`, `api`, `cli`, `frontend`, `docker`, `config`

---

## Project Structure

See the [Project Structure section in README.md](README.md#-project-structure) for a complete overview of the codebase layout.

Key directories:
- `codeagent/` — Python backend (orchestration, tools, context engine, API)
- `frontend/` — React SPA (Vite, TailwindCSS, TypeScript)
- `tests/` — Unit, integration, e2e, and benchmark tests
- `scripts/` — Development startup and utility scripts
- `docs/` — Supplementary documentation

---

## Getting Help

- Open a [GitHub Discussion](https://github.com/your-org/codeagent/discussions) for questions
- File [GitHub Issues](https://github.com/your-org/codeagent/issues) for bugs or feature requests
- Check [docs/deployment.md](docs/deployment.md) for deployment troubleshooting
- Check [docs/api.md](docs/api.md) for API-specific questions
