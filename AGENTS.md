# CodeAgent development guide

## Standard workflow

Use the repository Harness as the single entry point on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 setup
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 doctor
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 up
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 check
powershell -ExecutionPolicy Bypass -File scripts/harness.ps1 verify
```

The UI runs at `http://127.0.0.1:5173`; FastAPI and OpenAPI run at
`http://127.0.0.1:8000` and `/docs`. Use `status`, `logs`, and `down` for lifecycle
management. Harness exit code 3 means an optional capability is deliberately unavailable.

## Change discipline

- Keep secrets only in `.env`; never commit or print them.
- Default local development to `USE_INLINE_RUNNER=true`; Redis is still required.
- Do not start Celery unless a change specifically exercises distributed execution.
- Run the smallest relevant tests while iterating, then `harness.ps1 test` before handoff.
- Do not run LLM-backed E2E without stating that it consumes API quota.
- `eval` contains deterministic proxies only; never describe it as official benchmark evidence.
- Preserve REST/WebSocket shapes or update docs, frontend types, and tests together.
- Treat `uv.lock` and `package-lock.json` as reproducibility sources.
