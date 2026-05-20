# Deployment Guide

This guide covers production deployment of CodeAgent using Docker Compose, environment configuration, and operational best practices.

---

## Quick Start

### Prerequisites

- **Docker** 24+ and **Docker Compose** 2.20+
- **LLM API Key** (DeepSeek, OpenAI, Anthropic, or any litellm-supported provider)

### One-Command Deploy

```bash
# Clone the repository
git clone https://github.com/your-org/codeagent.git
cd codeagent

# Configure your API key
export LLM_API_KEY=sk-your-key-here

# Start all services
docker compose up -d
```

Wait 10-15 seconds for services to initialize, then open **http://localhost** in your browser.

### Stopping

```bash
# Stop all services (preserves data)
docker compose down

# Stop and delete all data (volumes)
docker compose down -v
```

---

## Service Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│  Nginx   │───▶│  FastAPI │───▶│    Redis      │◀───│  Celery  │
│  :80     │    │  :8000   │    │  (queue+pubsub)│   │  Worker  │
│          │    │          │    │               │    │          │
│ SPA +    │    │ REST API │    │ Message Queue │    │ Agent    │
│ Proxy    │    │ WebSocket│    │ Event Bus     │    │ Tasks    │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
                                                        │
                                                   ┌────▼────┐
                                                   │  Jaeger │
                                                   │ Tracing │
                                                   └─────────┘
```

| Service | Base Image | Role | Scaling |
|---------|-----------|------|---------|
| **Nginx** | `nginx:alpine` | Reverse proxy, SPA hosting | 1 (fixed) |
| **FastAPI** | `python:3.12-slim` | REST API, WebSocket, health | Horizontal |
| **Celery Worker** | `python:3.12-slim` | Agent task execution | **Horizontal** (CPU-bound) |
| **Redis** | `redis:7-alpine` | Message broker, event bus, task status | 1 (or Sentinel cluster) |
| **Jaeger** | `jaegertracing/all-in-one` | Distributed tracing (OTLP) | 1 (optional) |

---

## Configuration Reference

All configuration is via environment variables. Create a `.env` file or pass them directly to Docker Compose.

### Required

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | API key for the LLM provider. **Must be set before starting.** |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | `https://api.deepseek.com/v1` | LLM API base URL |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash` | Model name (`provider/model` format) |
| `LLM_TIMEOUT` | `120` | Request timeout in seconds |

### Task Execution

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_LLM_CALLS_PER_TASK` | `50` | Max LLM calls per task (prevents runaway costs) |
| `MAX_TOKENS_PER_TASK` | `100000` | Max estimated token consumption per task |
| `CELERY_TASK_TIMEOUT` | `600` | Celery task timeout in seconds |
| `CONTEXT_BUDGET_TOKENS` | `8000` | Context assembly token budget |

### Sandbox

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_ENABLED` | `false` | Enable Docker-in-Docker sandboxed execution |
| `SANDBOX_TIMEOUT` | `60` | Container command timeout in seconds |
| `SANDBOX_MEMORY_MB` | `512` | Container memory limit in MB |

### API Security

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | (empty) | Comma-separated valid API keys. **Required in production.** |
| `RATE_LIMIT_RPM` | `100` | Max requests per minute per key |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://jaeger:4317` | OpenTelemetry OTLP gRPC endpoint |

### Redis & Deployment

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `USE_INLINE_RUNNER` | `false` | Run tasks inline (no Celery Worker needed for single-container dev setups) |
| `CHECKPOINT_DB_PATH` | `~/.codeagent/checkpoints.db` | SQLite checkpoint database path |

---

## Production Hardening

### 1. API Security

**Always set `API_KEYS` in production.** Without it, the API has no authentication:

```bash
export API_KEYS=sk-prod-key-1,sk-prod-key-2
# Multiple keys allow key rotation without downtime
```

Rate limiting is enabled by default at 100 req/min/key. Adjust with `RATE_LIMIT_RPM`.

### 2. TLS / HTTPS

For production, place a TLS-terminating reverse proxy in front of Nginx:

**Option A: Let's Encrypt with Certbot**

```nginx
# nginx.conf snippet for TLS
server {
    listen 443 ssl;
    server_name codeagent.example.com;

    ssl_certificate /etc/letsencrypt/live/codeagent.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/codeagent.example.com/privkey.pem;

    location / {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

server {
    listen 80;
    return 301 https://$host$request_uri;
}
```

**Option B: Cloud provider LB** (AWS ALB, GCP HTTPS LB, Cloudflare) — terminate TLS at the load balancer and forward HTTP to the Nginx container.

### 3. Resource Limits

Add resource constraints to `docker-compose.yml`:

```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
  api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
```

### 4. Scaling Workers

```bash
# Scale to 4 workers for higher throughput
docker compose up -d --scale worker=4
```

Celery tasks are picked up from the Redis queue by the first available worker. Workers are stateless and can be scaled horizontally. Task state is stored in Redis; checkpoints persist to SQLite.

### 5. Persistent Storage

The following volumes persist data across restarts:

| Volume | Contents | Backup Strategy |
|--------|----------|-----------------|
| `redis_data` | Redis RDB/AOF snapshots | Include in nightly backup |
| `workspace` | Task workspace files | Backup if code artifacts matter |
| `codeagent/checkpoints.db` | Task checkpoints | **Critical** — enables task recovery |

### 6. Monitoring

- **Health endpoint**: `GET /health` on the API service
- **Metrics endpoint**: `GET /metrics` (Prometheus format) on port 8000
- **Jaeger UI**: Port 16686 for distributed trace visualization
- **Redis health**: `redis-cli ping` should return `PONG`

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'codeagent'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
```

### 7. Logging

Logs are JSON-formatted in production (controlled by `LOG_LEVEL`). Ship them to your log aggregator:

```bash
# Docker logging driver to syslog
docker compose logs -f --tail=100
```

For centralized logging, configure the `gelf` or `splunk` log driver in `docker-compose.yml`:

```yaml
services:
  api:
    logging:
      driver: gelf
      options:
        gelf-address: udp://logstash:12201
```

---

## Development Mode (Single Container)

For local development without Celery:

```bash
USE_INLINE_RUNNER=true uvicorn codeagent.interaction.api.main:app --host 0.0.0.0 --port 8000
```

This runs Agent tasks as asyncio background tasks in the API process — no Redis or Celery Worker needed. Combine with the frontend dev server:

```bash
cd frontend
npm install
npm run dev  # http://localhost:5173
```

---

## Troubleshooting

### "LLM_API_KEY is required" on startup

The `docker-compose.yml` uses `${LLM_API_KEY:?LLM_API_KEY is required}` — this variable **must** be set in the shell environment or `.env` file before running `docker compose up`. It is not read from the container's environment.

**Fix:** Ensure the variable is exported:
```bash
export LLM_API_KEY=sk-your-key
# Or create a .env file in the project root:
echo "LLM_API_KEY=sk-your-key" >> .env
```

### API returns 503 "Gateway not initialized"

The API service started before Redis was ready. This resolves automatically within a few seconds.

**Fix:** Restart the API container:
```bash
docker compose restart api
```

### WebSocket disconnects

Redis connection issues or timeout. Check:
- `docker compose logs redis` for Redis errors
- `REDIS_URL` configuration matches the running Redis instance
- Network connectivity between services

### Worker not picking up tasks

```bash
# Check worker logs
docker compose logs worker

# Verify Redis connectivity from worker
docker compose exec worker redis-cli -h redis ping
```

### Worker runs out of memory

LLM responses can be large. Increase memory limits and reduce concurrency:

```bash
docker compose up -d --scale worker=1
# In docker-compose.yml, set worker CELERY_CONCURRENCY=1
```

### Docker sandbox not working

The sandbox uses Docker-in-Docker. In Compose deployment, the Docker socket is not mounted by default for security.

**Fix for trusted environments:** Uncomment the Docker socket mount in the worker service:
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

### Jaeger tracing not visible

- Ensure Jaeger is running: `docker compose ps jaeger`
- Check the OTEL exporter endpoint: `http://jaeger:4317`
- Open Jaeger UI at http://localhost:16686
- Only tasks executed after tracing is configured will appear

---

## Backup and Recovery

### SQLite Checkpoint Backup

```bash
# Backup checkpoints
docker compose exec worker cp /workspace/.codeagent/checkpoints.db /workspace/backups/checkpoints-$(date +%Y%m%d).db

# Restore
docker compose cp ./backups/checkpoints-20260520.db worker:/workspace/.codeagent/checkpoints.db
docker compose restart worker api
```

### Redis Data

```bash
# Manual Redis save
docker compose exec redis redis-cli save
# Backup the RDB file
docker compose cp redis:/data/dump.rdb ./redis-backup.rdb
```
