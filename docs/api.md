# API Reference

CodeAgent exposes a REST API for task management and a WebSocket endpoint for real-time event streaming. The interactive OpenAPI docs are available at `/docs` when the API server is running.

**Base URL:** `http://localhost:8000`

---

## Authentication

### API Key Authentication

All `/api/v1/*` endpoints require authentication when `API_KEYS` is configured:

```bash
# Pass the API key via the Authorization header
curl -H "Authorization: Bearer sk-your-api-key" http://localhost:8000/api/v1/tasks/{task_id}
```

**How it works:**

- `API_KEYS` is a comma-separated list of valid keys set via environment variable
- If `API_KEYS` is empty or unset, the API runs in **development mode** with no authentication
- Invalid keys receive `403 Forbidden`
- Missing credentials receive `401 Unauthorized` with `WWW-Authenticate: Bearer` header
- Multiple keys allow seamless key rotation

### Rate Limiting

Rate limiting is applied per API key (or per anonymous client in dev mode):

| Header | Description |
|--------|-------------|
| `Retry-After` | Seconds until the rate limit resets (included in `429 Too Many Requests` responses) |

**Configuration:** `RATE_LIMIT_RPM` environment variable (default: 100 requests per minute per key).

**Exempt endpoints:** `/health` and `/metrics` are not rate-limited.

---

## Endpoints

### Create Task

```
POST /api/v1/tasks
```

Submit a new Agent task. Returns immediately with a `task_id` for polling.

#### Request Body

```json
{
  "query": "Add type annotations to main.py",
  "project_root": "/path/to/project",
  "auto_mode": false,
  "max_retries": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Task description (1-10000 chars) |
| `project_root` | string | ✅ | Absolute path to the project root |
| `auto_mode` | boolean | ❌ | Skip human review (default: `false`) |
| `max_retries` | integer | ❌ | Max retry count (0-10, default: 3) |

#### Response (202 Accepted)

```json
{
  "success": true,
  "data": {
    "task_id": "abc123-def456",
    "status": "pending"
  }
}
```

---

### Get Task Status

```
GET /api/v1/tasks/{task_id}
```

Poll for the current task state.

#### Response

```json
{
  "success": true,
  "data": {
    "task_id": "abc123-def456",
    "state": "running",
    "progress": 0.45,
    "current_step": "Analyzing project structure",
    "errors": []
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `state` | string | One of: `pending`, `running`, `waiting_review`, `completed`, `failed`, `cancelled` |
| `progress` | float | 0.0 to 1.0 |
| `current_step` | string or null | Description of the current step |
| `errors` | string[] | Error messages (empty if no errors) |

#### Error Response (404)

```json
{
  "success": false,
  "error": "Task not found: abc123-def456"
}
```

---

### Get Task Report

```
GET /api/v1/tasks/{task_id}/report
```

Retrieve the full task completion report.

#### Response

```json
{
  "success": true,
  "data": {
    "task_id": "abc123-def456",
    "plan": [
      {
        "action": "modify",
        "file_path": "src/main.py",
        "description": "Add type annotations"
      }
    ],
    "changes": [
      {
        "file": "src/main.py",
        "action": "modify",
        "diff": "..."
      }
    ],
    "validation_results": [
      {
        "layer": "syntax",
        "passed": true,
        "errors": []
      }
    ],
    "duration": 45.23,
    "token_usage": 15234
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `plan` | array | Planned steps for the task |
| `changes` | array | Files modified during execution |
| `validation_results` | array | Results from syntax/static/runtime validation |
| `duration` | float | Total execution time in seconds |
| `token_usage` | integer | Estimated total token consumption |

#### Error Response (404)

```json
{
  "success": false,
  "error": "Report not found for task: abc123-def456"
}
```

---

### Submit Human Review Decision

```
POST /api/v1/tasks/{task_id}/decision
```

Submit a decision for a task paused at a human review gate.

#### Request Body

```json
{
  "decision": "approve",
  "modifications": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `decision` | string | ✅ | One of: `approve`, `reject`, `modify` |
| `modifications` | object or null | ❌ | Modification parameters (used with `modify` decision) |

#### Response

```json
{
  "success": true,
  "data": {
    "task_id": "abc123-def456",
    "decision": "approve"
  }
}
```

---

### Cancel Task

```
DELETE /api/v1/tasks/{task_id}
```

Cancel a running or pending task.

#### Response

```json
{
  "success": true,
  "data": {
    "cancelled": true
  }
}
```

---

### Health Check

```
GET /health
```

Unauthenticated endpoint. Returns basic service health information.

#### Response

```json
{
  "status": "ok",
  "inline_mode": false,
  "redis": "redis:6379"
}
```

### Prometheus Metrics

```
GET /metrics
```

Unauthenticated endpoint. Returns Prometheus-formatted metrics.

#### Response (Content-Type: text/plain)

```
# HELP codeagent_llm_calls_total LLM 调用次数
# TYPE codeagent_llm_calls_total counter
codeagent_llm_calls_total{model="deepseek/deepseek-v4-flash",status="success"} 42
# HELP codeagent_task_duration_seconds ...
```

---

## WebSocket Event Stream

```
ws://localhost:8000/api/v1/tasks/{task_id}/stream
```

Connect to receive real-time execution events for a task.

### Connection Lifecycle

1. Client opens WebSocket connection
2. Server replays any historical events (from Redis event log)
3. Server streams new events as they occur
4. Server sends a `ping` keepalive every 25 seconds
5. Connection closes automatically on `task_complete` or `task_error`

### Event Types

#### `node_start`

Emitted when a workflow node begins execution.

```json
{
  "type": "node_start",
  "node": "planning",
  "summary": "Analyzing request and generating plan",
  "timestamp": "2026-05-20T10:00:00.123Z"
}
```

#### `tool_call`

Emitted when the Agent invokes a tool.

```json
{
  "type": "tool_call",
  "tool": "write_file",
  "arguments": {
    "file_path": "src/main.py",
    "content": "..."
  },
  "step_id": 3,
  "timestamp": "2026-05-20T10:00:05.456Z"
}
```

#### `tool_result`

Emitted when a tool call completes.

```json
{
  "type": "tool_result",
  "tool": "write_file",
  "success": true,
  "duration_ms": 12.3,
  "step_id": 3,
  "timestamp": "2026-05-20T10:00:05.470Z"
}
```

#### `progress`

Periodic progress updates during execution.

```json
{
  "type": "progress",
  "progress": 0.65,
  "current_step": "Running static analysis",
  "step_index": 4,
  "total_steps": 6,
  "timestamp": "2026-05-20T10:00:10.789Z"
}
```

#### `human_review`

Emitted when the workflow pauses for a human decision.

```json
{
  "type": "human_review",
  "summary": "Deviations detected from original plan",
  "options": ["approve", "reject", "modify"],
  "timestamp": "2026-05-20T10:01:00.000Z"
}
```

#### `validation_result`

Emitted after each validation layer runs.

```json
{
  "type": "validation_result",
  "layer": "syntax",
  "passed": true,
  "errors": [],
  "timestamp": "2026-05-20T10:00:30.111Z"
}
```

#### `task_complete`

Terminal event — task finished successfully.

```json
{
  "type": "task_complete",
  "status": "success",
  "duration": 45.23,
  "token_usage": 15234,
  "timestamp": "2026-05-20T10:01:30.000Z"
}
```

#### `task_error`

Terminal event — task failed.

```json
{
  "type": "task_error",
  "error": "LLM call timed out after 60s",
  "timestamp": "2026-05-20T10:01:30.000Z"
}
```

#### `ping`

Keepalive heartbeat (sent every 25 seconds during idle periods).

```json
{
  "type": "ping",
  "timestamp": "2026-05-20T10:00:25.000Z"
}
```

### Client Example (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/tasks/abc123/stream");

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case "node_start":
      console.log(`Node started: ${msg.node}`);
      break;
    case "tool_call":
      console.log(`Tool: ${msg.tool}`, msg.arguments);
      break;
    case "tool_result":
      console.log(`Result: ${msg.tool} (${msg.success})`);
      break;
    case "task_complete":
      console.log(`Done in ${msg.duration}s`);
      ws.close();
      break;
    case "task_error":
      console.error(`Failed: ${msg.error}`);
      ws.close();
      break;
    case "ping":
      // Ignore keepalive pings
      break;
  }
};

ws.onclose = () => console.log("Connection closed");
```

---

## Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| `202 Accepted` | Task created successfully |
| `200 OK` | Request succeeded |
| `401 Unauthorized` | Missing or invalid authentication |
| `403 Forbidden` | Invalid API key |
| `404 Not Found` | Task ID or report not found |
| `429 Too Many Requests` | Rate limit exceeded |
| `500 Internal Server Error` | Unexpected server error |
| `503 Service Unavailable` | Gateway not initialized (starting up) |

---

## Interactive Docs

When the API server is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These are auto-generated from the FastAPI OpenAPI schema and provide a try-it-out interface for all endpoints.
