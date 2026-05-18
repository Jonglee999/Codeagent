#!/usr/bin/env bash
# scripts/start_dev.sh — 本地开发一键启动脚本
# 使用 Inline 模式：只需 FastAPI + Redis，无需 Celery Worker
#
# 使用方法：
#   bash scripts/start_dev.sh          # 启动后端（inline 模式）
#   bash scripts/start_dev.sh --celery # 启动后端（celery 模式）+ worker
#   bash scripts/start_dev.sh --frontend # 同时启动前端

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

USE_CELERY=false
USE_FRONTEND=false
for arg in "$@"; do
  case "$arg" in
    --celery)   USE_CELERY=true ;;
    --frontend) USE_FRONTEND=true ;;
  esac
done

# 检查 .env
if [ ! -f "$ROOT/.env" ]; then
  echo "[WARN] .env not found, using defaults. Copy .env.example to .env and fill in LLM_API_KEY."
fi

# 检查 Redis
echo "[INFO] Checking Redis..."
if python -c "import redis; redis.from_url('redis://localhost:6379/0').ping()" 2>/dev/null; then
  echo "[OK] Redis is running"
else
  echo "[ERROR] Redis is not running. Please start Redis first:"
  echo "  Windows: redis-server (or run Redis via Docker)"
  echo "  Docker:  docker run -d -p 6379:6379 redis:7-alpine"
  exit 1
fi

cleanup() {
  echo ""
  echo "[INFO] Shutting down..."
  kill $(jobs -p) 2>/dev/null || true
  wait
}
trap cleanup EXIT INT TERM

if $USE_CELERY; then
  echo "[INFO] Starting Celery Worker..."
  export PYTHONPATH="$ROOT:$PYTHONPATH"
  python -m celery -A codeagent.worker worker --loglevel=info --concurrency=2 &
  CELERY_PID=$!
  sleep 2
  echo "[OK] Celery Worker started (PID=$CELERY_PID)"
  export USE_INLINE_RUNNER=false
else
  echo "[INFO] Using INLINE mode (no Celery Worker needed)"
  export USE_INLINE_RUNNER=true
fi

echo "[INFO] Starting FastAPI..."
export PYTHONPATH="$ROOT:$PYTHONPATH"
python -m uvicorn codeagent.interaction.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info &
API_PID=$!
sleep 2

# 健康检查
if curl -sf http://localhost:8000/health > /dev/null; then
  echo "[OK] FastAPI is running at http://localhost:8000"
  echo "     Health: $(curl -s http://localhost:8000/health)"
else
  echo "[ERROR] FastAPI failed to start"
  exit 1
fi

if $USE_FRONTEND; then
  echo "[INFO] Starting Vite frontend..."
  cd "$ROOT/frontend"
  npm run dev &
  VITE_PID=$!
  cd "$ROOT"
  sleep 3
  echo "[OK] Frontend running at http://localhost:5173"
fi

echo ""
echo "==================================================="
echo "  CodeAgent is running!"
echo ""
if $USE_FRONTEND; then
  echo "  Frontend: http://localhost:5173"
fi
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Mode:     $([ $USE_CELERY = true ] && echo 'Celery' || echo 'Inline (no Worker needed)')"
echo "==================================================="
echo ""
echo "Press Ctrl+C to stop all services."
wait
