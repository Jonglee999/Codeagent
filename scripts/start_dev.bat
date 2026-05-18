@echo off
REM scripts\start_dev.bat — Windows 本地开发一键启动脚本
REM 使用 Inline 模式（默认）：只需 FastAPI + Redis，无需 Celery Worker
REM
REM 用法:
REM   start_dev.bat               -- Inline 模式（推荐本地开发）
REM   start_dev.bat --celery      -- Celery 模式（需要单独 Worker）
REM   start_dev.bat --frontend    -- 同时启动前端

setlocal EnableDelayedExpansion

cd /d "%~dp0.."
set ROOT=%CD%

set USE_CELERY=false
set USE_FRONTEND=false

for %%a in (%*) do (
    if "%%a"=="--celery"   set USE_CELERY=true
    if "%%a"=="--frontend" set USE_FRONTEND=true
)

REM 检查 Redis
echo [INFO] Checking Redis...
python -c "import redis; redis.from_url('redis://localhost:6379/0').ping(); print('OK')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Redis is not running. Start Redis first:
    echo   Download: https://github.com/microsoftarchive/redis/releases
    echo   Or Docker: docker run -d -p 6379:6379 redis:7-alpine
    pause
    exit /b 1
)
echo [OK] Redis is running

set PYTHONPATH=%ROOT%;%PYTHONPATH%

if "%USE_CELERY%"=="true" (
    echo [INFO] Starting Celery Worker...
    start "Celery Worker" cmd /k "cd /d %ROOT% && python -m celery -A codeagent.worker worker --loglevel=info --concurrency=2"
    timeout /t 3 >nul
    set USE_INLINE_RUNNER=false
    echo [OK] Celery Worker started
) else (
    echo [INFO] Using INLINE mode - no Celery Worker needed
    set USE_INLINE_RUNNER=true
)

echo [INFO] Starting FastAPI server...
start "CodeAgent API" cmd /k "cd /d %ROOT% && set USE_INLINE_RUNNER=%USE_INLINE_RUNNER% && python -m uvicorn codeagent.interaction.api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info"
timeout /t 3 >nul

python -c "import httpx; r=httpx.get('http://localhost:8000/health'); print('[OK] FastAPI running:', r.json())" 2>nul || echo [WARN] FastAPI may still be starting...

if "%USE_FRONTEND%"=="true" (
    echo [INFO] Starting Vite frontend...
    start "Vite Frontend" cmd /k "cd /d %ROOT%\frontend && npm run dev"
    timeout /t 4 >nul
    echo [OK] Frontend starting at http://localhost:5173
)

echo.
echo ===================================================
echo   CodeAgent is running!
echo.
if "%USE_FRONTEND%"=="true" echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
if "%USE_INLINE_RUNNER%"=="true" (
    echo   Mode: INLINE (no Celery Worker needed^)
) else (
    echo   Mode: Celery
)
echo ===================================================
echo.
echo All services started in separate windows.
echo Close those windows to stop the services.
pause
