# =============================================================================
# CodeAgent — 多阶段构建
#
# Stage 1: frontend-builder — 构建 React SPA（Vite + TailwindCSS）
# Stage 2: app           — Python 后端（uv sync 安装生产依赖）
# Stage 3: nginx         — 生产 Nginx 容器（内嵌前端静态文件）
# =============================================================================

# ── Stage 1: 构建前端 ────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

# 先复制依赖声明文件，利用 Docker 缓存层
COPY frontend/package*.json ./
RUN npm ci

# 复制前端源码并构建
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python 应用 ─────────────────────────────────────────────
FROM python:3.12-slim AS app
WORKDIR /app

# 安装 uv（快速 Python 包管理器）
RUN pip install --no-cache-dir uv

# 复制依赖声明文件并安装（利用 Docker 缓存层）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 复制项目源码
COPY codeagent/ ./codeagent/

# 安装项目自身
RUN uv sync --frozen --no-dev

# 复制前端构建产物到 static/ 目录（供静态文件服务或备用）
COPY --from=frontend-builder /app/frontend/dist ./static/

# ── Stage 3: Nginx 生产容器 ──────────────────────────────────────────
FROM nginx:alpine AS nginx
COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 默认 CMD（仅为 Stage 2 的 CMD，Stage 3 由 docker-compose 覆盖）
CMD ["uvicorn", "codeagent.interaction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
