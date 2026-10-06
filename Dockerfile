# Multi-stage Dockerfile for Agromet Agentic Engine (A²E)
# Fast, lightweight, production-ready container (< 350MB)

FROM python:3.11-slim AS base

# Install system dependencies & Bun runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
    && curl -fsSL https://bun.sh/install | bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.bun/bin:${PATH}"
WORKDIR /app

# Install Python backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install and build Bun frontend
COPY frontend/package.json frontend/bun.lockb* frontend/tsconfig.json frontend/
WORKDIR /app/frontend
RUN bun install --frozen-lockfile || bun install
COPY frontend/src/ ./src/
COPY frontend/public/ ./public/
COPY frontend/server.ts .
RUN bun run build

WORKDIR /app
# Copy backend application and models
COPY services/ ./services/
COPY data/ ./data/
COPY .env.example .env

# Expose Frontend (3000) and Backend (8000)
EXPOSE 3000 8000

# Entrypoint script to launch both services
CMD ["sh", "-c", "uvicorn services.api.main:app --host 0.0.0.0 --port 8000 & bun run --cwd frontend server.ts"]
