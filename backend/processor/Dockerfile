# syntax=docker/dockerfile:1

############################
# 1. Build stage – install deps into /install
############################
FROM python:3.11-slim AS builder
WORKDIR /app
COPY processor/requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

############################
# 2. Runtime stage – small final image
############################
FROM python:3.11-slim
WORKDIR /app

# Minimal OS packages (curl for health-check)
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy installed Python libs & source code
COPY --from=builder /install /usr/local
COPY processor/ .
COPY lib/ ./lib/

# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Cloud Run convention
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Health check (Cloud Run will periodically hit /health, uses the PORT env var)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Allow overriding the service to run at build time (defaults to analyzer)
ARG SERVICE=analyzer

# ---- START APP ----
# Gunicorn binds to 0.0.0.0:$PORT (Cloud Run injects PORT)
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers ${GUNICORN_WORKERS:-2} --threads ${GUNICORN_THREADS:-4} --timeout 60 "${SERVICE}_main:app"
