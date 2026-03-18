# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

ARG APP_ENV=dev
ARG ENABLE_LOCAL_RAG=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
COPY requirements-ec2-lite.txt .
RUN pip install --upgrade pip && \
    if [ "$ENABLE_LOCAL_RAG" = "true" ] || [ "$ENABLE_LOCAL_RAG" = "True" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    elif [ "$APP_ENV" = "production" ]; then \
        pip install --no-cache-dir -r requirements-ec2-lite.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Install browser tooling only for non-production builds.
RUN mkdir -p /root/.cache/ms-playwright && \
    if [ "$ENABLE_LOCAL_RAG" = "true" ] || [ "$ENABLE_LOCAL_RAG" = "True" ] || [ "$APP_ENV" != "production" ]; then \
        playwright install chromium; \
    fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ARG APP_ENV=dev
ARG ENABLE_LOCAL_RAG=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    if [ "$ENABLE_LOCAL_RAG" = "true" ] || [ "$ENABLE_LOCAL_RAG" = "True" ] || [ "$APP_ENV" != "production" ]; then \
        apt-get install -y --no-install-recommends \
            libnss3 \
            libnspr4 \
            libatk1.0-0 \
            libatk-bridge2.0-0 \
            libcups2 \
            libdrm2 \
            libxkbcommon0 \
            libxcomposite1 \
            libxdamage1 \
            libxfixes3 \
            libxrandr2 \
            libgbm1 \
            libasound2 \
            libpango-1.0-0 \
            libcairo2 \
            libx11-6 \
            libxext6 \
            libxshmfence1; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Copy venv + playwright browsers from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright

ENV PATH="/opt/venv/bin:$PATH"
# Tell playwright where browsers are
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/chroma_db /app/data /app/output && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /root/.cache/ms-playwright

# Copy app
COPY --chown=appuser:appuser app/                     ./app/
COPY --chown=appuser:appuser prompts/                 ./prompts/
COPY --chown=appuser:appuser data/permit_portals.json ./data/permit_portals.json
COPY --chown=appuser:appuser scripts/                 ./scripts/
COPY --chown=appuser:appuser generate_ca_permit_mapping.py .

RUN chmod +x scripts/entrypoint.sh

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["scripts/entrypoint.sh"]
