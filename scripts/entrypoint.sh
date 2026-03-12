#!/bin/bash
# scripts/entrypoint.sh
# Runs on every container start. Generates data files if missing, then starts API.

set -e

echo "============================================"
echo " CA Permit RAG System — Starting Up"
echo "============================================"

# ── Step 1: ZIP → County crosswalk ───────────────────────────────────────────
if [ ! -f "data/zip_county_crosswalk.csv" ]; then
    echo "[1/3] Generating ZIP→County crosswalk from Census data..."
    python scripts/download_crosswalk.py
else
    echo "[1/3] ZIP crosswalk already exists — skipping download."
fi

# ── Step 2: Permit mapping JSON ───────────────────────────────────────────────
echo "[2/3] Generating CA permit mapping..."
python generate_ca_permit_mapping.py

# ── Step 3: Start API ─────────────────────────────────────────────────────────
echo "[3/3] Starting FastAPI server..."
echo "============================================"

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
