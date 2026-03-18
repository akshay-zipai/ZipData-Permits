#!/usr/bin/env bash

set -euo pipefail

REPO_URL_DEFAULT="https://github.com/akshay-zipai/ZipData-Permits.git"
APP_DIR_DEFAULT="$HOME/ZipData-Permits"
PYTHON_BIN_DEFAULT="python3"

read -r -p "GitHub username: " GITHUB_USERNAME
read -r -s -p "GitHub personal access token: " GITHUB_TOKEN
echo
read -r -p "Repo URL [$REPO_URL_DEFAULT]: " REPO_URL
REPO_URL="${REPO_URL:-$REPO_URL_DEFAULT}"
read -r -p "Install directory [$APP_DIR_DEFAULT]: " APP_DIR
APP_DIR="${APP_DIR:-$APP_DIR_DEFAULT}"
read -r -p "Bedrock region [us-east-1]: " BEDROCK_REGION
BEDROCK_REGION="${BEDROCK_REGION:-us-east-1}"
read -r -p "Bedrock model ID [google.gemma-3-12b-it]: " BEDROCK_MODEL_ID
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-google.gemma-3-12b-it}"

if command -v dnf >/dev/null 2>&1; then
  sudo dnf update -y
  sudo dnf install -y git "$PYTHON_BIN_DEFAULT" "$PYTHON_BIN_DEFAULT"-pip
elif command -v yum >/dev/null 2>&1; then
  sudo yum update -y
  sudo yum install -y git "$PYTHON_BIN_DEFAULT" "$PYTHON_BIN_DEFAULT"-pip
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git python3 python3-pip python3-venv
else
  echo "Unsupported package manager. Install git and Python 3 manually."
  exit 1
fi

if [ -d "$APP_DIR/.git" ]; then
  echo "Repo already exists at $APP_DIR"
else
  CLONE_URL="${REPO_URL/https:\/\//https:\/\/${GITHUB_USERNAME}:${GITHUB_TOKEN}@}"
  git clone "$CLONE_URL" "$APP_DIR"
  git -C "$APP_DIR" remote set-url origin "$REPO_URL"
fi

unset GITHUB_TOKEN

cd "$APP_DIR"

$PYTHON_BIN_DEFAULT -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements-ec2-lite.txt

mkdir -p data output chroma_db

if [ ! -f .env ]; then
  cat > .env <<EOF
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
BEDROCK_REGION=$BEDROCK_REGION
BEDROCK_MODEL_ID=$BEDROCK_MODEL_ID
ENABLE_LOCAL_RAG=false
ENABLE_AUTO_CRAWL=false
EOF
fi

python scripts/download_crosswalk.py
python generate_ca_permit_mapping.py

pkill -f "uvicorn app.main:app" || true
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 > app.log 2>&1 &

echo "App started. Tail logs with: tail -f $APP_DIR/app.log"
