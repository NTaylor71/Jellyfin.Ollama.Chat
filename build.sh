#!/usr/bin/env bash
# Jellyfin.Ollama.Chat Build Script (Linux/macOS)
# - Loads .env into session
# - Validates Docker + Compose
# - Builds and launches dev stack

set -euo pipefail

COMPOSE_FILE="docker-compose.dev.yml"

echo "🔧 Jellychat: Starting build..."

# Check Docker
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ Docker not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "❌ Compose file '$COMPOSE_FILE' not found." >&2
  exit 1
fi

# Load .env if present
if [[ -f ".env" ]]; then
  echo "📦 Loading .env..."
  export $(grep -v '^#' .env | xargs -d '\n')
else
  echo "⚠️  .env file not found — continuing without it."
fi

# Build and launch
echo "🔨 Building Docker containers..."
docker compose -f "$COMPOSE_FILE" build

echo "🚀 Launching stack..."
docker compose -f "$COMPOSE_FILE" up
