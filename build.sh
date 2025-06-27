#!/bin/bash

# FAISS RAG Build Script
# • Loads .env if present
# • Validates docker + compose
# • Builds and launches containers

set -e

COMPOSE_FILE="docker-compose.dev.yml"

echo "🔧 FAISS RAG: Starting build..."

# Validate Docker
if ! command -v docker &> /dev/null
then
    echo "❌ Docker not found in PATH."
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Compose file '$COMPOSE_FILE' not found."
    exit 1
fi

# Load .env if present
if [ -f ".env" ]; then
    echo "📦 Loading .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️ .env file not found — continuing without it."
fi

echo "🔨 Building Docker containers..."
docker compose -f "$COMPOSE_FILE" build

echo "🚀 Launching stack..."
docker compose -f "$COMPOSE_FILE" up

echo ""
echo "✅ Done."
