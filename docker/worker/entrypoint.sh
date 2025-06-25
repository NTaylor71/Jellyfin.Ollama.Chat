#!/bin/bash
set -e
echo "⏳ Waiting for dependencies..."
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER"; do sleep 1; done
until curl -sSf "$VECTORDB_URL/health" > /dev/null; do sleep 1; done
until curl -sSf "$OLLAMA_BASE_URL" > /dev/null; do sleep 1; done
echo "🚀 Starting queue listener..."
exec python worker/main.py
