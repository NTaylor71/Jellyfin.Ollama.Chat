#!/bin/bash
set -e

echo "⏳ Waiting for PostgreSQL..."
until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER"; do
    sleep 1
done

echo "⏳ Waiting for Qdrant vector DB..."
until curl -sSf "$VECTORDB_URL/health" > /dev/null; do
    sleep 1
done

echo "⏳ Waiting for Ollama..."
until curl -sSf "$OLLAMA_BASE_URL" > /dev/null; do
    sleep 1
done

echo "🚀 Starting queue listener..."
exec python src/worker/main.py
