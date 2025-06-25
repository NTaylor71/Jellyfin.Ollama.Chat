#!/bin/bash
set -e

echo "⏳ Waiting for Postgres..."
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

echo "📥 Starting ingestion script..."
exec python src/ingestor/main.py
