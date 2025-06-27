#!/bin/bash
set -e

MAX_ATTEMPTS=30
SLEEP_INTERVAL=2

echo "🌐 Redis Queue Worker Environment:"
echo "VECTORDB_URL: $VECTORDB_URL"
echo "OLLAMA_BASE_URL: $OLLAMA_BASE_URL"

# --- Wait for FAISS Service ---
echo "⏳ Waiting for FAISS Service at $VECTORDB_URL/health..."
for i in $(seq 1 $MAX_ATTEMPTS); do
    if curl -sSf "$VECTORDB_URL/health" > /dev/null; then
        echo "✅ FAISS Service is ready."
        break
    fi
    echo "⏳ Attempt $i/$MAX_ATTEMPTS - FAISS Service not ready yet."
    sleep $SLEEP_INTERVAL
    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "❌ FAISS Service did not become ready in time."
        exit 1
    fi
done

# --- Wait for Ollama ---
echo "⏳ Waiting for Ollama at $OLLAMA_BASE_URL/api/tags..."
for i in $(seq 1 $MAX_ATTEMPTS); do
    if curl -sSf "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
        echo "✅ Ollama is ready."
        break
    fi
    echo "⏳ Attempt $i/$MAX_ATTEMPTS - Ollama not ready yet."
    sleep $SLEEP_INTERVAL
    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "❌ Ollama did not become ready in time."
        exit 1
    fi
done

echo "🚀 Starting Redis Queue Worker..."
exec python src/redis_queue_worker/main.py
