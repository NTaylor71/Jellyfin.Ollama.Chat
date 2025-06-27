#!/bin/bash
set -e

MAX_ATTEMPTS=30
SLEEP_INTERVAL=2

echo "🌐 FAISS Service Environment:"
echo "VECTORDB_URL: $VECTORDB_URL"
echo "OLLAMA_BASE_URL: $OLLAMA_BASE_URL"

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

echo "🚀 Starting FAISS Service..."
exec python src/faiss_service/main.py
