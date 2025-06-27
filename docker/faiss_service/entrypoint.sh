#!/bin/bash
set -e

MAX_ATTEMPTS=30
SLEEP_INTERVAL=2

echo "🌐 FAISS Service Environment:"
echo "VECTORDB_URL: $VECTORDB_URL"
echo "OLLAMA_EMBED_BASE_URL: $OLLAMA_EMBED_BASE_URL"

# --- Wait for Ollama Embed ---
echo "⏳ Waiting for Ollama Embed at $OLLAMA_EMBED_BASE_URL/api/tags..."
for i in $(seq 1 $MAX_ATTEMPTS); do
    if curl -sSf "$OLLAMA_EMBED_BASE_URL/api/tags" > /dev/null; then
        echo "✅ Ollama Embed is ready."
        break
    fi
    echo "⏳ Attempt $i/$MAX_ATTEMPTS - Ollama Embed not ready yet."
    sleep $SLEEP_INTERVAL
    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "❌ Ollama Embed did not become ready in time."
        exit 1
    fi
done

echo "🚀 Starting FAISS Service..."
exec python src/faiss_service/main.py
