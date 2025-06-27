#!/bin/bash
set -e

echo "🌐 GPU Queue Worker Environment:"
echo "VECTORDB_URL: $VECTORDB_URL"
echo "OLLAMA_EMBED_BASE_URL: $OLLAMA_EMBED_BASE_URL"
echo "OLLAMA_CHAT_BASE_URL: $OLLAMA_CHAT_BASE_URL"

source /wait_for_services.sh

wait_for_service "FAISS Service" "$VECTORDB_URL/health"
wait_for_service "Ollama Embed" "$OLLAMA_EMBED_BASE_URL/api/tags"
wait_for_service "Ollama Chat" "$OLLAMA_CHAT_BASE_URL/api/tags"

echo "🚀 Starting GPU Queue Worker..."
exec python /app/src/gpu_queue_worker/main.py
