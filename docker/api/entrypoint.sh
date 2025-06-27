#!/bin/bash
set -e

echo "🌐 API Environment:"
echo "VECTORDB_URL: $VECTORDB_URL"
echo "OLLAMA_EMBED_BASE_URL: $OLLAMA_EMBED_BASE_URL"
echo "OLLAMA_CHAT_BASE_URL: $OLLAMA_CHAT_BASE_URL"
echo "REDIS_HOST: $REDIS_HOST"

source /wait_for_services.sh

wait_for_service "FAISS vector DB" "$VECTORDB_URL/health"
wait_for_service "Ollama Embed" "$OLLAMA_EMBED_BASE_URL/api/tags"
wait_for_service "Ollama Chat" "$OLLAMA_CHAT_BASE_URL/api/tags"
wait_for_service "Redis" "$REDIS_HOST:6379/ping"

echo "🚀 Starting API server..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000

