#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-nomic-embed-text}"

echo "🚀 Starting Ollama Embed server just to pull model..."
ollama serve &  # Correct binary: plain 'ollama'
SERVE_PID=$!

# Wait for server to start accepting requests
echo "⏳ Waiting for Ollama Embed server to come online..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "📦 Pulling Ollama embedding model: $MODEL"
until ollama pull "$MODEL"; do
  echo "❌ Pull failed, retrying in 5s..."
  sleep 5
done

echo "✅ Embedding model ready: $MODEL"
kill "$SERVE_PID"
sleep 1  # Give it time to exit

echo "🧠 Restarting Ollama Embed server in foreground on port 12435..."
exec ollama serve
