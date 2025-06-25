#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-mistral}"

echo "🚀 Starting Ollama server just to pull model..."
ollama serve &
SERVE_PID=$!

# Wait for server to start accepting requests
echo "⏳ Waiting for Ollama server to come online..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "📦 Pulling Ollama model: $MODEL"
until ollama pull "$MODEL"; do
  echo "❌ Pull failed, retrying in 5s..."
  sleep 5
done

echo "✅ Model ready: $MODEL"
kill "$SERVE_PID"
sleep 1  # Give it time to exit

echo "🧠 Restarting Ollama server in foreground..."
exec ollama serve
