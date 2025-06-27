#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

echo "🚀 Starting Ollama Chat server just to pull model..."
ollama serve &  # Correct: should be plain 'ollama'
SERVE_PID=$!

# Wait for server to start accepting requests
echo "⏳ Waiting for Ollama Chat server to come online..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "📦 Pulling Ollama chat model: $MODEL"
until ollama pull "$MODEL"; do
  echo "❌ Pull failed, retrying in 5s..."
  sleep 5
done

echo "✅ Chat model ready: $MODEL"
kill "$SERVE_PID"
sleep 1  # Give it time to exit

echo "🧠 Restarting Ollama Chat server in foreground on port 12434..."
exec ollama serve
