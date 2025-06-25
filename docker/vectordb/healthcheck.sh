#!/bin/bash
set -e

for i in {1..15}; do
  echo "[Qdrant healthcheck] Attempt $i..."
  if curl -sf http://localhost:6333/health | grep -q '"status":"ok"'; then
    echo "[Qdrant healthcheck] Qdrant is healthy"
    exit 0
  fi
  sleep 2
done

echo "[Qdrant healthcheck] Failed after 15 attempts"
exit 1
