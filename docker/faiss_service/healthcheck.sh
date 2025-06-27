#!/bin/bash
set -e

MAX_ATTEMPTS=30
WAIT_SECONDS=2

for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[FAISS Service healthcheck] Attempt $i of $MAX_ATTEMPTS..."
  if curl -sf http://localhost:6333/health > /dev/null; then
    echo "[FAISS Service healthcheck] FAISS Service is healthy"
    exit 0
  fi
  sleep $WAIT_SECONDS
done

echo "[FAISS Service healthcheck] Failed after $MAX_ATTEMPTS attempts"
exit 1
