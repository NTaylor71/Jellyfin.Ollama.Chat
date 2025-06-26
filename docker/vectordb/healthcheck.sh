#!/bin/bash
set -e

MAX_ATTEMPTS=30
WAIT_SECONDS=2

for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[Qdrant healthcheck] Attempt $i of $MAX_ATTEMPTS..."
  if curl -sf http://localhost:6333/collections > /dev/null; then
    echo "[Qdrant healthcheck] Qdrant is healthy"
    exit 0
  fi
  sleep $WAIT_SECONDS
done

echo "[Qdrant healthcheck] Failed after $MAX_ATTEMPTS attempts"
exit 1
