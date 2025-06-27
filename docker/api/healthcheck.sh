#!/bin/bash
set -e

MAX_ATTEMPTS=30
WAIT_SECONDS=2

for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[API healthcheck] Attempt $i of $MAX_ATTEMPTS..."
  if curl -sf http://localhost:8000/health > /dev/null; then
    echo "[API healthcheck] API is healthy"
    exit 0
  fi
  sleep $WAIT_SECONDS
done

echo "[API healthcheck] Failed after $MAX_ATTEMPTS attempts"
exit 1
