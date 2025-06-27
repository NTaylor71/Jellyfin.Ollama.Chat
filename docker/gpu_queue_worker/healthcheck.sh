#!/bin/bash
set -e

MAX_ATTEMPTS=30
WAIT_SECONDS=2

for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[GPU Queue Worker healthcheck] Attempt $i of $MAX_ATTEMPTS..."
  if pgrep -f gpu_queue_worker > /dev/null; then
    echo "[GPU Queue Worker healthcheck] Process is running."
    exit 0
  fi
  sleep $WAIT_SECONDS
done

echo "[GPU Queue Worker healthcheck] Failed after $MAX_ATTEMPTS attempts."
exit 1
