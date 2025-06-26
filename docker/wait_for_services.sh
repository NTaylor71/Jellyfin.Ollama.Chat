#!/bin/bash
set -e

MAX_ATTEMPTS=30
SLEEP_INTERVAL=2

wait_for_service() {
    local NAME=$1
    local URL=$2

    echo "⏳ Waiting for $NAME at $URL..."

    for i in $(seq 1 $MAX_ATTEMPTS); do
        if [[ "$URL" == *":6379/ping" ]]; then
            # Redis TCP check
            if redis-cli -h "${URL%%:*}" ping | grep -q "PONG"; then
                echo "✅ $NAME is ready."
                return 0
            fi
        else
            # HTTP check
            if curl -sSf "$URL" > /dev/null; then
                echo "✅ $NAME is ready."
                return 0
            fi
        fi
        echo "⏳ Attempt $i/$MAX_ATTEMPTS - $NAME not ready yet."
        sleep $SLEEP_INTERVAL
    done

    echo "❌ $NAME did not become ready in time."
    exit 1
}
