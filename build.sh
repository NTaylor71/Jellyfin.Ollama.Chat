#!/bin/bash

COMPOSE_FILE="docker-compose.dev.yml"

# Allow override via first argument
if [ -n "$1" ]; then
  COMPOSE_FILE="$1"
fi

echo "🔧 Building Docker images..."
docker compose -f "$COMPOSE_FILE" build

echo "🚀 Launching Docker containers..."
docker compose -f "$COMPOSE_FILE" up
