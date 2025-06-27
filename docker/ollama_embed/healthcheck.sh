#!/bin/sh

MODEL="${OLLAMA_MODEL:-nomic-embed-text}"

if ! ollama list | grep -q "$MODEL"; then
  echo "Model $MODEL not yet ready"
  exit 1
fi
