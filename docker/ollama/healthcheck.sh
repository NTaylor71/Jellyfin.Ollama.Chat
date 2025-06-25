#!/bin/sh
MODELS="${OLLAMA_MODELS:-mistral}"
for MODEL in $(echo "$MODELS" | tr ',' ' '); do
  if ! ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL not yet ready"
    exit 1
  fi
done
