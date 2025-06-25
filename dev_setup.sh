#!/usr/bin/env bash

set -euo pipefail

echo "🧠 Jellychat dev setup (Linux/macOS)"

# Find highest available python3
PYTHON=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
if [[ -z "$PYTHON" ]]; then
  echo "❌ No suitable Python 3.x interpreter found."
  exit 1
fi

echo "🐍 Using interpreter: $PYTHON"

# Remove any existing venv
if [[ -d ".venv" ]]; then
  echo "🧼 Removing existing .venv..."
  rm -rf .venv
fi

# Create new venv
"$PYTHON" -m venv .venv

# Activate and install
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo -e "\n✅ Jellychat .venv ready and activated."
