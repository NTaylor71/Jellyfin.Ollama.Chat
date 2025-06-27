#!/bin/bash

# One-click FAISS RAG bootstrap (Linux/macOS)
# • Deletes any existing .venv
# • Requires Python 3.13+
# • Creates .venv and installs in editable mode
# • No global env persistence — fully repo-isolated

set -e

echo "🔧 FAISS RAG: Starting local dev setup..."

# Validate Python 3.13+
if ! command -v python3.13 &> /dev/null
then
    echo "❌ Python 3.13 not found in PATH."
    exit 1
fi

PYTHON_EXECUTABLE="python3.13"
export PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE
export PIP_NO_NETWORK_SSL_VERIFY=1

echo "🐍 Using interpreter: $PYTHON_EXECUTABLE"

# ❌ Remove existing virtual environment
if [ -d ".venv" ]; then
    echo "🧼 Removing existing .venv..."
    rm -rf .venv
fi

# ✅ Create new virtual environment
$PYTHON_EXECUTABLE -m venv .venv

# 🔁 Activate and install in editable mode
source .venv/bin/activate
pip install -e .[dev]

echo ""
echo "✅ FAISS RAG .venv ready and activated."
