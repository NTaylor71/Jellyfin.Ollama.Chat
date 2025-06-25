#!/usr/bin/env bash

# -------------------------------
# Jellychat Dev Bootstrap (Unix)
# -------------------------------
# • Deletes .venv if present
# • Requires Python 3.13+
# • Creates fresh virtual env
# • Installs editable mode package
# -------------------------------

set -euo pipefail

echo "🧠 Jellychat dev setup (Linux/macOS)"

# Step 1: Locate Python 3.13+
PYTHON=""
if command -v python3.13 >/dev/null 2>&1; then
    PYTHON="python3.13"
elif command -v python3 >/dev/null 2>&1 && [[ $(python3 -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)') -ge 313 ]]; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1 && [[ $(python -c 'import sys; print(sys.version_info.major * 10 + sys.version_info.minor)') -ge 313 ]]; then
    PYTHON="python"
else
    echo "❌ Python 3.13+ is required but not found."
    exit 1
fi

echo "🐍 Using interpreter: $PYTHON"

# Step 2: Delete existing .venv
if [[ -d .venv ]]; then
    echo "🧼 Removing existing .venv..."
    rm -rf .venv
fi

# Step 3: Create venv
"$PYTHON" -m venv .venv

# Step 4: Activate and install
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]

echo ""
echo "✅ Jellychat .venv ready and activated."
