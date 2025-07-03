#!/bin/bash
# =============================================================================
# Dev Setup - Always Clean Rebuild Approach (WSL/Linux version)
# =============================================================================

echo "🚀 Production RAG System - Clean Setup (WSL)..."

# Check Python
if command -v python3 &> /dev/null; then
    version=$(python3 --version)
    echo "✅ Python: $version"
elif command -v python &> /dev/null; then
    version=$(python --version)
    echo "✅ Python: $version" 
    # Use python instead of python3 for rest of script
    PYTHON_CMD="python"
else
    echo "❌ Python 3.11+ required"
    exit 1
fi

# Set python command (default to python3)
PYTHON_CMD=${PYTHON_CMD:-python3}

# ALWAYS clean rebuild - delete existing .venv
if [ -d ".venv" ]; then
    echo "🗑️ Removing existing .venv (clean rebuild)..."
    rm -rf .venv
fi

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
$PYTHON_CMD -m venv .venv

# Activate venv
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install ALL dependencies for local development (everything)
echo "📚 Installing ALL dependencies for local development..."
python -m pip install -e ".[local]"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Fresh installation complete"

# Create missing directories
echo "📁 Creating project directories..."
dirs=("src/api/routes" "data" "logs")
for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✅ Created: $dir"
    fi
done

# Create missing __init__.py files
init_files=("src/api/routes/__init__.py")
for file in "${init_files[@]}"; do
    if [ ! -f "$file" ]; then
        touch "$file"
        echo "✅ Created: $file"
    fi
done

# Setup .env
if [ ! -f ".env" ]; then
    cp ".env.example" ".env"
    echo "✅ Created .env from .env.example"
fi

# Test everything works
echo "🧪 Testing installation..."

# Test 1: Basic config
if python -c "from src.shared.config import get_settings; print('✅ Config loaded successfully')" 2>/dev/null; then
    echo "✅ Config test passed"
else
    echo "❌ Config test failed"
    exit 1
fi

# Test 2: FastAPI imports
if python -c "import fastapi, uvicorn, rich; print('✅ FastAPI imports successful')" 2>/dev/null; then
    echo "✅ FastAPI imports passed"
else
    echo "❌ FastAPI import failed"
    exit 1
fi

# Test 3: Plugin system imports
if python -c "from src.api.plugin_registry import plugin_registry; from src.plugins.base import BasePlugin; print('✅ Plugin system imports successful')" 2>/dev/null; then
    echo "✅ Plugin system imports passed"
else
    echo "⚠️ Plugin system imports failed (some dependencies might be missing)"
fi

# Test 4: Run basic tests if available
if [ -f "src/tests/test_config.py" ]; then
    if python -m pytest src/tests/test_config.py -v 2>/dev/null; then
        echo "✅ Basic tests passed"
    else
        echo "⚠️ Some tests failed, but setup is functional"
    fi
fi

# Show what's installed
echo ""
echo "📋 Key installed packages:"
python -c "
import pkg_resources
packages = ['fastapi', 'uvicorn', 'pydantic', 'rich', 'watchdog', 'prometheus-fastapi-instrumentator']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  ✅ {pkg}: {version}')
    except:
        print(f'  ❌ {pkg}: not found')
"

echo ""
echo "🎉 Clean setup complete!"
echo ""
echo "Next steps:"
echo "  1. Test plugin system: python -c \"from src.api.main import create_app; print('App created successfully')\""
echo "  2. Test API: python -m src.api.main"
echo "  3. Run tests: python test_api.py"
echo ""
echo "💡 Tip: Always run ./dev_setup.sh after changing pyproject.toml"
echo "💡 Use 'source .venv/bin/activate' to activate the environment"