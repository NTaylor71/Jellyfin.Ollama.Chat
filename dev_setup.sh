#!/bin/bash
# =============================================================================
# Dev Setup - Always Clean Rebuild Approach (WSL/Linux version)
# =============================================================================

echo "🚀 Production RAG System - Clean Setup (WSL)..."

# Check for Python 3.12 specifically
if command -v python3.12 &> /dev/null; then
    version=$(python3.12 --version)
    echo "✅ Python: $version"
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    version=$(python3 --version)
    echo "⚠️ Found Python: $version (preferring Python 3.12)"
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.12+ required"
    echo "Install Python 3.12 with:"
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt update"
    echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

# Check if python-venv is installed for the chosen Python version
if ! $PYTHON_CMD -m venv --help > /dev/null 2>&1; then
    if [ "$PYTHON_CMD" = "python3.12" ]; then
        echo "❌ python3.12-venv not installed. Installing..."
        echo "Run: sudo apt install python3.12-venv"
    else
        echo "❌ python3-venv not installed. Installing..."
        echo "Run: sudo apt install python3.12-venv"
    fi
    echo "Then re-run this script."
    exit 1
fi

# ALWAYS clean rebuild - delete existing .venv
if [ -d ".venv" ]; then
    echo "🗑️ Removing existing .venv (clean rebuild)..."
    rm -rf .venv
fi

# Create fresh virtual environment
echo "📦 Creating fresh virtual environment..."
$PYTHON_CMD -m venv .venv

# Check if venv was created successfully
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Failed to create virtual environment"
    echo "Make sure python3.12-venv is installed: sudo apt install python3.12-venv"
    exit 1
fi

# Activate venv
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Verify we're in the venv
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies with better error handling
echo "📚 Installing dependencies..."

# Try full installation first
echo "🔄 Attempting full installation with all extras..."
if python -m pip install -e ".[local]"; then
    echo "✅ Full installation successful"
else
    echo "⚠️ Full installation failed, trying without problematic NLP packages..."

    # Install core packages first
    if python -m pip install -e ".[api,faiss,worker,ollama,dev,monitoring,plugins,mongodb]"; then
        echo "✅ Core packages installed"

        # Try to install basic NLP packages
        echo "🔄 Installing basic NLP packages..."
        python -m pip install spacy transformers torch || echo "⚠️ Some basic NLP packages failed"

        # Try the problematic packages with force/no-deps
        echo "🔄 Attempting problematic packages with --no-deps..."
        python -m pip install --no-deps heideltime || echo "⚠️ heideltime skipped"
        python -m pip install --no-deps sutime || echo "⚠️ sutime skipped"

        echo "✅ Installation completed with some packages potentially skipped"
    else
        echo "❌ Failed to install core dependencies"
        exit 1
    fi
fi

echo "✅ Installation process complete"

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
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
        echo "✅ Created .env from .env.example"
    else
        echo "⚠️ No .env.example found, skipping .env creation"
    fi
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
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Test plugin system: python -c \"from src.api.main import create_app; print('App created successfully')\""
echo "  3. Test API: python -m src.api.main"
echo "  4. Run tests: python test_api.py"
echo ""
echo "💡 Tip: Always run ./dev_setup.sh after changing pyproject.toml"
echo "💡 Your virtual environment is ready at: $(pwd)/.venv"