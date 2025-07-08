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

# Install dependencies with controlled approach (match Docker)
echo "📚 Installing dependencies..."

# NO FALLBACKS - Install specific package groups with --no-cache-dir like Docker
echo "🔄 Installing with controlled dependencies (matching Docker approach)..."
if python -m pip install --no-cache-dir -e ".[api,faiss,worker,ollama,dev,monitoring,plugins,mongodb,nlp]"; then
    echo "✅ Controlled installation successful"
else
    echo "❌ Controlled installation failed - NO FALLBACKS"
    echo "❌ This means there are real dependency conflicts that need to be fixed"
    exit 1
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

# Note: Ollama now uses shared model-data volume, no separate directory needed

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

# Test everything works - ACTUAL VENV PACKAGES (NO CHEATING)
echo "🧪 Testing installation with REAL venv-specific packages..."

# Test 1: NLP packages that should ONLY be in venv
if python -c "import ollama; print('✅ Ollama available in venv')" 2>/dev/null; then
    echo "✅ Ollama venv test passed"
else
    echo "❌ Ollama not available - venv installation failed"
    exit 1
fi

# Test 2: Gensim (large ML package, definitely venv-only)
if python -c "import gensim; print('✅ Gensim available in venv')" 2>/dev/null; then
    echo "✅ Gensim venv test passed"
else
    echo "❌ Gensim not available - venv installation failed"
    exit 1
fi

# Test 3: FastAPI (should be venv-only in most systems)
if python -c "import fastapi, uvicorn; print('✅ FastAPI available in venv')" 2>/dev/null; then
    echo "✅ FastAPI venv test passed"
else
    echo "❌ FastAPI not available - venv installation failed"
    exit 1
fi

# Test 4: SpaCy (NLP package, definitely venv-only)
if python -c "import spacy; print('✅ SpaCy available in venv')" 2>/dev/null; then
    echo "✅ SpaCy venv test passed"
else
    echo "❌ SpaCy not available - venv installation failed"
    exit 1
fi

# Test 5: Our actual project imports (FAIL FAST if broken)
if python -c "from src.shared.config import get_settings; get_settings(); print('✅ Project config working')" 2>/dev/null; then
    echo "✅ Project imports test passed"
else
    echo "❌ Project imports failed - installation broken"
    exit 1
fi

# Show what's installed - ACTUAL IMPORTANT PACKAGES
echo ""
echo "📋 Key NLP and project packages installed in venv:"
python -c "
import pkg_resources
packages = ['ollama', 'gensim', 'spacy', 'transformers', 'torch', 'fastapi', 'redis', 'pymongo']
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
echo "⚠️  IMPORTANT: You must activate the virtual environment manually:"
echo ""
echo "    source .venv/bin/activate"
echo ""
echo "Next steps after activation:"
echo "  2. Test plugin system: python -c \"from src.api.main import create_app; print('App created successfully')\""
echo "  3. Test API: python -m src.api.main"
echo "  4. Run tests: python test_api.py"
echo ""
echo "🐳 Docker deployment with integrated GPU support:"
echo "  docker compose -f docker-compose.dev.yml up"
echo "  (Requires NVIDIA Docker runtime - will fail hard if not available)"
echo ""
echo "💡 Tip: Always run ./dev_setup.sh after changing pyproject.toml"
echo "💡 Your virtual environment is ready at: $(pwd)/.venv"
echo ""
echo "🚀 Quick activation command: source .venv/bin/activate"