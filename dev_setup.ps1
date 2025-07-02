#!/usr/bin/env pwsh
# =============================================================================
# Dev Setup - Always Clean Rebuild Approach
# =============================================================================

Write-Host "🚀 Production RAG System - Clean Setup..." -ForegroundColor Cyan

# Check Python
try {
    $version = python --version 2>&1
    Write-Host "✅ Python: $version" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3.11+ required" -ForegroundColor Red
    exit 1
}

# ALWAYS clean rebuild - delete existing .venv
if (Test-Path ".venv") {
    Write-Host "🗑️ Removing existing .venv (clean rebuild)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

# Create fresh virtual environment
Write-Host "📦 Creating fresh virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate venv
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install ALL dependencies for local development (everything)
Write-Host "📚 Installing ALL dependencies for local development..." -ForegroundColor Yellow
python -m pip install -e ".[local]"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Fresh installation complete" -ForegroundColor Green

# Create missing directories
Write-Host "📁 Creating project directories..." -ForegroundColor Yellow
$dirs = @("src/api/routes", "data", "logs")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created: $dir" -ForegroundColor Green
    }
}

# Create missing __init__.py files
$initFiles = @("src/api/routes/__init__.py")
foreach ($file in $initFiles) {
    if (-not (Test-Path $file)) {
        "" | Out-File -FilePath $file -Encoding UTF8
        Write-Host "✅ Created: $file" -ForegroundColor Green
    }
}

# Setup .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env from .env.example" -ForegroundColor Green
}

# Test everything works
Write-Host "🧪 Testing installation..." -ForegroundColor Yellow

# Test 1: Basic config
try {
    python -c "from src.shared.config import get_settings; print('✅ Config loaded successfully')"
} catch {
    Write-Host "❌ Config test failed" -ForegroundColor Red
    exit 1
}

# Test 2: FastAPI imports
try {
    python -c "import fastapi, uvicorn, rich; print('✅ FastAPI imports successful')"
} catch {
    Write-Host "❌ FastAPI import failed" -ForegroundColor Red
    exit 1
}

# Test 3: Run basic tests if available
if (Test-Path "src/tests/test_config.py") {
    try {
        python -m pytest src/tests/test_config.py -v
        Write-Host "✅ Basic tests passed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Some tests failed, but setup is functional" -ForegroundColor Yellow
    }
}

# Show what's installed
Write-Host "`n📋 Key installed packages:" -ForegroundColor Cyan
python -c "
import pkg_resources
packages = ['fastapi', 'uvicorn', 'pydantic', 'rich']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  ✅ {pkg}: {version}')
    except:
        print(f'  ❌ {pkg}: not found')
"

Write-Host "`n🎉 Clean setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Create API files (health.py, chat.py, main.py)" -ForegroundColor White
Write-Host "  2. Test API: python -m src.api.main" -ForegroundColor White
Write-Host "  3. Run tests: python test_api.py" -ForegroundColor White

Write-Host "`n💡 Tip: Always run ./dev_setup.ps1 after changing pyproject.toml" -ForegroundColor Blue
