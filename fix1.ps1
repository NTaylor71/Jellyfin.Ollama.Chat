#!/usr/bin/env pwsh
# =============================================================================
# Quick Fix Script - Simple and Working
# =============================================================================

Write-Host "🔧 Fixing setup issues..." -ForegroundColor Cyan

# Fix pyproject.toml by replacing the entire file
Write-Host "📦 Fixing pyproject.toml..." -ForegroundColor Yellow

# Create the fixed pyproject.toml content
$pyprojectFixed = @"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "production-rag-system"
version = "2.0.0"
description = "Production-ready FAISS RAG system"
requires-python = ">=3.11"

dependencies = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-dotenv>=1.0.0",
    "pytest>=7.4.0",
    "httpx>=0.26.0"
]

[project.optional-dependencies]
dev = [
    "black>=23.0.0",
    "isort>=5.12.0"
]

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["src/tests"]
pythonpath = ["src"]
"@

# Write the fixed pyproject.toml
$pyprojectFixed | Out-File -FilePath "pyproject.toml" -Encoding UTF8
Write-Host "✅ Fixed pyproject.toml" -ForegroundColor Green

# Create a simple working dev_setup.ps1
Write-Host "🔧 Creating simple dev_setup.ps1..." -ForegroundColor Yellow

$devSetupFixed = @"
#!/usr/bin/env pwsh
Write-Host "🚀 Simple RAG System Setup..." -ForegroundColor Cyan

# Check Python
try {
    `$version = python --version 2>&1
    Write-Host "✅ Python: `$version" -ForegroundColor Green
} catch {
    Write-Host "❌ Python required" -ForegroundColor Red
    exit 1
}

# Create venv if needed
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate venv
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"

# Install core deps
Write-Host "📚 Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install pydantic pydantic-settings python-dotenv pytest

# Setup .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env" -ForegroundColor Green
}

# Test config
Write-Host "🧪 Testing..." -ForegroundColor Yellow
python -c "from src.shared.config import get_settings; print('✅ Config works!')"

Write-Host "🎉 Setup complete!" -ForegroundColor Green
"@

# Write the fixed dev_setup.ps1
$devSetupFixed | Out-File -FilePath "dev_setup.ps1" -Encoding UTF8
Write-Host "✅ Created simple dev_setup.ps1" -ForegroundColor Green

Write-Host "`n🎉 Fix complete!" -ForegroundColor Green
Write-Host "Now run: ./dev_setup.ps1" -ForegroundColor Cyan