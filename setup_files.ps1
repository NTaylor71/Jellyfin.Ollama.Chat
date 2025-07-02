#!/usr/bin/env pwsh
# =============================================================================
# Setup Script - Create All Project Files
# =============================================================================
# Run this script to automatically create the complete project structure

Write-Host "🚀 Setting up Production RAG System files..." -ForegroundColor Cyan

# Create directory structure
$dirs = @(
    "src/shared",
    "src/api/routes",
    "src/api/services", 
    "src/api/models",
    "src/faiss_service",
    "src/redis_worker",
    "src/plugins/examples",
    "src/tests/integration",
    "docker/api",
    "docker/faiss",
    "docker/worker",
    "docker/base",
    "docker/monitoring",
    "scripts",
    "docs",
    "data",
    "logs"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "✅ Created directory: $dir" -ForegroundColor Green
}

# Create pyproject.toml
Write-Host "`n📦 Creating pyproject.toml..." -ForegroundColor Yellow
@'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "production-rag-system"
version = "2.0.0"
description = "Production-ready FAISS RAG system with Redis queue and hot-reloadable plugins"
authors = [
    {name = "Your Name", email = "your.email@domain.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"

# Core dependencies shared across all services
dependencies = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "python-dotenv>=1.0.0",
    "structlog>=23.2.0",
    "rich>=13.7.0",
    "typer>=0.9.0",
    "httpx>=0.26.0",
    "redis>=5.0.0",
    "asyncio-mqtt>=0.16.0",
]

[project.optional-dependencies]
# API service dependencies
api = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "python-multipart>=0.0.6",
    "slowapi>=0.1.9",  # Rate limiting
]

# FAISS service dependencies
faiss = [
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
]

# Worker dependencies
worker = [
    "celery[redis]>=5.3.0",
    "kombu>=5.3.0",
]

# Ollama integration
ollama = [
    "ollama>=0.1.7",
]

# Development dependencies
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.6.0",
    "pre-commit>=3.5.0",
    "httpx>=0.26.0",  # For testing
]

# All dependencies for complete installation
all = [
    "production-rag-system[api,faiss,worker,ollama,dev]"
]

[project.scripts]
rag-api = "src.api.main:main"
rag-faiss = "src.faiss_service.main:main"
rag-worker = "src.redis_worker.main:main"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = ["src/tests"]
pythonpath = ["src"]
'@ | Out-File -FilePath "pyproject.toml" -Encoding UTF8

# Create .env.example
Write-Host "📝 Creating .env.example..." -ForegroundColor Yellow
@'
# =============================================================================
# Production RAG System - Environment Configuration
# =============================================================================

# App Environment
ENV=localhost

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_QUEUE=chat:queue

# Ollama Services
OLLAMA_CHAT_BASE_URL=http://localhost:12434
OLLAMA_CHAT_MODEL=llama3.2:3b
OLLAMA_EMBED_BASE_URL=http://localhost:12435
OLLAMA_EMBED_MODEL=nomic-embed-text

# FAISS Vector Database
VECTORDB_URL=http://localhost:6333

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_URL=http://localhost:8000

# Jellyfin Integration
JELLYFIN_URL=http://your-jellyfin-server:8096
JELLYFIN_API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=structured

# Development Features
ENABLE_CORS=true
ENABLE_API_DOCS=true
ENABLE_METRICS=true

# Plugin System
PLUGIN_DIRECTORY=./src/plugins
PLUGIN_HOT_RELOAD=true
'@ | Out-File -FilePath ".env.example" -Encoding UTF8

# Create .gitignore
Write-Host "🚫 Creating .gitignore..." -ForegroundColor Yellow
@'
# Environment and secrets
.env
.env.local
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
build/
dist/
*.egg-info/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
.DS_Store

# Data and logs
data/
logs/
*.log

# FAISS indices
*.index
*.faiss
*.pkl

# Docker
docker-compose.override.yml

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS files
Thumbs.db
'@ | Out-File -FilePath ".gitignore" -Encoding UTF8

# Create basic config.py
Write-Host "⚙️ Creating src/shared/config.py..." -ForegroundColor Yellow
@'
"""Smart configuration management with automatic environment detection."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with automatic environment detection."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Environment
    ENV: str = Field(default="localhost")
    
    # Redis
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_QUEUE: str = Field(default="chat:queue")
    
    # Ollama
    OLLAMA_CHAT_BASE_URL: str = Field(default="http://localhost:12434")
    OLLAMA_CHAT_MODEL: str = Field(default="llama3.2:3b")
    OLLAMA_EMBED_BASE_URL: str = Field(default="http://localhost:12435")
    OLLAMA_EMBED_MODEL: str = Field(default="nomic-embed-text")
    
    # FAISS
    VECTORDB_URL: str = Field(default="http://localhost:6333")
    
    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_URL: str = Field(default="http://localhost:8000")
    
    # Jellyfin
    JELLYFIN_URL: str = Field(default="http://localhost:8096")
    JELLYFIN_API_KEY: str = Field(default="")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="structured")
    
    # Features
    ENABLE_CORS: bool = Field(default=True)
    ENABLE_API_DOCS: bool = Field(default=True)
    ENABLE_METRICS: bool = Field(default=True)
    
    # Plugins
    PLUGIN_DIRECTORY: str = Field(default="./src/plugins")
    PLUGIN_HOT_RELOAD: bool = Field(default=True)
    
    @property
    def is_localhost(self) -> bool:
        return self.ENV == "localhost"
    
    @property
    def is_docker(self) -> bool:
        return self.ENV == "docker"
    
    @property
    def is_production(self) -> bool:
        return self.ENV == "production"
    
    @property
    def redis_url(self) -> str:
        host = "redis" if self.is_docker else self.REDIS_HOST
        return f"redis://{host}:{self.REDIS_PORT}/0"
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            Path("./data"),
            Path("./logs"),
            Path(self.PLUGIN_DIRECTORY)
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
'@ | Out-File -FilePath "src/shared/config.py" -Encoding UTF8

# Create __init__.py files
$init_files = @(
    "src/__init__.py",
    "src/shared/__init__.py",
    "src/api/__init__.py",
    "src/api/routes/__init__.py",
    "src/api/services/__init__.py",
    "src/api/models/__init__.py",
    "src/faiss_service/__init__.py",
    "src/redis_worker/__init__.py",
    "src/plugins/__init__.py",
    "src/plugins/examples/__init__.py",
    "src/tests/__init__.py",
    "src/tests/integration/__init__.py",
    "scripts/__init__.py"
)

foreach ($file in $init_files) {
    "" | Out-File -FilePath $file -Encoding UTF8
}

# Create basic test
Write-Host "🧪 Creating basic test..." -ForegroundColor Yellow
@'
"""Basic configuration tests."""

import pytest
from src.shared.config import get_settings


def test_settings_load():
    """Test that settings can be loaded."""
    settings = get_settings()
    assert settings is not None
    assert settings.ENV in ["localhost", "docker", "production"]


def test_directories_created():
    """Test that required directories exist."""
    settings = get_settings()
    from pathlib import Path
    assert Path("./data").exists()
    assert Path("./logs").exists()
    assert Path(settings.PLUGIN_DIRECTORY).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'@ | Out-File -FilePath "src/tests/test_config.py" -Encoding UTF8

# Create simple dev_setup.ps1
Write-Host "🔧 Creating dev_setup.ps1..." -ForegroundColor Yellow
@'
#!/usr/bin/env pwsh
# Simple development setup script

Write-Host "🚀 Setting up Production RAG System..." -ForegroundColor Cyan

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3.11+ required" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
if ($IsWindows -or $env:OS -eq "Windows_NT") {
    & ".venv\Scripts\Activate.ps1"
} else {
    & ".venv/bin/Activate.ps1"
}

# Install dependencies
Write-Host "📚 Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -e ".[dev,all]"

# Setup environment
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "✅ Created .env from .env.example" -ForegroundColor Green
}

# Run basic test
Write-Host "🧪 Running basic tests..." -ForegroundColor Yellow
python -m pytest src/tests/test_config.py -v

Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review and update .env file" -ForegroundColor White
Write-Host "  2. Run tests: python -m pytest" -ForegroundColor White
Write-Host "  3. Start development!" -ForegroundColor White
'@ | Out-File -FilePath "dev_setup.ps1" -Encoding UTF8

# Create README
Write-Host "📚 Creating README.md..." -ForegroundColor Yellow
@'
# 🚀 Production RAG System

Professional FAISS RAG system with Redis queue processing and hot-reloadable plugins.

## Quick Start

1. **Setup**: `./dev_setup.ps1`
2. **Test**: `python -m pytest`
3. **Develop**: Edit code and enjoy!

## Key Features

- 🔍 FAISS vector search
- ⚡ Redis queue processing  
- 🔌 Hot-reloadable plugins
- 🐳 Docker-first design
- 📊 Production monitoring

## Architecture

```
API ──► Redis Queue ──► FAISS Service
 │                          │
 └──► Plugin System         └──► Vector Search
```

## Development

```bash
# Start development environment
./dev_setup.ps1

# Run tests
python -m pytest

# Check configuration
python -c "from src.shared.config import get_settings; print(get_settings())"
```

## Configuration

Key settings in `.env`:

- `ENV`: localhost, docker, or production
- `REDIS_HOST`: Redis server location
- `OLLAMA_CHAT_BASE_URL`: Chat model endpoint
- `VECTORDB_URL`: FAISS service endpoint

## Next Steps

This is a foundational setup. Additional components will be added incrementally:

1. ✅ Basic configuration and structure
2. 🔄 FastAPI application
3. 🔄 FAISS service
4. 🔄 Redis worker
5. 🔄 Plugin system
6. 🔄 Docker containers
7. 🔄 Monitoring

---

Happy coding! 🚀
'@ | Out-File -FilePath "README.md" -Encoding UTF8

Write-Host "`n🎉 All files created successfully!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Run: ./dev_setup.ps1" -ForegroundColor White
Write-Host "  2. Test: python -m pytest" -ForegroundColor White
Write-Host "  3. Start developing!" -ForegroundColor White
