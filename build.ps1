<#
Jellyfin.Ollama.Chat Build Script
• Loads .env
• Validates docker + compose
• Builds and launches containers
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ComposeFile = "docker-compose.dev.yml"

Write-Host "🔧 Jellychat: Starting build..." -ForegroundColor Cyan

# Validate Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker not found in PATH." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ComposeFile)) {
    Write-Host "❌ Compose file '$ComposeFile' not found." -ForegroundColor Red
    exit 1
}

# Load .env if present
if (Test-Path ".env") {
    Write-Host "📦 Loading .env..." -ForegroundColor DarkGray
    Get-Content ".env" |
        Where-Object { $_ -match "=" -and -not $_.Trim().StartsWith("#") } |
        ForEach-Object {
            $parts = $_ -split '=', 2
            if ($parts.Length -eq 2) {
                $key = $parts[0].Trim()
                $value = $parts[1].Trim()
                Set-Item -Path "Env:$key" -Value $value
            }
        }
} else {
    Write-Host "⚠️ .env file not found — continuing without it." -ForegroundColor Yellow
}

Write-Host "🔨 Building Docker containers..." -ForegroundColor Cyan
docker compose -f $ComposeFile build

if (-not $?) {
    Write-Host "❌ Build failed." -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Launching stack..." -ForegroundColor Cyan
docker compose -f $ComposeFile up

Write-Host "`n✅ Done."
