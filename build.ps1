param (
    [string]$ComposeFile = "docker-compose.dev.yml"
)
Write-Host "🔧 Building Docker images..."
docker compose -f $ComposeFile build
Write-Host "🚀 Launching Docker containers..."
docker compose -f $ComposeFile up
