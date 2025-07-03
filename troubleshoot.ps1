# Production RAG System Troubleshooting Script
Write-Host "🔧 Production RAG System Troubleshooting" -ForegroundColor Blue
Write-Host "=" * 50

# Check Docker
Write-Host "`n🐳 Docker Status:" -ForegroundColor Blue
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green

    $composeVersion = docker compose version
    Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not available" -ForegroundColor Red
    Write-Host "Install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check running containers
Write-Host "`n📦 Running Containers:" -ForegroundColor Blue
$containers = docker compose -f docker-compose.dev.yml ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"
if ($containers) {
    Write-Host $containers -ForegroundColor Cyan
} else {
    Write-Host "❌ No containers running" -ForegroundColor Red
    Write-Host "Start with: ./start_services.ps1" -ForegroundColor Yellow
}

# Check specific services
$services = @{
    "Redis" = @{
        "container" = "jellychat_redis"
        "port" = 6379
        "test" = "redis-cli ping"
    }
    "API" = @{
        "container" = "jellychat_api"
        "port" = 8000
        "test" = "curl http://localhost:8000/health"
    }
    "Queue Worker" = @{
        "container" = "jellychat_redis_queue_worker"
        "port" = $null
        "test" = "docker logs"
    }
}

Write-Host "`n🔍 Service Health Checks:" -ForegroundColor Blue
foreach ($serviceName in $services.Keys) {
    $service = $services[$serviceName]
    $containerName = $service.container

    Write-Host "`n  $serviceName ($containerName):" -ForegroundColor Cyan

    # Check if container exists and is running
    $containerStatus = docker ps --filter "name=$containerName" --format "{{.Status}}"
    if ($containerStatus) {
        Write-Host "    ✅ Container running: $containerStatus" -ForegroundColor Green

        # Show recent logs
        Write-Host "    📋 Recent logs:" -ForegroundColor Yellow
        $logs = docker logs --tail 5 $containerName 2>&1
        if ($logs) {
            $logs | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
        }
    } else {
        Write-Host "    ❌ Container not running" -ForegroundColor Red
    }
}

# Check ports
Write-Host "`n🌐 Port Accessibility:" -ForegroundColor Blue
$ports = @(8000, 6379, 9090, 3000)
foreach ($port in $ports) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            Write-Host "  ✅ Port $port accessible" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Port $port not accessible" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ❌ Port $port test failed" -ForegroundColor Red
    }
}

# Check Python environment
Write-Host "`n🐍 Python Environment:" -ForegroundColor Blue
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green

    # Check if we're in virtual environment
    if ($env:VIRTUAL_ENV) {
        Write-Host "✅ Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
    } else {
        Write-Host "⚠️ No virtual environment detected" -ForegroundColor Yellow
        Write-Host "Run: ./dev_setup.ps1" -ForegroundColor Yellow
    }

    # Check key packages
    $packages = @("redis", "httpx", "rich", "fastapi")
    foreach ($package in $packages) {
        try {
            $null = python -c "import $package; print('✅')" 2>$null
            Write-Host "✅ Package: $package" -ForegroundColor Green
        } catch {
            Write-Host "❌ Package missing: $package" -ForegroundColor Red
        }
    }

} catch {
    Write-Host "❌ Python not available" -ForegroundColor Red
}

# Check configuration files
Write-Host "`n⚙️ Configuration Files:" -ForegroundColor Blue
$configFiles = @(
    "docker-compose.dev.yml",
    ".env",
    "src/shared/config.py",
    "docker/monitoring/prometheus.yml"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
    }
}

# Quick fixes suggestions
Write-Host "`n💡 Quick Fixes:" -ForegroundColor Yellow
Write-Host "• Container not running: ./start_services.ps1 core" -ForegroundColor White
Write-Host "• Python issues: ./dev_setup.ps1" -ForegroundColor White
Write-Host "• Port conflicts: docker compose -f docker-compose.dev.yml down" -ForegroundColor White
Write-Host "• Check logs: docker compose -f docker-compose.dev.yml logs [service]" -ForegroundColor White
Write-Host "• Reset everything: docker compose -f docker-compose.dev.yml down -v" -ForegroundColor White

Write-Host "`n🧪 Run Tests:" -ForegroundColor Green
Write-Host "• Debug worker: python debug_worker.py" -ForegroundColor White
Write-Host "• Test metrics: python test_fastapi_metrics.py" -ForegroundColor White
Write-Host "• Full integration: python test_full_integration.py" -ForegroundColor White
