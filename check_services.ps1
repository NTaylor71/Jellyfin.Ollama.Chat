# Check what services are defined in docker-compose.dev.yml
Write-Host "🔍 Checking Docker Compose Services" -ForegroundColor Blue
Write-Host "=" * 40

if (!(Test-Path "docker-compose.dev.yml")) {
    Write-Host "❌ docker-compose.dev.yml not found" -ForegroundColor Red
    exit 1
}

# List all services
Write-Host "`n📋 Available Services:" -ForegroundColor Cyan
try {
    $services = docker compose -f docker-compose.dev.yml config --services
    if ($services) {
        $services | ForEach-Object {
            Write-Host "   • $_" -ForegroundColor Green
        }

        Write-Host "`n🎯 Suggested Commands:" -ForegroundColor Blue
        Write-Host "Start minimal: docker compose -f docker-compose.dev.yml up -d redis api" -ForegroundColor White

        # Check for worker-like services
        $workerServices = $services | Where-Object { $_ -match "worker|queue" }
        if ($workerServices) {
            Write-Host "Start with worker: docker compose -f docker-compose.dev.yml up -d redis api $($workerServices -join ' ')" -ForegroundColor White
        }

        # Check for monitoring services
        $monitoringServices = $services | Where-Object { $_ -match "prometheus|grafana" }
        if ($monitoringServices) {
            Write-Host "Add monitoring: docker compose -f docker-compose.dev.yml up -d $($monitoringServices -join ' ')" -ForegroundColor White
        }

    } else {
        Write-Host "❌ No services found" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Failed to read services: $($_.Exception.Message)" -ForegroundColor Red
}

# Check current running containers
Write-Host "`n🏃 Currently Running:" -ForegroundColor Cyan
try {
    $running = docker compose -f docker-compose.dev.yml ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"
    if ($running -and $running.Length -gt 1) {
        Write-Host $running -ForegroundColor Yellow
    } else {
        Write-Host "   No containers running" -ForegroundColor Gray
    }
} catch {
    Write-Host "   Could not check running containers" -ForegroundColor Gray
}

Write-Host "`n💡 Next Step:" -ForegroundColor Yellow
Write-Host "Run: ./start_services.ps1 core" -ForegroundColor White
