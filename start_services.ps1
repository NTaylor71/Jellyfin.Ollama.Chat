# Start Production RAG System Services
param(
    [string]$Services = "core",  # core, monitoring, all
    [switch]$Background = $true
)

Write-Host "🚀 Starting Production RAG System Services..." -ForegroundColor Green

# First, let's check what services are actually available
Write-Host "🔍 Checking available services..." -ForegroundColor Blue
try {
    $availableServices = docker compose -f docker-compose.dev.yml config --services 2>$null
    if ($availableServices) {
        Write-Host "📋 Available services:" -ForegroundColor Cyan
        $availableServices | ForEach-Object { Write-Host "   • $_" -ForegroundColor White }
    }
} catch {
    Write-Host "⚠️ Could not list services" -ForegroundColor Yellow
}

# Define service groups based on common naming patterns
$ServiceGroups = @{
    "core" = @("redis", "api")  # Start with minimal set first
    "worker" = @("worker", "redis_worker", "queue_worker", "redis_queue_worker")  # Try different worker names
    "monitoring" = @("prometheus", "grafana")
    "faiss" = @("faiss_service", "faiss", "vectordb")
    "all" = @("redis", "api", "worker", "faiss_service", "prometheus", "grafana")
}

# Get services to start
$ServicesToStart = $ServiceGroups[$Services]
if (-not $ServicesToStart) {
    Write-Host "❌ Unknown service group: $Services" -ForegroundColor Red
    Write-Host "Available groups: core, worker, monitoring, faiss, all" -ForegroundColor Yellow
    exit 1
}

Write-Host "🎯 Attempting to start: $($ServicesToStart -join ', ')" -ForegroundColor Cyan

# Filter to only services that actually exist
$validServices = @()
if ($availableServices) {
    foreach ($service in $ServicesToStart) {
        if ($availableServices -contains $service) {
            $validServices += $service
        } else {
            Write-Host "⚠️ Service '$service' not found in docker-compose.dev.yml" -ForegroundColor Yellow
        }
    }
} else {
    # If we can't list services, try them all and let Docker handle errors
    $validServices = $ServicesToStart
}

if ($validServices.Count -eq 0) {
    Write-Host "❌ No valid services found to start" -ForegroundColor Red
    Write-Host "💡 Check your docker-compose.dev.yml file" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Starting valid services: $($validServices -join ', ')" -ForegroundColor Green

# Build and start services
$dockerArgs = @("compose", "-f", "docker-compose.dev.yml", "up")
if ($Background) {
    $dockerArgs += "-d"
}
$dockerArgs += "--build"
$dockerArgs += $validServices

Write-Host "🔧 Running: docker $($dockerArgs -join ' ')" -ForegroundColor Blue

try {
    & docker @dockerArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Services started successfully!" -ForegroundColor Green

        # Show service status
        Write-Host "`n📊 Service Status:" -ForegroundColor Blue
        docker compose -f docker-compose.dev.yml ps

        # Show useful URLs
        Write-Host "`n🌐 Service URLs:" -ForegroundColor Blue
        Write-Host "   API: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
        Write-Host "   Metrics: http://localhost:8000/metrics" -ForegroundColor Cyan

        if ($validServices -contains "prometheus") {
            Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor Cyan
        }
        if ($validServices -contains "grafana") {
            Write-Host "   Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Cyan
        }

        # If core services started, suggest adding worker
        if ($Services -eq "core" -and $validServices.Count -eq 2) {
            Write-Host "`n💡 Core services started. To add worker:" -ForegroundColor Yellow
            Write-Host "   ./start_services.ps1 worker" -ForegroundColor White
        }

        Write-Host "`n🧪 Ready for testing!" -ForegroundColor Green
        Write-Host "Run: python debug_worker.py" -ForegroundColor White

    } else {
        Write-Host "❌ Failed to start services" -ForegroundColor Red
        Write-Host "Check logs with: docker compose -f docker-compose.dev.yml logs" -ForegroundColor Yellow
    }

} catch {
    Write-Host "❌ Error starting services: $($_.Exception.Message)" -ForegroundColor Red
}

# Show what to do next
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n📋 Next Steps:" -ForegroundColor Blue
    Write-Host "1. Run: ./troubleshoot.ps1" -ForegroundColor White
    Write-Host "2. Then: python debug_worker.py" -ForegroundColor White
    Write-Host "3. Check worker service name in docker-compose.dev.yml" -ForegroundColor White
}
