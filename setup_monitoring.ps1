# Setup monitoring directories and configuration files
Write-Host "🚀 Setting up comprehensive monitoring infrastructure..." -ForegroundColor Green

# Create all required monitoring directories
$monitoringDirs = @(
    "docker/monitoring",
    "docker/monitoring/grafana",
    "docker/monitoring/grafana/provisioning",
    "docker/monitoring/grafana/provisioning/datasources",
    "docker/monitoring/grafana/provisioning/dashboards",
    "docker/monitoring/grafana/provisioning/dashboards/json",
    "docker/monitoring/grafana/provisioning/dashboards/fastapi",
    "docker/monitoring/grafana/provisioning/dashboards/system"
)

foreach ($dir in $monitoringDirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Created directory: $dir" -ForegroundColor Cyan
    } else {
        Write-Host "📁 Directory exists: $dir" -ForegroundColor Yellow
    }
}

# Check if configuration files exist
$configFiles = @(
    @{Path="docker/monitoring/prometheus.yml"; Description="Prometheus configuration with FastAPI instrumentation"},
    @{Path="docker/monitoring/grafana/provisioning/datasources/prometheus.yml"; Description="Grafana datasource (comprehensive)"},
    @{Path="docker/monitoring/grafana/provisioning/dashboards/dashboard.yml"; Description="Dashboard provisioning config"},
    @{Path="docker/monitoring/grafana/provisioning/dashboards/json/production-rag-api.json"; Description="FastAPI dashboard JSON"}
)

foreach ($file in $configFiles) {
    if (Test-Path $file.Path) {
        Write-Host "✅ Configuration exists: $($file.Description)" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing configuration: $($file.Description)" -ForegroundColor Red
        Write-Host "   Expected at: $($file.Path)" -ForegroundColor Yellow
    }
}

# Test Prometheus configuration syntax
if (Test-Path "docker/monitoring/prometheus.yml") {
    Write-Host "`n🔍 Testing Prometheus configuration..." -ForegroundColor Blue
    try {
        $promTest = docker run --rm -v "${PWD}/docker/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus:v2.45.0 promtool check config /etc/prometheus/prometheus.yml 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Prometheus configuration is valid" -ForegroundColor Green
        } else {
            Write-Host "❌ Prometheus configuration has errors:" -ForegroundColor Red
            Write-Host $promTest -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ Failed to test Prometheus config: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Test Docker Compose syntax
Write-Host "`n🔍 Testing Docker Compose configuration..." -ForegroundColor Blue
try {
    $output = docker-compose -f docker-compose.dev.yml config 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker Compose configuration is valid" -ForegroundColor Green
    } else {
        Write-Host "❌ Docker Compose configuration has errors:" -ForegroundColor Red
        Write-Host $output -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Failed to test Docker Compose: $($_.Exception.Message)" -ForegroundColor Red
}

# Check if API is exposing metrics
Write-Host "`n📊 Checking API metrics availability..." -ForegroundColor Blue
try {
    $metricsTest = Invoke-WebRequest -Uri "http://localhost:8000/metrics" -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($metricsTest.StatusCode -eq 200) {
        Write-Host "✅ API metrics endpoint is accessible" -ForegroundColor Green
        $fastApiMetrics = ($metricsTest.Content | Select-String "fastapi_").Count
        Write-Host "   Found $fastApiMetrics FastAPI metrics" -ForegroundColor Cyan
    } else {
        Write-Host "❌ API metrics endpoint not accessible" -ForegroundColor Red
    }
} catch {
    Write-Host "⚠️ API not running or metrics not accessible" -ForegroundColor Yellow
    Write-Host "   Start API first: docker compose -f docker-compose.dev.yml up api" -ForegroundColor Yellow
}

Write-Host "`n📋 Next steps:" -ForegroundColor Blue
Write-Host "1. Copy all configuration files to their respective directories" -ForegroundColor White
Write-Host "2. Start monitoring stack: docker compose -f docker-compose.dev.yml up -d prometheus grafana" -ForegroundColor White
Write-Host "3. Access Grafana at http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "4. Check Prometheus targets at http://localhost:9090/targets" -ForegroundColor White
Write-Host "5. Import FastAPI dashboard from provisioned JSON" -ForegroundColor White

Write-Host "`n🎯 Production RAG API Monitoring Features:" -ForegroundColor Green
Write-Host "   • Request rates and response times (p50, p90, p95, p99)" -ForegroundColor Cyan
Write-Host "   • Error rates by status code" -ForegroundColor Cyan
Write-Host "   • In-progress requests monitoring" -ForegroundColor Cyan
Write-Host "   • Memory and CPU usage tracking" -ForegroundColor Cyan
Write-Host "   • Endpoint-specific metrics" -ForegroundColor Cyan
Write-Host "   • Health status monitoring" -ForegroundColor Cyan
