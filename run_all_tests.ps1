#!/usr/bin/env pwsh
# =============================================================================
# Comprehensive Test Runner - All Unit and Integration Tests
# =============================================================================

Write-Host "🧪 Running Comprehensive Test Suite..." -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Gray

# Set UTF-8 encoding for Python output
$env:PYTHONIOENCODING = "utf-8"

# Ensure we're in virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
}

$global:totalTests = 0
$global:passedTests = 0
$global:failedTests = @()

function Test-PythonScript {
    param(
        [string]$ScriptPath,
        [string]$Description
    )
    
    $global:totalTests++
    Write-Host "`n📋 Test $global:totalTests`: $Description" -ForegroundColor Blue
    Write-Host "   Running: $ScriptPath" -ForegroundColor Gray
    
    try {
        $result = python $ScriptPath
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PASSED: $Description" -ForegroundColor Green
            $global:passedTests++
            return $true
        } else {
            Write-Host "❌ FAILED: $Description (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
            $global:failedTests += $ScriptPath
            return $false
        }
    } catch {
        Write-Host "❌ FAILED: $Description - $($_.Exception.Message)" -ForegroundColor Red
        $global:failedTests += $ScriptPath
        return $false
    }
}

function Test-PytestScript {
    param(
        [string]$ScriptPath,
        [string]$Description
    )
    
    $global:totalTests++
    Write-Host "`n📋 Test $global:totalTests`: $Description" -ForegroundColor Blue
    Write-Host "   Running: pytest $ScriptPath -v" -ForegroundColor Gray
    
    try {
        $result = python -m pytest $ScriptPath -v
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PASSED: $Description" -ForegroundColor Green
            $global:passedTests++
            return $true
        } else {
            Write-Host "❌ FAILED: $Description (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
            $global:failedTests += $ScriptPath
            return $false
        }
    } catch {
        Write-Host "❌ FAILED: $Description - $($_.Exception.Message)" -ForegroundColor Red
        $global:failedTests += $ScriptPath
        return $false
    }
}

Write-Host "`n🔧 PHASE 1: Core System Unit Tests" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Core system tests
Test-PythonScript "src/tests/test_config.py" "Configuration System"
Test-PythonScript "test_api.py" "API Endpoints"
Test-PythonScript "test_redis_queue.py" "Redis Queue System"
Test-PythonScript "test_fastapi_metrics.py" "FastAPI Metrics"

Write-Host "`n🔌 PHASE 2: Plugin System Unit Tests" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Plugin system core tests
Test-PythonScript "test_plugin_registry.py" "Plugin Registry"
Test-PythonScript "test_plugin_hot_reload.py" "Plugin Hot Reload"
Test-PythonScript "test_plugin_execution.py" "Plugin Execution"
Test-PythonScript "test_cpu_optimization.py" "CPU Optimization"
Test-PythonScript "test_plugin_config.py" "Plugin Configuration"

Write-Host "`n🚀 PHASE 3: Individual Plugin Tests" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Individual plugin tests
Test-PythonScript "test_query_expander.py" "Query Expander Plugin"
Test-PythonScript "test_embed_enhancer.py" "Embed Enhancer Plugin"
Test-PythonScript "test_faiss_crud_plugin.py" "FAISS CRUD Plugin"

Write-Host "`n📊 PHASE 4: Monitoring & Metrics Tests" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Monitoring tests
Test-PythonScript "test_plugin_metrics.py" "Plugin Metrics"
Test-PythonScript "test_plugin_performance_monitoring.py" "Performance Monitoring"
Test-PythonScript "test_dashboard_json.py" "Dashboard Configuration"
Test-PythonScript "test_plugin_dashboard_data.py" "Dashboard Data"

Write-Host "`n🔗 PHASE 5: Integration Tests" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Integration tests
Test-PythonScript "test_embed_enhancer_integration.py" "Embed Enhancer Integration"
Test-PythonScript "test_faiss_crud_integration.py" "FAISS CRUD Integration"
Test-PythonScript "test_full_integration.py" "Full System Integration"

Write-Host "`n🎯 PHASE 6: Plugin System Integration" -ForegroundColor Magenta
Write-Host "-" * 40 -ForegroundColor Gray

# Plugin integration test (PowerShell script) 
$global:totalTests++
Write-Host "`n📋 Test $global:totalTests`: Plugin System Integration" -ForegroundColor Blue
Write-Host "   Running: ./test_plugin_integration.ps1" -ForegroundColor Gray

# Store current variables to avoid interference
$savedTotalTests = $global:totalTests
$savedPassedTests = $global:passedTests

try {
    $integrationResult = & "./test_plugin_integration.ps1"
    
    # Restore our variables (the integration script has its own counters)
    $global:totalTests = $savedTotalTests  
    $global:passedTests = $savedPassedTests
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PASSED: Plugin System Integration" -ForegroundColor Green
        $global:passedTests++
    } else {
        Write-Host "❌ FAILED: Plugin System Integration" -ForegroundColor Red
        $global:failedTests += "test_plugin_integration.ps1"
    }
} catch {
    Write-Host "❌ FAILED: Plugin System Integration - $($_.Exception.Message)" -ForegroundColor Red
    $global:failedTests += "test_plugin_integration.ps1"
    
    # Restore our variables  
    $global:totalTests = $savedTotalTests
    $global:passedTests = $savedPassedTests
}

# Final Summary
Write-Host "`n" + "=" * 70 -ForegroundColor Gray
Write-Host "📊 COMPREHENSIVE TEST RESULTS" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Gray

Write-Host "`nSUMMARY:" -ForegroundColor White
Write-Host "✅ Passed: $global:passedTests / $global:totalTests tests" -ForegroundColor Green

if ($global:failedTests.Count -gt 0) {
    Write-Host "❌ Failed Tests:" -ForegroundColor Red
    foreach ($test in $global:failedTests) {
        Write-Host "   • $test" -ForegroundColor Red
    }
}

$successRate = [math]::Round(($global:passedTests / $global:totalTests) * 100, 1)
Write-Host "`nSuccess Rate: $successRate%" -ForegroundColor $(if ($successRate -ge 90) { "Green" } elseif ($successRate -ge 75) { "Yellow" } else { "Red" })

if ($global:passedTests -eq $global:totalTests) {
    Write-Host "`n🎉 ALL TESTS PASSED! System is ready for production." -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  • Deploy to production environment" -ForegroundColor White
    Write-Host "  • Monitor metrics and performance" -ForegroundColor White
    Write-Host "  • Run integration tests in staging" -ForegroundColor White
} elseif ($successRate -ge 90) {
    Write-Host "`n✅ Excellent test coverage! Minor issues to address." -ForegroundColor Green
} elseif ($successRate -ge 75) {
    Write-Host "`n⚠️ Good test coverage with some failures to investigate." -ForegroundColor Yellow
} else {
    Write-Host "`n🔧 Multiple test failures require attention before production." -ForegroundColor Red
}

Write-Host "`n💡 Individual test commands:" -ForegroundColor Blue
Write-Host "   python test_<name>.py                # Run specific test" -ForegroundColor Gray
Write-Host "   python -m pytest test_<name>.py -v   # Run with pytest verbose" -ForegroundColor Gray
Write-Host "   python test_full_integration.py      # Full integration test" -ForegroundColor Gray

Write-Host "`n🏁 Test suite completed!" -ForegroundColor Cyan