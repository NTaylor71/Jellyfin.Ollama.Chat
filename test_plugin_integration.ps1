#!/usr/bin/env pwsh
# =============================================================================
# Test Plugin System Integration
# =============================================================================

Write-Host "🧪 Testing Plugin System Integration..." -ForegroundColor Cyan

# Set UTF-8 encoding for Python output
$env:PYTHONIOENCODING = "utf-8"

# Ensure we're in virtual environment
if (-not $env:VIRTUAL_ENV) {
    Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
}

$testsPassed = 0
$totalTests = 0

function Test-Step {
    param(
        [string]$Description,
        [scriptblock]$Test
    )
    
    $global:totalTests++
    Write-Host "`n📋 Test $global:totalTests: $Description" -ForegroundColor Blue
    
    try {
        $result = & $Test
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ PASSED: $Description" -ForegroundColor Green
            $global:testsPassed++
            return $true
        } else {
            Write-Host "❌ FAILED: $Description" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "❌ FAILED: $Description - $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Test 1: Basic imports
Test-Step "Import plugin system components" {
    python -c "
from src.api.plugin_registry import plugin_registry
from src.api.plugin_watcher import plugin_watcher
from src.plugins.base import BasePlugin, PluginType
from src.shared.hardware_config import get_hardware_profile
print('✅ All plugin imports successful')
"
}

# Test 2: FastAPI app creation with plugins
Test-Step "Create FastAPI app with plugin system" {
    python -c "
from src.api.main import create_app
app = create_app()
print('✅ FastAPI app with plugin system created successfully')
print(f'✅ Routes: {[route.path for route in app.routes]}')
"
}

# Test 3: Plugin registry initialization
Test-Step "Initialize plugin registry" {
    python -c "
import asyncio
from src.api.plugin_registry import plugin_registry

async def test_registry():
    await plugin_registry.initialize()
    status = await plugin_registry.get_plugin_status()
    print('SUCCESS: Plugin registry initialized')
    print('Total plugins:', status['total_plugins'])
    print('Enabled plugins:', status['enabled_plugins'])
    print('Plugin types:', list(status['plugins_by_type'].keys()))
    return status

result = asyncio.run(test_registry())
"
}

# Test 4: Hardware configuration
Test-Step "Test hardware configuration detection" {
    python -c "
import asyncio
from src.shared.hardware_config import get_hardware_profile, get_resource_limits

async def test_hardware():
    profile = await get_hardware_profile()
    limits = await get_resource_limits()
    print(f'SUCCESS: Hardware profile: {profile.local_cpu_cores}C/{profile.local_cpu_threads}T, {profile.local_memory_gb}GB RAM')
    print('CPU capacity:', limits['total_cpu_capacity'])
    print('GPU available:', limits['gpu_available'])
    return profile, limits

result = asyncio.run(test_hardware())
"
}

# Test 5: Example plugin loading
Test-Step "Load example plugins" {
    python -c "
import asyncio
from src.api.plugin_registry import plugin_registry

async def test_example_plugins():
    await plugin_registry.initialize()
    
    # Get query embellisher plugins
    from src.plugins.base import PluginType
    query_plugins = await plugin_registry.get_plugins_by_type(PluginType.QUERY_EMBELLISHER)
    
    print(f'SUCCESS: Found {len(query_plugins)} query embellisher plugins')
    for plugin in query_plugins:
        metadata = plugin.metadata
        print(f'  - {metadata.name} v{metadata.version}: {metadata.description}')
        
        # Test health check
        health = await plugin.health_check()
        print('    Health:', health['status'])
    
    return query_plugins

result = asyncio.run(test_example_plugins())
"
}

# Test 6: Plugin routes availability
Test-Step "Test plugin management routes" {
    python -c "
from src.api.main import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)

# Test root endpoint
response = client.get('/')
assert response.status_code == 200
print('SUCCESS: Root endpoint working')

# Check that plugin routes are registered
routes = [route.path for route in app.routes]
plugin_routes = [r for r in routes if r.startswith('/plugins')]
print('SUCCESS: Plugin routes found:', plugin_routes)

# Test that the routes exist (they might fail without async setup, but should be registered)
expected_routes = [
    '/plugins/status',
    '/plugins/list', 
    '/plugins/types',
    '/plugins/watcher/status'
]

for route in expected_routes:
    if route in routes:
        print(f'SUCCESS: Route registered: {route}')
    else:
        print(f'ERROR: Route missing: {route}')
        raise Exception(f'Missing route: {route}')
"
}

# Test 7: Plugin execution simulation
Test-Step "Simulate plugin execution" {
    python -c "
import asyncio
from src.api.plugin_registry import plugin_registry
from src.plugins.base import PluginType, PluginExecutionContext

async def test_plugin_execution():
    await plugin_registry.initialize()
    
    # Get query embellisher plugins
    plugins = await plugin_registry.get_plugins_by_type(PluginType.QUERY_EMBELLISHER)
    
    if plugins:
        plugin = plugins[0]
        metadata = plugin.metadata
        print(f'SUCCESS: Testing plugin: {metadata.name}')
        
        # Create execution context
        context = PluginExecutionContext(
            user_id='test_user',
            session_id='test_session',
            request_id='test_request'
        )
        
        # Test plugin execution
        test_query = 'find movies about space'
        result = await plugin.safe_execute(test_query, context)
        
        print(f'SUCCESS: Plugin execution: success={result.success}')
        if result.success:
            print(f'SUCCESS: Enhanced query: {result.data}')
        else:
            print(f'WARNING: Plugin error: {result.error_message}')
        
        return result
    else:
        print('INFO: No query embellisher plugins found to test')
        return None

result = asyncio.run(test_plugin_execution())
"
}

# Summary
Write-Host "`n📊 Test Summary:" -ForegroundColor Cyan
Write-Host "✅ Passed: $testsPassed / $totalTests tests" -ForegroundColor Green

if ($testsPassed -eq $totalTests) {
    Write-Host "`n🎉 All plugin integration tests PASSED!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Integrate plugins into chat route" -ForegroundColor White
    Write-Host "  2. Test end-to-end plugin execution" -ForegroundColor White
    Write-Host "  3. Test hot-reload functionality" -ForegroundColor White
} else {
    Write-Host "`n⚠️ Some tests failed. Check the output above." -ForegroundColor Yellow
    Write-Host "Plugin system may need fixes before proceeding." -ForegroundColor Yellow
}

Write-Host "`n💡 Run 'python -m src.api.main' to start the API with plugin system" -ForegroundColor Blue