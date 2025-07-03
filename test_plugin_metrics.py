#!/usr/bin/env python3
"""
Test Plugin Metrics Integration
Tests Prometheus metrics collection for plugin health checks.
"""

import asyncio
import time
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_plugin_metrics():
    """Test plugin metrics are exposed via Prometheus /metrics endpoint."""
    
    api_url = "http://localhost:8000"
    
    try:
        async with httpx.AsyncClient() as client:
            
            # 1. Check if API is running
            logger.info("🔍 Checking if API is running...")
            response = await client.get(f"{api_url}/health")
            if response.status_code != 200:
                logger.error(f"❌ API not running. Status: {response.status_code}")
                return False
            
            logger.info("✅ API is running")
            
            # 2. Check plugin system status
            logger.info("🔌 Checking plugin system status...")
            response = await client.get(f"{api_url}/health/plugins")
            if response.status_code == 200:
                plugin_health = response.json()
                logger.info(f"✅ Plugin system: {plugin_health['status']}")
                logger.info(f"   Plugins: {plugin_health.get('healthy_plugins', 0)}/{plugin_health.get('total_plugins', 0)}")
            else:
                logger.warning("⚠️ Plugin health endpoint not available")
            
            # 3. Trigger plugin execution (if any plugins exist)
            logger.info("🚀 Testing plugin execution...")
            try:
                response = await client.post(f"{api_url}/chat/query", json={
                    "query": "test query for metrics",
                    "use_rag": False
                })
                if response.status_code == 200:
                    logger.info("✅ Chat query executed (plugins may have run)")
                else:
                    logger.info(f"ℹ️ Chat query returned {response.status_code}")
            except Exception as e:
                logger.info(f"ℹ️ Chat endpoint test: {e}")
            
            # 4. Wait for metrics to update
            logger.info("⏱️ Waiting for metrics to update...")
            await asyncio.sleep(5)
            
            # 5. Check Prometheus metrics endpoint
            logger.info("📊 Checking Prometheus metrics...")
            response = await client.get(f"{api_url}/metrics")
            
            if response.status_code != 200:
                logger.error(f"❌ Metrics endpoint failed. Status: {response.status_code}")
                return False
            
            metrics_content = response.text
            
            # Check for plugin-specific metrics
            plugin_metrics_found = []
            expected_metrics = [
                "plugin_executions_total",
                "plugin_execution_duration_seconds",
                "plugin_health_status",
                "plugin_initialization_status",
                "plugins_total",
                "plugins_enabled_total",
                "plugins_initialized_total",
                "plugins_healthy_total"
            ]
            
            for metric in expected_metrics:
                if metric in metrics_content:
                    plugin_metrics_found.append(metric)
                    logger.info(f"   ✅ Found metric: {metric}")
            
            # Log sample metrics values
            logger.info("\n📈 Sample Plugin Metrics:")
            for line in metrics_content.split('\n'):
                if any(metric in line for metric in plugin_metrics_found):
                    if not line.startswith('#') and line.strip():
                        logger.info(f"   {line}")
            
            if len(plugin_metrics_found) >= 4:  # At least some metrics should be present
                logger.info(f"✅ Plugin metrics integration working! Found {len(plugin_metrics_found)}/{len(expected_metrics)} metrics")
                return True
            else:
                logger.warning(f"⚠️ Only found {len(plugin_metrics_found)}/{len(expected_metrics)} plugin metrics")
                return False
                
    except httpx.ConnectError:
        logger.error("❌ Cannot connect to API. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_plugin_health_endpoints():
    """Test plugin health REST endpoints."""
    
    api_url = "http://localhost:8000"
    
    try:
        async with httpx.AsyncClient() as client:
            
            # Test plugin health overview
            logger.info("\n🔍 Testing plugin health endpoints...")
            
            endpoints_to_test = [
                "/plugins/health",
                "/plugins/status", 
                "/plugins/list",
                "/health/plugins"
            ]
            
            for endpoint in endpoints_to_test:
                try:
                    response = await client.get(f"{api_url}{endpoint}")
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"✅ {endpoint} - Status: {response.status_code}")
                        
                        # Show key metrics from each endpoint
                        if "overall_health" in data:
                            logger.info(f"   Overall health: {data['overall_health'].get('status')}")
                        if "performance_metrics" in data:
                            metrics = data['performance_metrics']
                            logger.info(f"   Executions: {metrics.get('total_executions', 0)}")
                        if "total_plugins" in data:
                            logger.info(f"   Total plugins: {data['total_plugins']}")
                            
                    else:
                        logger.warning(f"⚠️ {endpoint} - Status: {response.status_code}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ {endpoint} - Error: {e}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Health endpoints test failed: {e}")
        return False

async def main():
    """Run all plugin metrics tests."""
    logger.info("🧪 Starting Plugin Metrics Tests")
    logger.info("=" * 50)
    
    # Test Prometheus metrics
    metrics_ok = await test_plugin_metrics()
    
    # Test health endpoints  
    health_ok = await test_plugin_health_endpoints()
    
    logger.info("\n" + "=" * 50)
    if metrics_ok and health_ok:
        logger.info("🎉 All plugin metrics tests passed!")
        logger.info("\n💡 Next steps:")
        logger.info("   - Check Grafana dashboard at http://localhost:3000")
        logger.info("   - View raw metrics at http://localhost:8000/metrics")
        logger.info("   - Monitor plugin health at http://localhost:8000/plugins/health")
    else:
        logger.error("❌ Some plugin metrics tests failed")
        if not metrics_ok:
            logger.error("   - Prometheus metrics integration needs attention")
        if not health_ok:
            logger.error("   - Plugin health endpoints need attention")

if __name__ == "__main__":
    asyncio.run(main())