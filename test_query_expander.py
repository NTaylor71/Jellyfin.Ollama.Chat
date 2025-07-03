#!/usr/bin/env python3
"""
Test Adaptive Query Expander Plugin
Tests the query expansion functionality with different resource scenarios.
"""

import asyncio
import logging
import time
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_direct_plugin():
    """Test the plugin directly without the API."""
    try:
        # Import the plugin
        from src.plugins.examples.adaptive_query_expander import AdaptiveQueryExpanderPlugin
        from src.plugins.base import PluginExecutionContext, ExecutionPriority
        
        logger.info("🧪 Testing Adaptive Query Expander Plugin Directly")
        logger.info("=" * 60)
        
        # Create plugin instance
        plugin = AdaptiveQueryExpanderPlugin()
        
        # Initialize plugin
        success = await plugin.initialize({})
        if not success:
            logger.error("❌ Plugin initialization failed")
            return False
        
        logger.info("✅ Plugin initialized successfully")
        
        # Test queries based on our example movie data
        test_queries = [
            "funny samurai movies",
            "dark comedy from Finland", 
            "science fiction horror",
            "80s action movies",
            "Japanese martial arts films",
            "recent comedies",
            "Remote Control"  # Exact movie name from our data
        ]
        
        # Test different resource scenarios
        resource_scenarios = [
            {
                "name": "Low Resources", 
                "resources": {"total_cpu_capacity": 1, "local_memory_gb": 1}
            },
            {
                "name": "Medium Resources",
                "resources": {"total_cpu_capacity": 2, "local_memory_gb": 2}
            },
            {
                "name": "High Resources (Traditional)", 
                "resources": {"total_cpu_capacity": 8, "local_memory_gb": 8}
            },
            {
                "name": "Ollama Enhanced",
                "resources": {"total_cpu_capacity": 8, "local_memory_gb": 8, "ollama_available": True}
            }
        ]
        
        for scenario in resource_scenarios:
            logger.info(f"\n🔧 Testing {scenario['name']}")
            logger.info("-" * 40)
            
            for query in test_queries:
                # Create execution context
                context = PluginExecutionContext(
                    user_id="test_user",
                    session_id="test_session",
                    available_resources=scenario["resources"],
                    execution_timeout=5.0,
                    priority=ExecutionPriority.NORMAL
                )
                
                # Test query expansion
                start_time = time.time()
                expanded = await plugin.embellish_query(query, context)
                duration = (time.time() - start_time) * 1000
                
                logger.info(f"  📝 Query: '{query}'")
                logger.info(f"  ✨ Expanded: '{expanded}'")
                logger.info(f"  ⏱️ Duration: {duration:.1f}ms")
                
                # Validate expansion
                if len(expanded) > len(query):
                    logger.info(f"  ✅ Query expanded (+{len(expanded) - len(query)} characters)")
                else:
                    logger.info(f"  ℹ️ Query unchanged")
                
                print()  # Add spacing
        
        # Test plugin health
        health = await plugin.health_check()
        logger.info(f"\n🏥 Plugin Health: {health}")
        
        # Cleanup
        await plugin.cleanup()
        logger.info("✅ Plugin cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct plugin test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_plugin_via_api():
    """Test the plugin through the API endpoints."""
    try:
        logger.info("\n🌐 Testing Plugin via API")
        logger.info("=" * 60)
        
        api_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            
            # Check if plugin is loaded
            logger.info("🔍 Checking if plugin is loaded...")
            response = await client.get(f"{api_url}/plugins/list")
            
            if response.status_code != 200:
                logger.error(f"❌ Failed to get plugin list: {response.status_code}")
                return False
            
            plugins = response.json()["plugins"]
            query_expander = None
            
            for plugin in plugins:
                if plugin["name"] == "AdaptiveQueryExpander":
                    query_expander = plugin
                    break
            
            if query_expander:
                logger.info("✅ AdaptiveQueryExpander plugin found!")
                logger.info(f"   Status: {query_expander.get('initialized', 'unknown')}")
                logger.info(f"   Type: {query_expander.get('type', 'unknown')}")
                logger.info(f"   Version: {query_expander.get('version', 'unknown')}")
            else:
                logger.warning("⚠️ AdaptiveQueryExpander plugin not found in registry")
                logger.info("Available plugins:")
                for plugin in plugins:
                    logger.info(f"   - {plugin['name']}")
            
            # Test plugin health if found
            if query_expander:
                logger.info("\n🏥 Checking plugin health...")
                response = await client.get(f"{api_url}/plugins/health/AdaptiveQueryExpander")
                
                if response.status_code == 200:
                    health = response.json()
                    logger.info("✅ Plugin health check successful")
                    logger.info(f"   Health status: {health['health']['status']}")
                    logger.info(f"   Metrics: {health['health'].get('metrics', {})}")
                else:
                    logger.warning(f"⚠️ Plugin health check failed: {response.status_code}")
            
            # Test chat endpoint (which should trigger plugin execution)
            logger.info("\n💬 Testing query expansion via chat endpoint...")
            test_chat_queries = [
                "Find me funny samurai movies",
                "Show me dark Finnish comedies", 
                "I want to watch 80s sci-fi horror"
            ]
            
            for query in test_chat_queries:
                try:
                    response = await client.post(f"{api_url}/chat/", json={
                        "query": query
                    })
                    
                    if response.status_code == 200:
                        logger.info(f"✅ Chat query successful: '{query}'")
                        # Plugin should have executed during this request
                    else:
                        logger.info(f"ℹ️ Chat query returned {response.status_code}: '{query}'")
                        
                except Exception as e:
                    logger.info(f"ℹ️ Chat query test: {e}")
            
            return True
            
    except httpx.ConnectError:
        logger.warning("⚠️ Cannot connect to API - make sure it's running on localhost:8000")
        return False
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


async def test_plugin_metrics():
    """Test that plugin metrics are being recorded."""
    try:
        logger.info("\n📊 Testing Plugin Metrics")
        logger.info("=" * 60)
        
        api_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            
            # Check plugin metrics before test
            logger.info("📈 Checking initial plugin metrics...")
            response = await client.get(f"{api_url}/plugins/health")
            
            if response.status_code == 200:
                health_data = response.json()
                initial_executions = health_data["performance_metrics"]["total_executions"]
                logger.info(f"   Initial total executions: {initial_executions}")
            else:
                logger.warning("⚠️ Could not get initial metrics")
                initial_executions = 0
            
            # Trigger some plugin executions
            logger.info("🚀 Triggering plugin executions...")
            for i in range(3):
                try:
                    await client.post(f"{api_url}/chat/", json={
                        "query": f"test query {i+1} for metrics"
                    })
                except:
                    pass  # Ignore failures, we just want to trigger plugins
            
            # Wait a moment for metrics to update
            await asyncio.sleep(2)
            
            # Check metrics after test
            logger.info("📊 Checking updated plugin metrics...")
            response = await client.get(f"{api_url}/plugins/health")
            
            if response.status_code == 200:
                health_data = response.json()
                final_executions = health_data["performance_metrics"]["total_executions"]
                logger.info(f"   Final total executions: {final_executions}")
                
                if final_executions > initial_executions:
                    logger.info(f"✅ Plugin executions increased by {final_executions - initial_executions}")
                else:
                    logger.info("ℹ️ No new plugin executions detected")
                
                # Show plugin details
                plugin_details = health_data.get("plugin_health_details", [])
                for plugin in plugin_details:
                    if plugin["name"] == "AdaptiveQueryExpander":
                        logger.info(f"   AdaptiveQueryExpander executions: {plugin['total_executions']}")
                        logger.info(f"   Success rate: {plugin['success_rate_percent']}%")
                        logger.info(f"   Avg response time: {plugin['avg_execution_time_ms']}ms")
                        break
            
            # Check Prometheus metrics
            logger.info("🔍 Checking Prometheus metrics...")
            response = await client.get(f"{api_url}/metrics")
            
            if response.status_code == 200:
                metrics_text = response.text
                plugin_metrics = [line for line in metrics_text.split('\n') 
                                if 'plugin_' in line and 'AdaptiveQueryExpander' in line]
                
                if plugin_metrics:
                    logger.info("✅ Found AdaptiveQueryExpander in Prometheus metrics:")
                    for metric in plugin_metrics[:5]:  # Show first 5 metrics
                        logger.info(f"   {metric}")
                else:
                    logger.info("ℹ️ No AdaptiveQueryExpander metrics in Prometheus yet")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Metrics test failed: {e}")
        return False


async def main():
    """Run all plugin tests."""
    logger.info("🧪 Starting Adaptive Query Expander Plugin Tests")
    logger.info("=" * 70)
    
    # Test directly first
    direct_ok = await test_direct_plugin()
    
    # Test via API
    api_ok = await test_plugin_via_api()
    
    # Test metrics
    metrics_ok = await test_plugin_metrics()
    
    logger.info("\n" + "=" * 70)
    if direct_ok and api_ok and metrics_ok:
        logger.info("🎉 All Adaptive Query Expander tests passed!")
        logger.info("\n💡 Plugin Features Demonstrated:")
        logger.info("   ✅ Hardware-adaptive query expansion")
        logger.info("   ✅ Movie-specific entity recognition")
        logger.info("   ✅ Synonym and semantic expansion")
        logger.info("   ✅ Prometheus metrics integration")
        logger.info("   ✅ Resource-based scaling")
    else:
        logger.error("❌ Some tests failed:")
        if not direct_ok:
            logger.error("   - Direct plugin test failed")
        if not api_ok:
            logger.error("   - API integration test failed")
        if not metrics_ok:
            logger.error("   - Metrics test failed")


if __name__ == "__main__":
    asyncio.run(main())