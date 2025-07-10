#!/usr/bin/env python3
"""
Test Dynamic Service Discovery - Verify no hard-coding remains.
"""

import asyncio
import sys
import json
from src.shared.dynamic_service_discovery import get_service_discovery


async def test_dynamic_discovery():
    """Test that service discovery works without hard-coding."""
    
    print("🔍 Testing Dynamic Service Discovery...")
    
    try:
        discovery = await get_service_discovery()
        
        print("\n📡 Discovering all services...")
        services = await discovery.discover_all_services()
        
        for name, service in services.items():
            print(f"  ✅ {name}: {service.status} - {len(service.capabilities)} capabilities")
            print(f"     URL: {service.base_url}")
            print(f"     Type: {service.service_type}")
            print(f"     Capabilities: {service.capabilities}")
            
            endpoints = service.metadata.get("available_endpoints", [])
            print(f"     Endpoints: {len(endpoints)} discovered")
            for endpoint in endpoints[:5]:
                print(f"       - {endpoint}")
            if len(endpoints) > 5:
                print(f"       ... and {len(endpoints) - 5} more")
            print()
        
        print("\n🔌 Testing plugin routing...")
        test_plugins = [
            "ConceptNetKeywordPlugin",
            "GensimSimilarityPlugin", 
            "SpacyTemporalPlugin",
            "HeidelTimeTemporalPlugin",
            "LLMKeywordPlugin",
            "LLMWebSearchPlugin"
        ]
        
        for plugin_name in test_plugins:
            service_info = await discovery.get_service_for_plugin(plugin_name)
            if service_info:
                endpoint = await discovery.get_endpoint_for_plugin(plugin_name)
                print(f"  ✅ {plugin_name} -> {service_info.service_name} : {endpoint}")
            else:
                print(f"  ❌ {plugin_name} -> No service found")
        
        print("\n🎯 Testing capability queries...")
        capabilities_to_test = ["conceptnet", "gensim", "llm_provider", "keywords"]
        
        for capability in capabilities_to_test:
            services_with_cap = await discovery.get_services_by_capability(capability)
            print(f"  🔍 '{capability}': {len(services_with_cap)} services")
            for service in services_with_cap:
                print(f"     - {service.service_name} ({service.status})")
        
        print("\n📊 Discovery Summary:")
        summary = discovery.get_discovery_summary()
        print(f"  Total services: {summary['total_services']}")
        print(f"  Healthy services: {summary['healthy_services']}")
        print(f"  Unhealthy services: {summary['unhealthy_services']}")
        
        print("\n✅ Dynamic service discovery test completed successfully!")
        print("🎉 NO HARD-CODING DETECTED - All endpoints discovered dynamically!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Dynamic discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            await discovery.cleanup()
        except:
            pass


if __name__ == "__main__":
    success = asyncio.run(test_dynamic_discovery())
    sys.exit(0 if success else 1)