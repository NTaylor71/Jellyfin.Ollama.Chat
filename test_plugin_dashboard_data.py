#!/usr/bin/env python3
"""
Generate Plugin Dashboard Data
Creates realistic plugin execution data to test the Grafana dashboard.
"""

import asyncio
import json
import random
import time
import httpx
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_QUERIES = [
    "Show me action movies with Tom Cruise",
    "Find romantic comedies from the 2000s",
    "Search for sci-fi films about AI",
    "Looking for thriller movies with good ratings",
    "Find animated movies for kids",
    "Show me documentaries about space",
    "Find horror movies from the 1980s",
    "Search for comedy movies with Will Ferrell",
    "Looking for drama films about family",
    "Find superhero movies from Marvel"
]

async def send_chat_request(query: str, session_id: str = None) -> Dict[str, Any]:
    """Send a chat request to the API to trigger plugin execution."""
    url = f"{API_BASE_URL}/chat/"
    
    payload = {
        "query": query,
        "context": {
            "session_id": session_id or f"test_session_{random.randint(1000, 9999)}",
            "user_id": f"test_user_{random.randint(1, 100)}"
        },
        "max_results": random.randint(3, 10),
        "include_metadata": True
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "status_code": response.status_code}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}", "data": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def check_api_health() -> bool:
    """Check if the API is healthy and responding."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/health", timeout=5)
            return response.status_code == 200
    except:
        return False

async def get_current_metrics() -> Dict[str, Any]:
    """Get current metrics from the API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/metrics", timeout=10)
            if response.status_code == 200:
                return {"success": True, "metrics": response.text}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def generate_plugin_load(num_requests: int = 20, delay_range: tuple = (1, 3)) -> None:
    """Generate load on the plugin system by sending multiple chat requests."""
    print(f"Generating plugin load with {num_requests} requests...")
    
    results = {
        "successful": 0,
        "failed": 0,
        "total_time": 0
    }
    
    for i in range(num_requests):
        query = random.choice(TEST_QUERIES)
        print(f"  [{i+1}/{num_requests}] Sending: '{query[:50]}...'")
        
        start_time = time.time()
        result = await send_chat_request(query)
        end_time = time.time()
        
        request_time = end_time - start_time
        results["total_time"] += request_time
        
        if result["success"]:
            results["successful"] += 1
            print(f"    ✓ Success ({request_time:.2f}s)")
        else:
            results["failed"] += 1
            print(f"    ✗ Failed: {result.get('error', 'Unknown error')} ({request_time:.2f}s)")
        
        # Random delay between requests
        if i < num_requests - 1:
            delay = random.uniform(delay_range[0], delay_range[1])
            await asyncio.sleep(delay)
    
    return results

async def run_burst_test(bursts: int = 3, requests_per_burst: int = 5) -> None:
    """Run burst tests to create interesting concurrency patterns."""
    print(f"\nRunning burst test: {bursts} bursts of {requests_per_burst} requests each...")
    
    for burst in range(bursts):
        print(f"\n--- Burst {burst + 1}/{bursts} ---")
        
        # Send multiple requests concurrently
        tasks = []
        for i in range(requests_per_burst):
            query = random.choice(TEST_QUERIES)
            task = send_chat_request(query, f"burst_{burst}_{i}")
            tasks.append(task)
        
        # Wait for all requests in this burst to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        
        print(f"  Burst completed in {end_time - start_time:.2f}s")
        print(f"  Success: {successful}/{requests_per_burst}, Failed: {failed}/{requests_per_burst}")
        
        # Wait between bursts
        if burst < bursts - 1:
            print("  Waiting 10 seconds before next burst...")
            await asyncio.sleep(10)

def analyze_plugin_metrics(metrics_text: str) -> Dict[str, Any]:
    """Analyze plugin metrics from the metrics endpoint."""
    analysis = {
        "plugin_executions": 0,
        "plugin_health": {},
        "plugin_memory": {},
        "plugin_cpu": {},
        "total_plugins": 0,
        "healthy_plugins": 0
    }
    
    lines = metrics_text.split('\n')
    
    for line in lines:
        if line.startswith('plugin_executions_total{') and not line.startswith('#'):
            # Extract plugin execution count
            if 'status="success"' in line:
                value = float(line.split()[-1])
                analysis["plugin_executions"] += value
        
        elif line.startswith('plugin_health_status{') and not line.startswith('#'):
            # Extract plugin health
            if 'plugin_name="' in line:
                plugin_name = line.split('plugin_name="')[1].split('"')[0]
                value = float(line.split()[-1])
                analysis["plugin_health"][plugin_name] = value == 1.0
        
        elif line.startswith('plugin_memory_usage_bytes{') and not line.startswith('#'):
            # Extract memory usage
            if 'plugin_name="' in line:
                plugin_name = line.split('plugin_name="')[1].split('"')[0]
                value = float(line.split()[-1])
                analysis["plugin_memory"][plugin_name] = value / (1024 * 1024)  # Convert to MB
        
        elif line.startswith('plugins_total '):
            analysis["total_plugins"] = int(float(line.split()[-1]))
        
        elif line.startswith('plugins_healthy_total '):
            analysis["healthy_plugins"] = int(float(line.split()[-1]))
    
    return analysis

async def main():
    """Main test function."""
    print("=== Plugin Dashboard Data Generation Test ===\n")
    
    # Check API health
    print("1. Checking API health...")
    if not await check_api_health():
        print("   ✗ API is not responding. Please ensure the API service is running.")
        return 1
    print("   ✓ API is healthy and responding")
    
    # Get initial metrics
    print("\n2. Getting initial metrics...")
    initial_metrics = await get_current_metrics()
    if not initial_metrics["success"]:
        print(f"   ✗ Failed to get metrics: {initial_metrics['error']}")
        return 1
    
    initial_analysis = analyze_plugin_metrics(initial_metrics["metrics"])
    print(f"   ✓ Initial state: {initial_analysis['total_plugins']} plugins, {initial_analysis['healthy_plugins']} healthy")
    print(f"     Plugin executions so far: {initial_analysis['plugin_executions']}")
    
    # Generate sequential load
    print("\n3. Generating sequential plugin load...")
    sequential_results = await generate_plugin_load(15, (0.5, 2.0))
    print(f"   ✓ Sequential test completed:")
    print(f"     Success: {sequential_results['successful']}")
    print(f"     Failed: {sequential_results['failed']}")
    print(f"     Average time: {sequential_results['total_time']/(sequential_results['successful'] + sequential_results['failed']):.2f}s")
    
    # Wait a bit
    print("\n4. Waiting 5 seconds before burst test...")
    await asyncio.sleep(5)
    
    # Generate burst load
    print("\n5. Generating burst load for concurrency testing...")
    await run_burst_test(3, 4)
    
    # Get final metrics
    print("\n6. Getting final metrics...")
    await asyncio.sleep(2)  # Let metrics settle
    final_metrics = await get_current_metrics()
    if final_metrics["success"]:
        final_analysis = analyze_plugin_metrics(final_metrics["metrics"])
        
        print(f"   ✓ Final state: {final_analysis['total_plugins']} plugins, {final_analysis['healthy_plugins']} healthy")
        print(f"     Total plugin executions: {final_analysis['plugin_executions']}")
        print(f"     Executions added this test: {final_analysis['plugin_executions'] - initial_analysis['plugin_executions']}")
        
        if final_analysis["plugin_health"]:
            print("     Plugin health status:")
            for plugin, healthy in final_analysis["plugin_health"].items():
                status = "HEALTHY" if healthy else "UNHEALTHY"
                print(f"       {plugin}: {status}")
        
        if final_analysis["plugin_memory"]:
            print("     Plugin memory usage:")
            for plugin, memory_mb in final_analysis["plugin_memory"].items():
                print(f"       {plugin}: {memory_mb:.1f} MB")
    
    print("\n=== Test Complete ===")
    print("\nYou can now check the Grafana dashboard at http://localhost:3000")
    print("Look for the 'Plugin Performance Monitoring' dashboard in the 'Production RAG' folder.")
    print("The dashboard should show:")
    print("- Plugin health overview with current counts")
    print("- Plugin execution rates (if recent activity)")
    print("- Plugin memory usage by plugin")
    print("- Plugin health status table")
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))