#!/usr/bin/env python3
"""
Test script for the RAG API endpoints.
Run this to verify the API is working correctly.
"""

import asyncio
import time
import httpx
from rich.console import Console
from rich.table import Table

console = Console()


async def test_api():
    """Test all API endpoints."""
    
    base_url = "http://localhost:8000"
    
    console.print("🧪 Testing Production RAG API", style="bold blue")
    console.print(f"Base URL: {base_url}\n")
    
    async with httpx.AsyncClient() as client:
        
        # Test 1: Root endpoint
        console.print("1. Testing root endpoint...", style="yellow")
        try:
            response = await client.get(f"{base_url}/")
            if response.status_code == 200:
                data = response.json()
                console.print(f"✅ Root: {data['name']} v{data['version']}", style="green")
            else:
                console.print(f"❌ Root failed: {response.status_code}", style="red")
        except Exception as e:
            console.print(f"❌ Root error: {e}", style="red")
            return
        
        # Test 2: Health checks
        console.print("\n2. Testing health endpoints...", style="yellow")
        
        health_endpoints = [
            ("/health", "Basic Health"),
            ("/health/ready", "Readiness"),
            ("/health/live", "Liveness"),
            ("/health/ping", "Ping"),
            ("/health/version", "Version")
        ]
        
        for endpoint, name in health_endpoints:
            try:
                response = await client.get(f"{base_url}{endpoint}")
                if response.status_code == 200:
                    console.print(f"✅ {name}: OK", style="green")
                else:
                    console.print(f"❌ {name}: {response.status_code}", style="red")
            except Exception as e:
                console.print(f"❌ {name}: {e}", style="red")
        
        # Test 3: Detailed health
        console.print("\n3. Testing detailed health...", style="yellow")
        try:
            response = await client.get(f"{base_url}/health/detailed")
            if response.status_code == 200:
                data = response.json()
                console.print(f"✅ Detailed health: {data['status']}", style="green")
                console.print(f"   Environment: {data['environment']}")
                console.print(f"   Dependencies: {data['dependencies']}")
            else:
                console.print(f"❌ Detailed health: {response.status_code}", style="red")
        except Exception as e:
            console.print(f"❌ Detailed health: {e}", style="red")
        
        # Test 4: Chat test endpoint
        console.print("\n4. Testing chat test endpoint...", style="yellow")
        try:
            response = await client.post(f"{base_url}/chat/test?query=science fiction movies")
            if response.status_code == 200:
                data = response.json()
                console.print(f"✅ Chat test: {data['status']}", style="green")
                console.print(f"   Query: {data['query']}")
                console.print(f"   Results: {len(data['results'])} items")
            else:
                console.print(f"❌ Chat test: {response.status_code}", style="red")
        except Exception as e:
            console.print(f"❌ Chat test: {e}", style="red")
        
        # Test 5: Full chat workflow
        console.print("\n5. Testing full chat workflow...", style="yellow")
        try:
            # Submit query
            chat_request = {
                "query": "movies about artificial intelligence",
                "max_results": 3,
                "include_metadata": True
            }
            
            response = await client.post(f"{base_url}/chat/", json=chat_request)
            if response.status_code == 200:
                submit_data = response.json()
                job_id = submit_data["job_id"]
                console.print(f"✅ Chat submitted: Job {job_id[:8]}...", style="green")
                
                # Wait a moment for processing
                await asyncio.sleep(0.5)
                
                # Get result
                response = await client.get(f"{base_url}/chat/result/{job_id}")
                if response.status_code == 200:
                    result_data = response.json()
                    console.print(f"✅ Chat result: {result_data['status']}", style="green")
                    if result_data['results']:
                        console.print(f"   Found: {len(result_data['results'])} movies")
                        console.print(f"   Response: {result_data['response'][:100]}...")
                else:
                    console.print(f"❌ Chat result: {response.status_code}", style="red")
            else:
                console.print(f"❌ Chat submit: {response.status_code}", style="red")
        except Exception as e:
            console.print(f"❌ Chat workflow: {e}", style="red")
        
        # Test 6: List jobs
        console.print("\n6. Testing job listing...", style="yellow")
        try:
            response = await client.get(f"{base_url}/chat/jobs")
            if response.status_code == 200:
                data = response.json()
                console.print(f"✅ Job list: {data['total']} total jobs", style="green")
            else:
                console.print(f"❌ Job list: {response.status_code}", style="red")
        except Exception as e:
            console.print(f"❌ Job list: {e}", style="red")
    
    # Summary
    console.print("\n🎉 API testing complete!", style="bold green")
    console.print("\nNext steps:", style="bold blue")
    console.print("• Visit http://localhost:8000/docs for interactive API docs")
    console.print("• Visit http://localhost:8000/health/detailed for service status")
    console.print("• Try the chat endpoints with your own queries")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_api())
    except KeyboardInterrupt:
        console.print("\n🛑 Test interrupted", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Test failed: {e}", style="red")


if __name__ == "__main__":
    main()
