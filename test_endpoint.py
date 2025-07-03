#!/usr/bin/env python3
"""
Test Chat Endpoint
Quick test to determine the correct endpoint path.
"""

import asyncio
import httpx

async def test_endpoints():
    """Test both /chat and /chat/ endpoints."""
    endpoints = [
        "http://localhost:8000/chat",
        "http://localhost:8000/chat/"
    ]
    
    payload = {"query": "test"}
    
    for url in endpoints:
        print(f"Testing {url}...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=5)
                print(f"  Status: {response.status_code}")
                if response.status_code < 400:
                    print(f"  ✓ SUCCESS: {url} works!")
                    print(f"  Response: {response.text[:100]}...")
                else:
                    print(f"  ✗ FAILED: {response.status_code}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        print()

if __name__ == "__main__":
    asyncio.run(test_endpoints())