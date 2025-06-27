import asyncio
import httpx

FAISS_URL = "http://localhost:6333"

sample_vectors = [
    {
        "id": "vec-1",
        "vector": [0.1 * i for i in range(4096)],
        "metadata": {"title": "The Matrix", "year": 1999}
    },
    {
        "id": "vec-2",
        "vector": [0.2 * i for i in range(4096)],
        "metadata": {"title": "Pulp Fiction", "year": 1994}
    }
]

async def test_faiss_service():
    async with httpx.AsyncClient(timeout=30) as client:
        # Test Add
        print("📤 Adding sample vectors...")
        add_response = await client.post(f"{FAISS_URL}/add", json={"vectors": sample_vectors})
        assert add_response.status_code == 200, "Add request failed."
        print(f"✅ Add response: {add_response.json()}")

        # Test Search
        print("🔍 Searching for nearest neighbors...")
        search_payload = {"query_vector": [0.1 * i for i in range(4096)], "top_k": 2}
        search_response = await client.post(f"{FAISS_URL}/search", json=search_payload)
        assert search_response.status_code == 200, "Search request failed."
        print(f"✅ Search response: {search_response.json()}")

        # Test Health
        print("🩺 Testing FAISS Service health endpoint...")
        health_response = await client.get(f"{FAISS_URL}/health")
        assert health_response.status_code == 200, "Healthcheck failed."
        print(f"✅ Healthcheck response: {health_response.json()}")

if __name__ == "__main__":
    asyncio.run(test_faiss_service())
