import os
import httpx

from src.config import VECTORDB_URL

client = httpx.AsyncClient(timeout=30)

async def faiss_add(vectors: list[dict]) -> dict:
    """Send add request to FAISS Service."""
    url = f"{VECTORDB_URL}/add"
    response = await client.post(url, json={"vectors": vectors})
    response.raise_for_status()
    return response.json()

async def faiss_search(query_vector: list[float], top_k: int = 5) -> dict:
    """Send search request to FAISS Service."""
    url = f"{VECTORDB_URL}/search"
    response = await client.post(url, json={"query_vector": query_vector, "top_k": top_k})
    response.raise_for_status()
    return response.json()

async def faiss_delete(ids: list[str]) -> dict:
    """Send delete request to FAISS Service."""
    url = f"{VECTORDB_URL}/delete"
    response = await client.post(url, json={"ids": ids})
    response.raise_for_status()
    return response.json()

async def faiss_healthcheck() -> bool:
    """Check FAISS Service health endpoint."""
    url = f"{VECTORDB_URL}/health"
    try:
        response = await client.get(url)
        return response.status_code == 200
    except httpx.RequestError:
        return False
