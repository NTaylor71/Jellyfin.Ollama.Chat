import os
import asyncio
import httpx
import json
from uuid import uuid4
from src.config import VECTORDB_URL, r
from src.api.load_query_embellishers import load_query_embellishers
from src.config import GPU_QUEUE, GPU_RESULT_PREFIX, GPU_ERROR_PREFIX

VECTOR_DIM = 4096

# Load plugins (can be hot-reloaded later)
query_embellishers = load_query_embellishers()

async def process_job(job_obj: dict) -> str:
    """
    Process a chat job from the Redis queue.
    """
    query = job_obj.get("query", "")
    user_id = job_obj.get("user_id", "unknown")

    print(f"💬 Received query: {query}")

    # Step 1: Apply query embellishers
    for embellisher in query_embellishers:
        query = embellisher(query)

    # Step 2: Request embedding via GPU queue
    query_vector = await request_gpu_embedding(query)

    if not query_vector:
        return "Failed to generate embedding."

    # Step 3: Search FAISS
    search_results = await search_faiss(query_vector)

    # Step 4: Format response
    formatted_response = format_results(query, search_results)

    return formatted_response

async def request_gpu_embedding(prompt: str) -> list:
    """
    Submit GPU embedding job to Redis queue and wait for result.
    """
    embedding_job_id = str(uuid4())
    gpu_job = {"job_id": embedding_job_id, "prompt": prompt}

    await r.rpush(GPU_QUEUE, json.dumps(gpu_job))

    print(f"⏳ Waiting for GPU embedding job {embedding_job_id}...")

    for _ in range(60):  # wait up to 60 seconds
        result = await r.get(f"{GPU_RESULT_PREFIX}{embedding_job_id}")
        error = await r.get(f"{GPU_ERROR_PREFIX}{embedding_job_id}")

        if result:
            embedding = json.loads(result.decode())
            if len(embedding) != VECTOR_DIM:
                print(f"⚠️ Embedding dimension mismatch. Expected {VECTOR_DIM}, got {len(embedding)}.")
                return []
            return embedding

        if error:
            print(f"❌ GPU embedding job failed: {error.decode()}")
            return []

        await asyncio.sleep(1)

    print(f"❌ GPU embedding job {embedding_job_id} timed out.")
    return []

async def search_faiss(query_vector: list) -> list:
    """
    Perform a search against the FAISS service.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{VECTORDB_URL}/search", json={"query_vector": query_vector, "top_k": 5})
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            print(f"⚠️ FAISS search failed: {e}")
            return []

def format_results(query: str, results: list) -> str:
    """
    Format FAISS search results into a human-readable string.
    """
    if not results:
        return f"Sorry, no results found for: {query}"

    lines = [f"Results for '{query}':"]
    for idx, result in enumerate(results, 1):
        item = result.get("document", "Unknown item")
        lines.append(f"{idx}. {item}")

    return "\n".join(lines)
