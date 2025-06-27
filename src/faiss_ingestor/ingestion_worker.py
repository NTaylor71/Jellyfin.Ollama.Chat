import asyncio
import json
import random
from src.config import r
import httpx

VECTORDB_URL = "http://faiss_service:6333"
FAISS_VECTOR_DIM = 4096  # Must match faiss_index.py

async def start_ingestion():
    print("📥 FAISS Ingestor is listening for ingestion jobs...")

    while True:
        try:
            queue_item = await r.blpop("chat:ingest", timeout=5)
            if not queue_item:
                continue

            _, raw_payload = queue_item
            job = json.loads(raw_payload)

            print(f"📦 Ingesting job: {job.get('job_id')}")

            vectors = job.get("vectors", [])
            if not vectors:
                print("⚠️ No vectors provided in job.")
                continue

            await send_to_faiss(vectors)
            print(f"✅ Successfully ingested {len(vectors)} vectors.")

        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            await asyncio.sleep(2)

async def send_to_faiss(vectors):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{VECTORDB_URL}/add", json={"vectors": vectors})
        response.raise_for_status()

def generate_fake_vectors(count=5, dim=FAISS_VECTOR_DIM):
    """
    Generate a list of fake vectors for ingestion.

    Args:
        count (int): Number of vectors to generate.
        dim (int): Dimensionality of each vector.

    Returns:
        list: A list of vector items with id, vector, and metadata.
    """
    return [
        {
            "id": f"item_{i}",
            "vector": [random.random() for _ in range(dim)],
            "metadata": {"title": f"Fake Item {i}"}
        }
        for i in range(count)
    ]
