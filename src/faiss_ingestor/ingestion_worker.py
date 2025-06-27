import asyncio
import json
from uuid import uuid4
import httpx
from src.config import r, VECTORDB_URL, REDIS_QUEUE, RESULT_PREFIX, ERROR_PREFIX
from src.data.sample_entries import get_sample_entries

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
    """Send the generated vectors to FAISS."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{VECTORDB_URL}/add", json={"vectors": vectors})
        response.raise_for_status()


async def generate_movie_vectors():
    """Generate embedding jobs for movies and collect their embedding results from the Redis queue."""
    sample_entries = get_sample_entries()
    movie_data = []

    for entry in sample_entries:
        formatted_text = (
            f"{entry['title']} ({entry['year']}) - {entry['genres']} - {entry['overview']}"
        )

        job_id = str(uuid4())
        gpu_job = {
            "job_id": job_id,
            "task_type": "embedding",
            "prompt": formatted_text
        }

        # Push embedding job to Redis queue
        await r.rpush(REDIS_QUEUE, json.dumps(gpu_job))
        print(f"📤 Submitted embedding job {job_id} to Redis queue.")

        # Wait for the embedding result
        for _ in range(60):  # Up to 60 seconds wait
            embedding_result = await r.get(f"{RESULT_PREFIX}{job_id}")
            error_result = await r.get(f"{ERROR_PREFIX}{job_id}")

            if embedding_result:
                vector = json.loads(embedding_result)
                movie_data.append({
                    "id": str(uuid4()),
                    "vector": vector,
                    "metadata": entry
                })
                print(f"✅ Received embedding result for job {job_id}")
                break

            if error_result:
                error_message = error_result.decode()
                print(f"❌ Job {job_id} failed: {error_message}")
                break

            await asyncio.sleep(1)

        else:
            print(f"❌ Job {job_id} timed out.")

    return movie_data


async def start_real_ingestion():
    """Start ingestion with real movie vectors using Redis queue."""
    vectors = await generate_movie_vectors()
    await send_to_faiss(vectors)
    print(f"✅ Ingested {len(vectors)} real movie vectors.")
