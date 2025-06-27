import asyncio
import json
import uuid
from src.config import r
from src.config import GPU_QUEUE, GPU_RESULT_PREFIX, GPU_ERROR_PREFIX, TIMEOUT_SEC

async def submit_gpu_job(prompt: str) -> str:
    """Submit a GPU embedding job to Redis."""
    job_id = str(uuid.uuid4())
    gpu_job = {
        "job_id": job_id,
        "prompt": prompt
    }
    await r.rpush(GPU_QUEUE, json.dumps(gpu_job))
    print(f"📤 Submitted GPU job: {job_id}")
    return job_id

async def wait_for_gpu_result(job_id: str, timeout: int = TIMEOUT_SEC) -> list:
    """Wait for GPU worker to return embedding result or error."""
    print(f"⏳ Waiting for GPU job result: {job_id} (up to {timeout}s)")
    for _ in range(timeout):
        result = await r.get(f"{GPU_RESULT_PREFIX}{job_id}")
        error = await r.get(f"{GPU_ERROR_PREFIX}{job_id}")

        if result:
            print(f"✅ Result for GPU job {job_id} received.")
            return json.loads(result.decode())

        if error:
            print(f"❌ GPU job {job_id} failed: {error.decode()}")
            return []

        await asyncio.sleep(1)

    print(f"❌ Timeout waiting for GPU job {job_id}.")
    return []

async def main():
    prompt = "List famous sci-fi movies about artificial intelligence."
    job_id = await submit_gpu_job(prompt)
    embedding = await wait_for_gpu_result(job_id)

    if embedding:
        print(f"✅ Embedding received. Length: {len(embedding)}")
    else:
        print("❌ Embedding not received or job failed.")

if __name__ == "__main__":
    asyncio.run(main())
