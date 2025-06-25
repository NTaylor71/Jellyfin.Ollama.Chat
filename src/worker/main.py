import os
import json
import asyncio
import redis.asyncio as redis
from langchain.chains import RetrievalQA
from langchain_qdrant import Qdrant
from langchain_ollama import Ollama

# Init Redis
r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Init LangChain RAG chain
llm = Ollama(
    model=os.getenv("OLLAMA_MODEL", "mistral"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)

vectorstore = Qdrant(
    url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
    collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag"),
    prefer_grpc=False
)

retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

async def process_job(job_json: str):
    try:
        job = json.loads(job_json)
        job_id = job["job_id"]
        query = job["query"]
        print(f"🔧 Processing job: {job_id} | {query}")

        result = qa.run(query)
        await r.set(f"chat:result:{job_id}", result, ex=3600)  # expires in 1 hour

    except Exception as e:
        print(f"❌ Error: {e}")
        await r.set(f"chat:error:{job.get('job_id', 'unknown')}", str(e), ex=3600)

async def worker_loop():
    print("🚀 Worker ready. Listening on chat:queue")
    while True:
        job = await r.blpop("chat:queue", timeout=5)
        if job:
            _, job_data = job
            await process_job(job_data.decode("utf-8"))
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(worker_loop())
