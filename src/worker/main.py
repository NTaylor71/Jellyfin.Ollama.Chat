import os
import asyncio
import redis.asyncio as redis
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA

async def wait_for_collection(client, collection_name):
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        print(f"❌ Collection '{collection_name}' not found. Worker will exit.")
        exit(1)

async def worker_loop():
    qdrant_url = os.getenv("VECTORDB_URL", "http://vectordb:6333")
    collection_name = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")

    qdrant_client = QdrantClient(url=qdrant_url)
    await wait_for_collection(qdrant_client, collection_name)

    embedder = OllamaEmbeddings(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://jellychat_ollama:11434")
    )

    vectorstore = QdrantVectorStore.from_texts(
        texts=[],
        embedding=embedder,
        url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
        collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")
    )

    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff"
    )

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    print("🚀 Worker is ready. Listening for jobs...")

    while True:
        job = await r.blpop("chat:queue", timeout=5)
        if job:
            _, job_data = job
            print(f"🔧 Processing job: {job_data.decode()}")
            # Process job here
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(worker_loop())
