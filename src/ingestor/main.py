import os
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

# --- Robust Collection Management ---

async def ensure_collection(client: QdrantClient, collection_name: str, embedding_size: int, replace: bool):
    collections = [c.name for c in client.get_collections().collections]

    if replace and collection_name in collections:
        print(f"♻️ Deleting existing collection '{collection_name}'")
        client.delete_collection(collection_name)

    if collection_name not in collections:
        print(f"🆕 Creating collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
        )
    else:
        print(f"✅ Collection '{collection_name}' already exists")

# --- Ingest Script ---

async def main():
    docs = [
        Document(page_content="The Matrix (1999)", metadata={"type": "movie", "year": "1999"}),
        Document(page_content="Pulp Fiction (1994)", metadata={"type": "movie", "year": "1994"}),
    ]

    qdrant_url = os.getenv("VECTORDB_URL", "http://vectordb:6333")
    collection_name = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")
    embedder = OllamaEmbeddings(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://jellychat_ollama:11434")
    )

    qdrant_client = QdrantClient(url=qdrant_url)

    await ensure_collection(qdrant_client, collection_name, embedding_size=4096, replace=True)

    vectorstore = QdrantVectorStore.from_texts(
        texts=[],
        embedding=embedder,
        url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
        collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")
    )

    vectorstore.add_documents(docs)

    print(f"✅ Ingested {len(docs)} documents into '{collection_name}'")

if __name__ == "__main__":
    asyncio.run(main())
