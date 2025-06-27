import os
from pprint import pprint
from src.rag.formatter import render_media_text_block

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from uuid import uuid4

# 1. Render via Jinja2
sample_entry = {
    "title": "The Matrix",
    "year": 1999,
    "genres": ["Science Fiction", "Action"],
    "overview": "A hacker discovers the world is a simulation.",
    "actors": ["Keanu Reeves", "Laurence Fishburne"],
    "tagline": "Welcome to the Real World",
    "certificate": "R",
    "media_type": "Movie",
    "language": "English"
}
text_block = render_media_text_block(sample_entry)
print("📝 Rendered text:\n", text_block)

# 2. Embed via Ollama
embedder = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL", "mistral"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)
vector = embedder.embed_documents([text_block])
print(f"\n🔢 Vector length: {len(vector[0])}")

# 3. Store in Qdrant
doc_id = str(uuid4())
qdrant_client = QdrantClient(url=os.getenv("VECTORDB_URL", "http://vectordb:6333"))
collection_name = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")

# Use `from_documents` helper to create/reuse a collection, auto-creating as needed
qdrant_store = QdrantVectorStore.from_documents(
    texts=[text_block],
    embedding=embedder,
    url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
    collection_name=collection_name
)
print(f"\n✅ Stored in Qdrant in collection '{collection_name}'")
