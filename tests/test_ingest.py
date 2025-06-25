import os
from pprint import pprint
from src.rag.formatter import render_media_text_block
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from uuid import uuid4

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

# Step 1: Render via Jinja2
text_block = render_media_text_block(sample_entry)
print("📝 Rendered text:\n")
print(text_block)
print("\n" + "="*60)

# Step 2: Embed via Ollama
embedder = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL", "mistral"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)
vector = embedder.embed_documents([text_block])
print(f"\n🔢 Vector length: {len(vector[0])}")

# Step 3: Store in Qdrant
doc_id = str(uuid4())
qdrant = Qdrant(
    url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
    collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag"),
    prefer_grpc=False
)
qdrant.add_texts(texts=[text_block], metadatas=None, ids=[doc_id], embeddings=vector)
print(f"\n✅ Stored in Qdrant with ID: {doc_id}")
