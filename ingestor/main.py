from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os

docs = [
    Document(page_content="The Matrix (1999)", metadata={"type": "movie", "year": "1999"}),
    Document(page_content="Pulp Fiction (1994)", metadata={"type": "movie", "year": "1994"}),
]

embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_MODEL", "mistral"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)

vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
    collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag"),
    prefer_grpc=False
)

print("✅ Ingested and stored embeddings.")
