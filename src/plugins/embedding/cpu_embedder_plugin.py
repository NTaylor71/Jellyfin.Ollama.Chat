from typing import List
from ollama import AsyncClient
from src.config import OLLAMA_EMBED_MODEL

class CPUEmbedderPlugin:
    """Uses the embedding model (CPU) to vectorize enriched documents."""

    def __init__(self, model: str = OLLAMA_EMBED_MODEL, host: str = "http://localhost:12435"):
        self.client = AsyncClient(host=host)
        self.model = model

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embed(model=self.model, input=texts)
        return response.get("embeddings", [])
