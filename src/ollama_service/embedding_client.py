import httpx
from src.config import OLLAMA_EMBED_BASE_URL

async def get_ollama_embedding(text: str, model: str) -> list:
    """Query Ollama Embed service to generate embedding for the provided text using the specified model."""
    url = f"{OLLAMA_EMBED_BASE_URL}/api/embeddings"
    payload = {
        "model": model,  # Passed in by the caller
        "prompt": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    embedding = response.json().get("embedding", [])
    if not embedding:
        raise ValueError("No embedding returned from Ollama Embed service.")
    return embedding
