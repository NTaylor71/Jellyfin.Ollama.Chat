import httpx
from src.config import OLLAMA_CHAT_BASE_URL

async def get_ollama_chat_response(messages: list, model: str) -> str:
    """
    Query Ollama Chat API using the specified model.

    Args:
        messages (list): List of message dicts, e.g. [{"role": "user", "content": "Your question here."}]
        model (str): Model name to use for chat (e.g., "llama3.2:3b")

    Returns:
        str: Chat response content.
    """
    url = f"{OLLAMA_CHAT_BASE_URL}/api/chat"
    payload = {"model": model, "messages": messages}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    return response.json().get("message", {}).get("content", "")
