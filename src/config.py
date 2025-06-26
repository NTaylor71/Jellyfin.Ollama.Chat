import os
import redis.asyncio as redis
from dotenv import load_dotenv

if os.getenv("ENV") != "docker":
    load_dotenv()

# Environment toggle
IS_DOCKER = os.getenv("ENV") == "docker"

# Redis
REDIS_HOST = "redis" if IS_DOCKER else os.getenv("REDIS_HOST", "localhost")
REDIS_URL = f"redis://{REDIS_HOST}:6379"
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "chat:queue")

# Ollama
OLLAMA_BASE_URL = "http://jellychat_ollama:11434" if IS_DOCKER else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Qdrant
VECTORDB_URL = "http://vectordb:6333" if IS_DOCKER else os.getenv("VECTORDB_URL", "http://localhost:6333")
VECTORDB_COLLECTION = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")

API_URL = os.getenv("API_URL", "http://localhost:8000")

r = redis.from_url(REDIS_URL)
