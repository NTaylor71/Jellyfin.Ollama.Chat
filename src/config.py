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

# 🚨 Single Redis Queue for All Jobs
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "chat:queue")  # Unified queue name
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "chat:result:")
ERROR_PREFIX = os.getenv("ERROR_PREFIX", "chat:error:")

# Ollama Embedding Service
OLLAMA_EMBED_BASE_URL = "http://ollama_embed:11434" if IS_DOCKER else os.getenv("OLLAMA_EMBED_BASE_URL", "http://localhost:12435")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Ollama Chat Service
OLLAMA_CHAT_BASE_URL = "http://ollama_chat:11434" if IS_DOCKER else os.getenv("OLLAMA_CHAT_BASE_URL", "http://localhost:114434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

ENRICH_CHAT_BASE_URL = "http://ollama_chat:11434" if IS_DOCKER else os.getenv("ENRICH_CHAT_BASE_URL", "http://localhost:114434")
ENRICH_CHAT_MODEL = os.getenv("ENRICH_CHAT_MODEL", "phi3:3.8b")

KEYWORD_SIMILARITY_THRESHOLD=0.58

# FAISS Service
VECTORDB_URL = "http://faiss_service:6333" if IS_DOCKER else os.getenv("VECTORDB_URL", "http://localhost:6333")
FAISS_VECTOR_DIM = 4096  # Dimension for FAISS vectors

# API URL for CLI or external services
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ⏲️ Timeouts
TIMEOUT_SEC = int(os.getenv("TIMEOUT_SEC", 60))  # General timeout

# Redis Client
r = redis.from_url(REDIS_URL)
