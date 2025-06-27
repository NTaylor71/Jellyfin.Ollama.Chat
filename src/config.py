# /src/config.py

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
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "chat:result:")
ERROR_PREFIX = os.getenv("ERROR_PREFIX", "chat:error:")

# Ollama
OLLAMA_BASE_URL = "http://ollama:11434" if IS_DOCKER else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# FAISS Service
VECTORDB_URL = "http://faiss_service:6333" if IS_DOCKER else os.getenv("VECTORDB_URL", "http://localhost:6333")

# API URL for CLI or external services
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Redis Client
r = redis.from_url(REDIS_URL)
