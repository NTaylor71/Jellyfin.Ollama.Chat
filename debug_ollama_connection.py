#!/usr/bin/env python3
"""
Debug Ollama Connection
Quick test to verify the Ollama connection works.
"""

import asyncio
import httpx
import logging
from src.shared.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ollama_connection():
    """Test connection to Ollama service."""
    settings = get_settings()
    
    logger.info(f"Testing Ollama connection to: {settings.OLLAMA_CHAT_BASE_URL}")
    logger.info(f"Expected model: {settings.OLLAMA_CHAT_MODEL}")
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            
            # Test basic connection
            logger.info("1. Testing basic connection...")
            response = await client.get(f"{settings.OLLAMA_CHAT_BASE_URL}/api/tags")
            
            if response.status_code == 200:
                logger.info("✅ Basic connection successful")
                data = response.json()
                models = data.get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                logger.info(f"Available models: {model_names}")
                
                if settings.OLLAMA_CHAT_MODEL in model_names:
                    logger.info(f"✅ Model '{settings.OLLAMA_CHAT_MODEL}' is available")
                else:
                    logger.warning(f"⚠️ Model '{settings.OLLAMA_CHAT_MODEL}' not found")
                
                # Test simple generation
                logger.info("2. Testing simple generation...")
                test_payload = {
                    "model": settings.OLLAMA_CHAT_MODEL,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                
                response = await client.post(f"{settings.OLLAMA_CHAT_BASE_URL}/api/generate", json=test_payload)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Generation successful: {result.get('response', '')}")
                    return True
                else:
                    logger.error(f"❌ Generation failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return False
                    
            else:
                logger.error(f"❌ Connection failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
    except httpx.ConnectError as e:
        logger.error(f"❌ Connection error: {e}")
        logger.error("Please check:")
        logger.error(f"  - Is Ollama running at {settings.OLLAMA_CHAT_BASE_URL}?")
        logger.error("  - Is the URL correct?")
        logger.error("  - Are there any firewall issues?")
        return False
    except httpx.TimeoutError:
        logger.error("❌ Connection timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama_connection())
    if success:
        logger.info("🎉 Ollama connection test passed!")
    else:
        logger.error("💥 Ollama connection test failed!")