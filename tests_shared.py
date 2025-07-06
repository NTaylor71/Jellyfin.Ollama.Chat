from src.shared.config import get_settings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def settings_to_console():
    """Display current environment settings to console for debugging."""
    logger.info("🧠 .env Settings :")
    
    settings = get_settings()
    
    # Core environment
    logger.info(f"✅ ENV: {settings.ENV}")
    logger.info(f"🔗 Redis: {settings.redis_url}")
    logger.info(f"🤖 Ollama Chat: {settings.OLLAMA_CHAT_BASE_URL}")
    logger.info(f"🔤 Ollama Embed: {settings.OLLAMA_EMBED_BASE_URL}")
    logger.info(f"🌐 API: {settings.API_URL}")
    logger.info(f"📊 VectorDB: {settings.VECTORDB_URL}")
    logger.info(f"🗄️ MongoDB: {settings.mongodb_url}")
    
    # Development features
    logger.info(f"🔓 CORS Enabled: {settings.ENABLE_CORS}")
    logger.info(f"📚 API Docs: {settings.ENABLE_API_DOCS}")
    logger.info(f"📝 Log Level: {settings.LOG_LEVEL}")
    logger.info(f"🔥 Hot Reload: {settings.PLUGIN_HOT_RELOAD}")
    
    # Java configuration
    logger.info(f"☕ Java Home: {settings.JAVA_HOME}")
    
    # Plugin configuration
    logger.info(f"🔌 Plugin Dir: {settings.PLUGIN_DIRECTORY}")
    
    logger.info("<<<---- Settings loaded from .env")
