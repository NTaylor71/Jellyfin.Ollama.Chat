from src.shared.config import get_settings
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def settings_to_console():
    """
    
    """
    logger.info("🧠 .env Settings :")

    settings = get_settings()

    logger.info(f"✅ ENV: {settings.ENV}")
