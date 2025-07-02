"""Basic configuration tests."""

import pytest
from src.shared.config import get_settings


def test_settings_load():
    """Test that settings can be loaded."""
    settings = get_settings()
    assert settings is not None
    assert settings.ENV in ["localhost", "docker", "production"]


def test_directories_created():
    """Test that required directories exist."""
    settings = get_settings()
    from pathlib import Path
    assert Path("./data").exists()
    assert Path("./logs").exists()
    assert Path(settings.PLUGIN_DIRECTORY).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
