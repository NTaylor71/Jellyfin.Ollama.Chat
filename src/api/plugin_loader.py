import os
import importlib
import sys
from pathlib import Path

from .plugin_registry import plugin_registry

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"

def load_plugins():
    """Load all plugins from the /plugins/ directory."""
    print("🔍 Loading plugins...")

    plugin_registry.clear()

    sys.path.insert(0, str(PLUGINS_DIR.parent))

    for plugin_file in PLUGINS_DIR.glob("*.py"):
        if plugin_file.name.startswith("_"):
            continue  # Skip private files

        module_name = f"plugins.{plugin_file.stem}"
        try:
            if module_name in sys.modules:
                print(f"♻️ Reloading {module_name}")
                importlib.reload(sys.modules[module_name])
            else:
                print(f"📦 Importing {module_name}")
                importlib.import_module(module_name)
        except Exception as e:
            print(f"❌ Failed to load plugin {module_name}: {e}")

    print(f"✅ {len(plugin_registry)} plugin(s) loaded.")

def discover_plugin_files():
    """Utility to list all plugin files (used for hot reload system)."""
    return [f.name for f in PLUGINS_DIR.glob("*.py") if not f.name.startswith("_")]
