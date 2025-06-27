from collections import defaultdict

# Global plugin registry
plugin_registry = defaultdict(list)

# Plugin Types
QUERY_EMBELLISHER = "query_embellisher"
EMBED_DATA_EMBELLISHER = "embed_data_embellisher"
FAISS_CRUD_PLUGIN = "faiss_crud_plugin"

def register_plugin(plugin_type: str, plugin_func, weight: int = 100):
    """Register a plugin with its type and weight."""
    plugin_registry[plugin_type].append((weight, plugin_func))
    plugin_registry[plugin_type].sort(key=lambda x: x[0])  # Sort by weight

def get_plugins(plugin_type: str):
    """Retrieve all plugins for a specific type, ordered by weight."""
    return [func for _, func in plugin_registry.get(plugin_type, [])]

def clear_registry():
    """Clear all loaded plugins."""
    plugin_registry.clear()
