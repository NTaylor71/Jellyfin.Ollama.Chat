# /src/api/load_query_embellishers.py

from src.api.plugin_loader import load_plugins
from src.api.plugin_registry import plugin_registry

def load_query_embellishers():
    """
    Load all plugins and return only query embellishers.
    A query embellisher is a plugin that declares its type as 'query_embellisher'.
    """
    load_plugins()

    # Validate and return only the functions from the query embellisher plugins
    embellishers = []
    for plugin in plugin_registry:
        if plugin.get("type") == "query_embellisher":
            func = plugin.get("function")
            if callable(func):
                embellishers.append(func)
            else:
                print(f"⚠️ Plugin registered as 'query_embellisher' has no valid function: {plugin}")

    print(f"✅ Loaded {len(embellishers)} query embellisher(s).")
    return embellishers
