from src.api.plugin_registry import register_plugin, FAISS_CRUD_PLUGIN

def on_faiss_add(vectors: list[dict]) -> list[dict]:
    """Example: Log or enrich metadata on FAISS add events."""
    print(f"📦 FAISS CRUD Plugin: Adding {len(vectors)} vectors")
    return vectors

# Register the plugin with weight 100
register_plugin(FAISS_CRUD_PLUGIN, on_faiss_add, weight=100)
