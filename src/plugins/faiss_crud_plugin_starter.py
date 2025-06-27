# /src/plugins/faiss_crud_plugin_starter.py

from src.api.plugin_registry import register_plugin

def on_faiss_add(vectors: list[dict]) -> list[dict]:
    """Example: Log or enrich metadata on FAISS add events."""
    print(f"📦 FAISS CRUD Plugin: Adding {len(vectors)} vectors")
    return vectors

# Register the plugin
register_plugin({
    "type": "faiss_crud_plugin",
    "function": on_faiss_add,
    "weight": 100
})
