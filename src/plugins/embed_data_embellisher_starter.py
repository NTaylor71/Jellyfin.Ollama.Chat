# /src/plugins/embed_data_embellisher_starter.py

from src.api.plugin_registry import register_plugin

def embellish_embedding_document(document: dict) -> dict:
    """Example: Automatically tag English-language content if missing."""
    if not document.get("language"):
        document["language"] = "English"
    return document

# Register the plugin
register_plugin({
    "type": "embed_data_embellisher",
    "function": embellish_embedding_document,
    "weight": 100
})
