from src.api.plugin_registry import register_plugin, EMBED_DATA_EMBELLISHER

def embellish_embedding_document(document: dict) -> dict:
    """Example: Automatically tag English-language content if missing."""
    if not document.get("language"):
        document["language"] = "English"
    return document

# Register the plugin with weight 100
register_plugin(EMBED_DATA_EMBELLISHER, embellish_embedding_document, weight=100)
