# /src/plugins/sample_query_embellisher.py

from src.api.plugin_registry import register_plugin

def add_surfer_dude_flair(query: str) -> str:
    """Example embellisher that adds some Surfer Dude flair to the query."""
    return f"{query} — totally rad, dude!"

# Register the plugin
register_plugin({
    "type": "query_embellisher",
    "function": add_surfer_dude_flair
})
