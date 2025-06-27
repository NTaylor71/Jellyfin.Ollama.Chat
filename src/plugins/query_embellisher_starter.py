# /src/plugins/query_embellisher_starter.py

from src.api.plugin_registry import register_plugin

def embellish_query(query: str) -> str:
    """Example: Automatically add 'sci-fi' genre if user says 'space'."""
    if "space" in query.lower() and "genre:sci-fi" not in query.lower():
        return f"{query} genre:sci-fi"
    return query

# Register the plugin
register_plugin({
    "type": "query_embellisher",
    "function": embellish_query,
    "weight": 100
})
