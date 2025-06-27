from src.api.plugin_registry import register_plugin, QUERY_EMBELLISHER

def embellish_query(query: str) -> str:
    """Example: Automatically add 'sci-fi' genre if user says 'space'."""
    if "space" in query.lower() and "genre:sci-fi" not in query.lower():
        return f"{query} genre:sci-fi"
    return query

# Register the plugin with weight 100 (can adjust later)
register_plugin(QUERY_EMBELLISHER, embellish_query, weight=100)
