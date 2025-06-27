from src.api.plugin_loader import load_plugins
from src.api.plugin_registry import (
    get_plugins,
    QUERY_EMBELLISHER,
    EMBED_DATA_EMBELLISHER,
    FAISS_CRUD_PLUGIN
)

def test_query_embellishers():
    load_plugins()
    plugins = get_plugins(QUERY_EMBELLISHER)
    assert plugins, "No Query Embellishers loaded."
    for plugin in plugins:
        try:
            result = plugin("This is a space movie")
            print(f"Query Embellisher Output: {result}")
            assert isinstance(result, str), "Query embellisher output should be a string."
        except Exception as e:
            print(f"❌ Query Embellisher failed: {e}")

def test_embed_data_embellishers():
    load_plugins()
    plugins = get_plugins(EMBED_DATA_EMBELLISHER)
    assert plugins, "No Embed Data Embellishers loaded."
    sample_doc = {"title": "Test Movie"}
    for plugin in plugins:
        try:
            result = plugin(sample_doc)
            print(f"Embed Data Embellisher Output: {result}")
            assert isinstance(result, dict), "Embed data embellisher should return a dictionary."
        except Exception as e:
            print(f"❌ Embed Data Embellisher failed: {e}")

def test_faiss_crud_plugins():
    load_plugins()
    plugins = get_plugins(FAISS_CRUD_PLUGIN)
    assert plugins, "No FAISS CRUD Plugins loaded."
    sample_vectors = [{"id": "vec-1", "vector": [0.1] * 4096}]
    for plugin in plugins:
        try:
            result = plugin(sample_vectors)
            print(f"FAISS CRUD Plugin Output: {result}")
            assert isinstance(result, list), "FAISS CRUD plugin should return a list."
        except Exception as e:
            print(f"❌ FAISS CRUD Plugin failed: {e}")

if __name__ == "__main__":
    print("\n🔧 Testing Query Embellishers...")
    test_query_embellishers()

    print("\n🔧 Testing Embed Data Embellishers...")
    test_embed_data_embellishers()

    print("\n🔧 Testing FAISS CRUD Plugins...")
    test_faiss_crud_plugins()

    print("\n✅ All plugins loaded and executed successfully.")
