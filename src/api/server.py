import uvicorn
from .bootstrap import bootstrap

def launch():
    """Launch the FAISS RAG API server with bootstrap."""
    bootstrap()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    launch()
