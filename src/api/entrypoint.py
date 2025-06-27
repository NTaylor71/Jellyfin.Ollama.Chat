import uvicorn
from .bootstrap import bootstrap

def start_api():
    # Run bootstrap tasks (e.g., plugin load, signal hooks)
    bootstrap()

    # Start the FastAPI server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    start_api()
