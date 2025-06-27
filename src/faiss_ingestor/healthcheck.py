from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    """FAISS Ingestor health endpoint."""
    return {"status": "healthy"}
