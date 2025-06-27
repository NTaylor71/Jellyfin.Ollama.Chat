from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    """FAISS Service health endpoint."""
    return {"status": "healthy"}
