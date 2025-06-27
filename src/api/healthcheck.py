from fastapi import APIRouter
from src.api.faiss_client import faiss_healthcheck

router = APIRouter()

@router.get("/health", tags=["Health"])
async def health():
    """API health endpoint."""
    faiss_ok = await faiss_healthcheck()

    if not faiss_ok:
        return {"status": "degraded", "faiss_service": "unreachable"}

    return {"status": "healthy", "faiss_service": "ok"}
