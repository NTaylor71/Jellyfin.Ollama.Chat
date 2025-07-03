"""
Chat endpoints for RAG system.
"""

import logging
import time
import uuid
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator

from src.shared.config import get_settings
from src.api.plugin_registry import get_plugin_registry
from src.plugins.base import PluginType, PluginExecutionContext

logger = logging.getLogger(__name__)


router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model with security validation."""
    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results")
    include_metadata: Optional[bool] = Field(default=True, description="Include result metadata")
    
    @validator("query")
    def validate_query_security(cls, v):
        """Security validation for query content."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
            
        # Basic content filtering for security
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.',
            r'sys\.',
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Query contains potentially unsafe content")
        
        return v.strip()
    
    @validator("context")
    def validate_context_security(cls, v):
        """Security validation for context data."""
        if v is None:
            return v
            
        # Limit context size to prevent memory exhaustion
        import json
        context_str = json.dumps(v)
        if len(context_str) > 10000:  # 10KB limit
            raise ValueError("Context data too large")
            
        return v


class ChatResponse(BaseModel):
    """Chat response model."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    query: str = Field(..., description="Original query")
    timestamp: float = Field(..., description="Request timestamp")
    estimated_time: Optional[float] = Field(default=None, description="Estimated completion time")


class ChatResult(BaseModel):
    """Chat result model."""
    job_id: str
    status: str  # completed, failed, processing
    query: str
    results: Optional[List[Dict[str, Any]]] = None
    response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: float


# In-memory storage for demo purposes (will be replaced with Redis)
_job_storage: Dict[str, ChatResult] = {}


@router.post("/", response_model=ChatResponse)
async def submit_chat_query(request: ChatRequest):
    """
    Submit a chat query for processing.
    
    This endpoint accepts a query and returns a job ID for tracking.
    The actual processing happens asynchronously.
    """
    settings = get_settings()
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Execute query embellisher plugins
    enhanced_query = await _execute_query_embellisher_plugins(request.query, request.context)
    
    # Create initial job record with enhanced query
    job_result = ChatResult(
        job_id=job_id,
        status="processing",
        query=enhanced_query,
        timestamp=time.time()
    )
    
    # Store job (temporary in-memory storage)
    _job_storage[job_id] = job_result
    
    # Simulate processing (replace with actual queue later)
    await _simulate_processing(job_id, request, enhanced_query)
    
    return ChatResponse(
        job_id=job_id,
        status="accepted",
        query=enhanced_query,
        timestamp=time.time(),
        estimated_time=5.0  # 5 seconds estimated
    )


@router.get("/result/{job_id}", response_model=ChatResult)
async def get_chat_result(job_id: str):
    """
    Get the result of a chat query by job ID.
    
    Returns the current status and results if processing is complete.
    """
    if job_id not in _job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return _job_storage[job_id]


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get just the status of a job."""
    if job_id not in _job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _job_storage[job_id]
    return {
        "job_id": job_id,
        "status": job.status,
        "timestamp": job.timestamp
    }


@router.get("/jobs")
async def list_recent_jobs(limit: int = 10):
    """List recent jobs (for debugging)."""
    jobs = list(_job_storage.values())
    # Sort by timestamp, most recent first
    jobs.sort(key=lambda x: x.timestamp, reverse=True)
    
    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "query": job.query[:100] + "..." if len(job.query) > 100 else job.query,
                "timestamp": job.timestamp
            }
            for job in jobs[:limit]
        ],
        "total": len(_job_storage)
    }


async def _execute_query_embellisher_plugins(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute query embellisher plugins to enhance the user's query.
    
    Returns the enhanced query or original query if no plugins are available.
    """
    try:
        plugin_registry = await get_plugin_registry()
        
        # Create execution context
        execution_context = PluginExecutionContext(
            request_id=str(uuid.uuid4()),
            metadata={
                "user_context": context or {},
                "timestamp": time.time(),
                "query_type": "chat"
            }
        )
        
        # Execute query embellisher plugins
        results = await plugin_registry.execute_plugins(
            PluginType.QUERY_EMBELLISHER,
            query,
            execution_context
        )
        
        # Use the enhanced query from the last successful plugin
        enhanced_query = query
        for result in results:
            if result.success and result.data:
                enhanced_query = result.data
        
        return enhanced_query
        
    except Exception as e:
        logger.error(f"Error executing query embellisher plugins: {e}")
        return query


async def _simulate_processing(job_id: str, request: ChatRequest, enhanced_query: str):
    """
    Simulate chat processing (temporary implementation).
    
    This will be replaced with actual Redis queue processing.
    """
    import asyncio
    
    # Simulate some processing time
    await asyncio.sleep(0.1)
    
    # Create mock results
    mock_results = [
        {
            "title": "Sample Movie 1",
            "summary": f"This is a sample movie result for query: {enhanced_query}",
            "relevance_score": 0.95,
            "metadata": {
                "genre": "Drama",
                "year": 2023,
                "duration": "120 min"
            }
        },
        {
            "title": "Sample Movie 2", 
            "summary": f"Another relevant movie for: {enhanced_query}",
            "relevance_score": 0.87,
            "metadata": {
                "genre": "Action",
                "year": 2022,
                "duration": "95 min"
            }
        }
    ]
    
    # Update job with results
    job = _job_storage[job_id]
    job.status = "completed"
    job.results = mock_results[:request.max_results] if request.max_results else mock_results
    job.response = f"Found {len(job.results)} relevant movies for your query: '{enhanced_query}'"
    job.processing_time = 0.1
    job.metadata = {
        "total_results": len(mock_results),
        "processing_method": "mock",
        "filters_applied": [],
        "query_enhanced": enhanced_query != request.query,
        "original_query": request.query
    }


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job (if still processing)."""
    if job_id not in _job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _job_storage[job_id]
    if job.status == "processing":
        job.status = "cancelled"
        return {"message": "Job cancelled successfully"}
    else:
        return {"message": f"Job is {job.status}, cannot cancel"}


@router.post("/test")
async def test_chat_endpoint(query: str = "test movies"):
    """
    Simple test endpoint for quick testing.
    
    This bypasses the job system and returns results immediately.
    """
    return {
        "query": query,
        "results": [
            {
                "title": "Test Movie",
                "summary": f"This is a test result for: {query}",
                "score": 1.0
            }
        ],
        "status": "completed",
        "timestamp": time.time()
    }
    