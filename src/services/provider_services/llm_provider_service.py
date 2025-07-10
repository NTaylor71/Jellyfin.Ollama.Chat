"""
LLM Provider Service - FastAPI service hosting Ollama/LLM operations.

Dedicated service for LLM operations, separate from NLP providers for
better resource isolation and scaling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.shared.config import get_settings
from src.providers.llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """Request to LLM provider."""
    concept: str = Field(..., min_length=1, max_length=500)
    media_context: str = Field(default="movie")
    max_concepts: int = Field(default=10, ge=1, le=50)
    field_name: str = Field(default="concept")
    options: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from LLM provider."""
    success: bool
    execution_time_ms: float
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMServiceHealth(BaseModel):
    """LLM service health status."""
    status: str
    uptime_seconds: float
    llm_provider: Dict[str, Any]
    total_requests: int
    failed_requests: int
    models_available: List[str]


class LLMProviderManager:
    """Manages LLM provider instance and its lifecycle."""
    
    def __init__(self):
        self.provider: Optional[LLMProvider] = None
        self.initialization_error: Optional[str] = None
        self.request_count = 0
        self.error_count = 0
        self.start_time = asyncio.get_event_loop().time()
        self.initialization_state = "starting"
        self.initialization_progress = {}
    
    async def initialize_provider(self):
        """Initialize LLM provider."""
        logger.info("Initializing LLM provider...")
        self.initialization_state = "connecting_ollama"
        self.initialization_progress = {"phase": "connecting_ollama", "current_task": "importing LLM provider"}
        
        try:
            from src.providers.llm.llm_provider import LLMProvider
            self.initialization_progress["current_task"] = "creating LLM provider instance"
            
            self.provider = LLMProvider()
            self.initialization_progress["current_task"] = "connecting to Ollama service"
            
            if await self.provider.initialize():
                self.initialization_state = "ready"
                self.initialization_progress = {
                    "phase": "completed",
                    "status": "ready"
                }
                logger.info("✅ LLM provider initialized")
            else:
                self.initialization_error = "LLM provider waiting for models"
                self.initialization_progress["error"] = "Models not ready yet - waiting for model-manager"
                logger.info("⏳ LLM provider waiting for models to download (this is normal during startup)")
        except ImportError as e:
            self.initialization_error = f"Import error: {e}"
            self.initialization_progress["error"] = f"Import error: {e}"
            logger.error(f"❌ LLM provider import failed: {e}")
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_progress["error"] = str(e)
            logger.error(f"❌ LLM provider error: {e}")
        
        logger.info("LLM Provider initialization complete.")
    
    async def cleanup_provider(self):
        """Cleanup LLM provider."""
        logger.info("Cleaning up LLM provider...")
        if self.provider:
            try:
                if hasattr(self.provider, 'cleanup'):
                    await self.provider.cleanup()
                logger.info("✅ LLM provider cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up LLM provider: {e}")
    
    def get_provider(self) -> LLMProvider:
        """Get the LLM provider."""
        if not self.provider:
            raise HTTPException(
                status_code=503, 
                detail=f"LLM provider not available. Error: {self.initialization_error}"
            )
        return self.provider
    
    def get_health_status(self) -> LLMServiceHealth:
        """Get service health status."""
        current_time = asyncio.get_event_loop().time()
        uptime = current_time - self.start_time
        
        provider_status = {}
        models_available = []
        
        if self.provider:
            try:
                metadata = self.provider.metadata
                provider_status = {
                    "status": "healthy",
                    "type": metadata.provider_type,
                    "version": metadata.version,
                    "initialized": getattr(self.provider, '_initialized', False)
                }
                
                
                if hasattr(self.provider, 'get_available_models'):
                    models_available = self.provider.get_available_models()
                
            except Exception as e:
                provider_status = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            provider_status = {
                "status": "failed_to_initialize",
                "error": self.initialization_error or "Unknown error"
            }
        

        service_status = "starting"
        if self.initialization_state == "ready":
            service_status = "healthy" if self.provider else "degraded"
        elif self.initialization_state == "connecting_ollama":
            service_status = "initializing"
        else:
            service_status = "starting"
        
        return LLMServiceHealth(
            status=service_status,
            uptime_seconds=uptime,
            llm_provider=provider_status,
            total_requests=self.request_count,
            failed_requests=self.error_count,
            models_available=models_available
        )



provider_manager = LLMProviderManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""

    await provider_manager.initialize_provider()
    yield

    await provider_manager.cleanup_provider()



settings = get_settings()
app = FastAPI(
    title="LLM Provider Service",
    description="FastAPI service hosting Ollama/LLM operations",
    version="1.0.0",
    lifespan=lifespan
)


if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=settings.CORS_METHODS,
        allow_headers=settings.CORS_HEADERS,
    )


@app.get("/health", response_model=LLMServiceHealth)
async def health_check():
    """Get service health status."""
    return provider_manager.get_health_status()


@app.get("/health/detailed")
async def detailed_health_check():
    """Get detailed health status with initialization progress."""
    health = provider_manager.get_health_status()
    
    return {
        "status": health.status,
        "initialization_state": provider_manager.initialization_state,
        "initialization_progress": provider_manager.initialization_progress,
        "uptime_seconds": health.uptime_seconds,
        "llm_provider": health.llm_provider,
        "total_requests": health.total_requests,
        "failed_requests": health.failed_requests,
        "models_available": health.models_available,
        "ready": provider_manager.initialization_state == "ready"
    }


@app.get("/health/ready")
async def readiness_check():
    """Simple readiness check for Docker health checks."""
    if provider_manager.initialization_state == "ready":
        return {"ready": True, "status": "healthy"}
    else:
        return {"ready": False, "status": provider_manager.initialization_state}, 503


@app.get("/providers")
async def get_provider_info():
    """Get LLM provider information."""
    provider = provider_manager.get_provider()
    metadata = provider.metadata
    
    return {
        "name": metadata.name,
        "type": metadata.provider_type,
        "version": metadata.version,
        "strengths": metadata.strengths,
        "best_for": metadata.best_for,
        "context_aware": metadata.context_aware,
        "initialization_error": provider_manager.initialization_error
    }


@app.post("/providers/llm/expand", response_model=LLMResponse)
async def expand_concept(
    request: LLMRequest,
    background_tasks: BackgroundTasks
):
    """Expand a concept using the LLM provider."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        

        from src.providers.nlp.base_provider import ExpansionRequest
        
        
        expansion_request = ExpansionRequest(
            concept=request.concept,
            media_context=request.media_context,
            max_concepts=request.max_concepts,
            field_name=request.field_name,
            options=request.options
        )
        

        result = await provider.expand_concept(expansion_request)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return LLMResponse(
            success=result.success,
            execution_time_ms=execution_time_ms,
            result=result.enhanced_data if result.success else None,
            error_message=result.error_message if not result.success else None,
            metadata={
                "confidence_score": result.confidence_score.overall if result.success else None,
                "cache_key": result.cache_key.generate_key() if result.success else None,
                "model_used": getattr(result.plugin_metadata, 'model_used', None)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Error expanding concept with LLM: {e}")
        
        return LLMResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            error_message=str(e)
        )


@app.post("/providers/llm/health")
async def check_provider_health():
    """Check health of LLM provider."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'health_check'):
            health_status = await provider.health_check()
        else:
            health_status = {
                "status": "healthy",
                "initialized": getattr(provider, '_initialized', False)
            }
        
        return {
            "provider": "llm",
            "health": health_status,
            "metadata": {
                "name": provider.metadata.name,
                "type": provider.metadata.provider_type,
                "version": provider.metadata.version
            }
        }
        
    except Exception as e:
        return {
            "provider": "llm",
            "health": {
                "status": "error",
                "error": str(e)
            }
        }


@app.get("/models")
async def list_available_models():
    """List available LLM models."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'get_available_models'):
            models = provider.get_available_models()
            return {
                "available_models": models,
                "current_model": getattr(provider, 'current_model', None)
            }
        else:
            return {
                "available_models": [],
                "current_model": None,
                "note": "Model listing not supported by this provider"
            }
            
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "available_models": [],
            "error": str(e)
        }


@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model (if supported by provider)."""
    try:
        provider = provider_manager.get_provider()
        
        if hasattr(provider, 'load_model'):
            success = await provider.load_model(model_name)
            return {
                "success": success,
                "model": model_name,
                "message": f"Model {model_name} {'loaded' if success else 'failed to load'}"
            }
        else:
            return {
                "success": False,
                "model": model_name,
                "error": "Dynamic model loading not supported by this provider"
            }
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return {
            "success": False,
            "model": model_name,
            "error": str(e)
        }






class KeywordExpansionRequest(BaseModel):
    """Request for LLM keyword expansion."""
    keywords: List[str] = Field(..., min_items=1, max_items=10)
    context: str = Field(default="")
    field_name: str = Field(default="keywords")
    max_concepts: int = Field(default=15, ge=1, le=50)
    expansion_style: str = Field(default="semantic_related")
    prompt_template: str = Field(default="")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    smart_retry_until: str = Field(default="list")


class KeywordExpansionResponse(BaseModel):
    """Response from LLM keyword expansion."""
    concepts: List[str]
    metadata: Dict[str, Any]


@app.post("/providers/llm/keywords/expand", response_model=KeywordExpansionResponse)
async def expand_keywords(request: KeywordExpansionRequest):
    """Expand keywords using LLM semantic understanding."""
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        
        start_time = asyncio.get_event_loop().time()
        
        
        if request.prompt_template:
            
            keywords_text = ", ".join(request.keywords)
            if request.context:
                prompt = request.prompt_template.format(value=f"{keywords_text} (Context: {request.context})")
            else:
                prompt = request.prompt_template.format(value=keywords_text)
        else:

            keywords_text = ", ".join(request.keywords)
            if request.expansion_style == "semantic_related":
                prompt = f"Given these keywords: {keywords_text}, provide {request.max_concepts} semantically related concepts, themes, and terms that would help categorize and find similar content. Return as a simple list."
            elif request.expansion_style == "genre_expansion":
                prompt = f"Given these genres/categories: {keywords_text}, provide {request.max_concepts} related genres, subgenres, and thematic categories. Return as a simple list."
            else:
                prompt = f"Expand these keywords: {keywords_text} with {request.max_concepts} related concepts. Return as a simple list."
            
            if request.context:
                prompt += f" Context: {request.context}"
        
        
        from src.providers.nlp.base_provider import ExpansionRequest
        expansion_request = ExpansionRequest(
            concept=prompt,
            media_context=request.field_name,
            max_concepts=request.max_concepts,
            field_name=request.field_name
        )
        

        result = await provider.expand_concept(expansion_request)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        

        if isinstance(result, dict) and "concepts" in result:
            concepts = result["concepts"]
        elif isinstance(result, list):
            concepts = result
        else:

            concepts = str(result).split("\n") if result else []
        

        cleaned_concepts = []
        for concept in concepts:
            if isinstance(concept, str):
                clean = concept.strip().strip("-").strip("*").strip()
                if clean and len(clean) > 1:
                    cleaned_concepts.append(clean)
            elif isinstance(concept, dict) and "concept" in concept:
                clean = str(concept["concept"]).strip()
                if clean and len(clean) > 1:
                    cleaned_concepts.append(clean)
        

        cleaned_concepts = cleaned_concepts[:request.max_concepts]
        
        return KeywordExpansionResponse(
            concepts=cleaned_concepts,
            metadata={
                "execution_time_ms": execution_time_ms,
                "expansion_style": request.expansion_style,
                "input_keywords": request.keywords,
                "result_count": len(cleaned_concepts),
                "field_name": request.field_name,
                "temperature": request.temperature
            }
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        logger.error(f"LLM keyword expansion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))






class WebSearchRequest(BaseModel):
    """Request for web search with template substitution."""
    search_templates: List[str] = Field(..., min_items=1, max_items=5, description="Search query templates")
    template_variables: Dict[str, Any] = Field(default_factory=dict, description="Variables for template substitution")
    max_results_per_template: int = Field(default=10, ge=1, le=20)
    domains: Optional[List[str]] = Field(default=None, description="Preferred domains to search")
    content_types: Optional[List[str]] = Field(default=None, description="Content types: article, review, academic")
    date_range: Optional[str] = Field(default=None, description="Date range filter")


class WebSearchProcessRequest(BaseModel):
    """Request for LLM processing of web search results."""
    search_results: List[Dict[str, Any]] = Field(..., description="Raw search results to process")
    processing_prompt: str = Field(..., min_length=10, description="LLM prompt for processing results")
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    expected_format: str = Field(default="dict", description="Expected output format")
    fields: Optional[List[str]] = Field(default=None, description="Expected output fields")
    max_length: Optional[int] = Field(default=None, description="Maximum response length")


class WebSearchFullRequest(BaseModel):
    """Request for combined web search + LLM processing."""
    search_templates: List[str] = Field(..., min_items=1, max_items=5)
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    max_results_per_template: int = Field(default=10, ge=1, le=20)
    domains: Optional[List[str]] = Field(default=None)
    processing_prompt: str = Field(..., min_length=10)
    expected_format: str = Field(default="dict")
    fields: Optional[List[str]] = Field(default=None)
    rate_limit_delay: float = Field(default=2.0, ge=0.0, le=10.0, description="Delay between searches in seconds")


class WebSearchResponse(BaseModel):
    """Response from web search operations."""
    success: bool
    execution_time_ms: float
    search_results: Optional[List[Dict[str, Any]]] = None
    processed_content: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


async def perform_web_search(query: str, max_results: int = 10, domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Perform web search using SearXNG meta-search engine (safe, self-hosted).
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        domains: Optional list of domains to prioritize
        
    Returns:
        List of search result dictionaries
    """
    import httpx
    import urllib.parse
    
    logger.info(f"Performing web search: '{query}' (max_results: {max_results})")
    
    try:
        
        searxng_url = settings.SEARXNG_URL
        
        
        search_query = query
        if domains:
            domain_restriction = " site:" + " OR site:".join(domains)
            search_query = query + domain_restriction
        

        search_params = {
            'q': search_query,
            'format': 'json',
            'categories': 'general',
            'engines': 'bing,google,duckduckgo',  
            'pageno': 1
        }
        
        headers = {
            'User-Agent': 'MediaFramework/1.0 (Internal Search)',
            'Accept': 'application/json'
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{searxng_url}/search",
                params=search_params,
                headers=headers
            )
            response.raise_for_status()
            
            search_data = response.json()
            raw_results = search_data.get('results', [])
            

            results = []
            for result in raw_results[:max_results]:
                try:

                    processed_result = {
                        "title": result.get('title', 'No title'),
                        "url": result.get('url', ''),
                        "snippet": result.get('content', 'No description'),
                        "source": "searxng",
                        "engines": result.get('engines', []),
                        "score": result.get('score', 0)
                    }
                    
                    if processed_result["url"] and processed_result["title"]:
                        results.append(processed_result)
                        
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} search results for query: '{query}'")
            return results
            
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")

        return []


@app.post("/providers/llm/websearch/search", response_model=WebSearchResponse)
async def web_search_only(request: WebSearchRequest):
    """Perform web search with template substitution."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider_manager.request_count += 1
        all_results = []
        
        for template in request.search_templates:

            try:
                query = template.format(**request.template_variables)
            except KeyError as e:
                logger.warning(f"Template variable missing: {e}, using template as-is")
                query = template
            

            results = await perform_web_search(
                query=query,
                max_results=request.max_results_per_template,
                domains=request.domains
            )
            
            
            for result in results:
                result['query'] = query
                result['template'] = template
            
            all_results.extend(results)
            

            if len(request.search_templates) > 1:
                await asyncio.sleep(2.0)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return WebSearchResponse(
            success=True,
            execution_time_ms=execution_time_ms,
            search_results=all_results,
            metadata={
                "total_results": len(all_results),
                "templates_used": len(request.search_templates),
                "domains_filtered": request.domains,
                "queries_executed": [template.format(**request.template_variables) for template in request.search_templates]
            }
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Web search failed: {e}")
        
        return WebSearchResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            metadata={"error": "web_search_failed"},
            error_message=str(e)
        )


@app.post("/providers/llm/websearch/process", response_model=WebSearchResponse)
async def process_search_results(request: WebSearchProcessRequest):
    """Process web search results using LLM."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        provider_manager.request_count += 1
        

        results_text = ""
        for i, result in enumerate(request.search_results, 1):
            results_text += f"\n--- Result {i} ---\n"
            results_text += f"Title: {result.get('title', 'N/A')}\n"
            results_text += f"Source: {result.get('url', 'N/A')}\n"
            results_text += f"Content: {result.get('snippet', 'N/A')}\n"
        

        try:
            prompt = request.processing_prompt.format(
                search_results=results_text,
                **request.template_variables
            )
        except KeyError as e:
            logger.warning(f"Template variable missing in prompt: {e}")
            prompt = request.processing_prompt.replace("{search_results}", results_text)
        

        result = await _perform_content_analysis(
            provider, 
            prompt, 
            request.expected_format, 
            request.fields,
            request.max_length
        )
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        

        if result and result.get("success", False):
            processed_content = result.get("analysis", {})
        else:
            processed_content = {"error": result.get("error", "Content analysis failed") if result else "No result returned"}
        
        return WebSearchResponse(
            success=result.get("success", False) if result else False,
            execution_time_ms=execution_time_ms,
            processed_content=processed_content,
            metadata={
                "input_results_count": len(request.search_results),
                "prompt_length": len(prompt),
                "llm_confidence": result.get("confidence", 0.8) if result else None,
                "expected_format": request.expected_format
            },
            error_message=result.get("error") if result and not result.get("success", False) else None
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Search result processing failed: {e}")
        
        return WebSearchResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            metadata={"error": "llm_processing_failed"},
            error_message=str(e)
        )


@app.post("/providers/llm/websearch/full", response_model=WebSearchResponse)
async def web_search_and_process(request: WebSearchFullRequest):
    """Combined web search + LLM processing."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider_manager.request_count += 1
        

        search_request = WebSearchRequest(
            search_templates=request.search_templates,
            template_variables=request.template_variables,
            max_results_per_template=request.max_results_per_template,
            domains=request.domains
        )
        
        search_response = await web_search_only(search_request)
        
        if not search_response.success or not search_response.search_results:
            return WebSearchResponse(
                success=False,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                metadata={"error": "web_search_failed", "search_error": search_response.error_message},
                error_message="Web search failed or returned no results"
            )
        

        process_request = WebSearchProcessRequest(
            search_results=search_response.search_results,
            processing_prompt=request.processing_prompt,
            template_variables=request.template_variables,
            expected_format=request.expected_format,
            fields=request.fields
        )
        
        process_response = await process_search_results(process_request)
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return WebSearchResponse(
            success=process_response.success,
            execution_time_ms=execution_time_ms,
            search_results=search_response.search_results,
            processed_content=process_response.processed_content,
            metadata={
                **search_response.metadata,
                **process_response.metadata,
                "total_execution_time_ms": execution_time_ms
            },
            error_message=process_response.error_message
        )
        
    except Exception as e:
        provider_manager.error_count += 1
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        logger.error(f"Combined web search and processing failed: {e}")
        
        return WebSearchResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            metadata={"error": "combined_operation_failed"},
            error_message=str(e)
        )






class ModelStatusResponse(BaseModel):
    """Response for model status check."""
    success: bool
    execution_time_ms: float
    models: Dict[str, Any]
    summary: Dict[str, Any]
    error_message: Optional[str] = None


class ModelDownloadRequest(BaseModel):
    """Request to download models."""
    model_ids: Optional[List[str]] = Field(default=None, description="Specific model IDs to download, or None for all required")
    force_download: bool = Field(default=False, description="Force re-download existing models")


class ModelDownloadResponse(BaseModel):
    """Response for model download."""
    success: bool
    execution_time_ms: float
    downloaded_models: List[str]
    failed_models: List[str]
    error_message: Optional[str] = None


@app.get("/models/status", response_model=ModelStatusResponse)
async def get_models_status():
    """Get status of Ollama models."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        settings = get_settings()
        
        
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{settings.OLLAMA_INGESTION_BASE_URL}/api/tags")
                ollama_models = response.json().get("models", [])
            except Exception as e:
                logger.warning(f"Failed to get Ollama models: {e}")
                ollama_models = []
        

        required_models = [
            "llama3.2:1b",
            "llama3.2:3b" 
        ]
        
        
        models_info = {}
        available_count = 0
        required_count = len(required_models)
        
        for model_name in required_models:
            model_available = any(model["name"] == model_name for model in ollama_models)
            
            models_info[model_name.replace(":", "_")] = {
                "name": model_name,
                "package": "ollama",
                "storage_path": "/root/.ollama",
                "size_mb": 0,
                "required": True,
                "status": "available" if model_available else "missing",
                "error_message": None if model_available else "Model not downloaded"
            }
            
            if model_available:
                available_count += 1
        
        
        for model in ollama_models:
            model_key = model["name"].replace(":", "_")
            if model_key not in models_info:
                models_info[model_key] = {
                    "name": model["name"],
                    "package": "ollama", 
                    "storage_path": "/root/.ollama",
                    "size_mb": model.get("size", 0) // (1024 * 1024),
                    "required": False,
                    "status": "available",
                    "error_message": None
                }
                available_count += 1
        
        summary = {
            "total_models": len(models_info),
            "available_models": available_count,
            "required_models": required_count,
            "missing_required": required_count - sum(1 for m in models_info.values() if m["required"] and m["status"] == "available")
        }
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ModelStatusResponse(
            success=True,
            execution_time_ms=execution_time_ms,
            models=models_info,
            summary=summary
        )
        
    except Exception as e:
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.error(f"Failed to get models status: {e}")
        
        return ModelStatusResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            models={},
            summary={"total_models": 0, "available_models": 0, "required_models": 0, "missing_required": 0},
            error_message=str(e)
        )


@app.post("/models/download", response_model=ModelDownloadResponse)
async def download_models(request: ModelDownloadRequest):
    """Download Ollama models."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        provider = provider_manager.get_provider()
        settings = get_settings()
        

        required_models = ["llama3.2:1b", "llama3.2:3b"]
        

        if request.model_ids:

            models_to_download = [mid.replace("_", ":") for mid in request.model_ids]
        else:
            models_to_download = required_models
        
        downloaded_models = []
        failed_models = []
        
        
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                response = await client.get(f"{settings.OLLAMA_INGESTION_BASE_URL}/api/tags")
                current_models = {model["name"] for model in response.json().get("models", [])}
            except Exception as e:
                logger.warning(f"Failed to get current models: {e}")
                current_models = set()
        
        for model_name in models_to_download:
            try:

                if model_name in current_models and not request.force_download:
                    downloaded_models.append(model_name.replace(":", "_"))
                    logger.info(f"Model {model_name} already available, skipping download")
                    continue
                
                logger.info(f"Downloading model: {model_name}")
                

                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(
                        f"{settings.OLLAMA_INGESTION_BASE_URL}/api/pull",
                        json={"name": model_name},
                        timeout=600.0
                    )
                    
                    if response.status_code == 200:
                        downloaded_models.append(model_name.replace(":", "_"))
                        logger.info(f"✅ Successfully downloaded {model_name}")
                    else:
                        failed_models.append(model_name.replace(":", "_"))
                        logger.error(f"❌ Failed to download {model_name}: HTTP {response.status_code}")
                        
            except Exception as e:
                failed_models.append(model_name.replace(":", "_"))
                logger.error(f"❌ Error downloading {model_name}: {e}")
        
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        success = len(failed_models) == 0
        
        return ModelDownloadResponse(
            success=success,
            execution_time_ms=execution_time_ms,
            downloaded_models=downloaded_models,
            failed_models=failed_models,
            error_message=None if success else f"Failed to download {len(failed_models)} models"
        )
        
    except Exception as e:
        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        logger.error(f"Failed to download models: {e}")
        
        return ModelDownloadResponse(
            success=False,
            execution_time_ms=execution_time_ms,
            downloaded_models=[],
            failed_models=request.model_ids or ["all"],
            error_message=str(e)
        )


@app.get("/models/ready")
async def models_ready_check():
    """Check if all required models are available."""
    try:
        provider = provider_manager.get_provider()
        
        
        status_response = await get_models_status()
        
        if status_response.success:
            missing_required = status_response.summary.get("missing_required", 0)
            if missing_required == 0:
                return {"models_ready": True}
            else:
                return {"models_ready": False, "missing_models": missing_required}, 503
        else:
            return {"models_ready": False, "error": "Failed to check model status"}, 503
            
    except Exception as e:
        logger.error(f"Models ready check failed: {e}")
        return {"models_ready": False, "error": str(e)}, 503


async def _perform_content_analysis(
    provider,
    analysis_prompt: str, 
    expected_format: str = "dict",
    fields: Optional[List[str]] = None,
    max_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform content analysis using LLM without going through concept expansion.
    
    This keeps the LLM provider focused on concept expansion while handling
    content analysis at the service level.
    """
    try:
        
        system_prompt = "You are an expert content analyzer. Your task is to analyze and synthesize information from web search results."
        
        if expected_format == "dict" and fields:
            field_descriptions = ", ".join(fields)
            system_prompt += f"\n\nYou must return your analysis as a JSON object with these exact fields: {field_descriptions}. Each field should contain relevant insights extracted from the provided content."
            system_prompt += "\n\nExample format:\n{"
            for i, field in enumerate(fields):
                system_prompt += f'"{field}": "your analysis here"'
                if i < len(fields) - 1:
                    system_prompt += ", "
            system_prompt += "}"
        else:
            system_prompt += "\n\nProvide a structured analysis of the content."
        
        system_prompt += "\n\nBe factual, concise, and focus on extracting meaningful insights from the provided search results."
        
        
        from src.providers.llm.base_llm_client import LLMRequest
        
        llm_request = LLMRequest(
            prompt=analysis_prompt,
            concept="content_analysis",
            media_context="websearch",
            max_tokens=max_length or 2000,
            temperature=0.2,
            system_prompt=system_prompt
        )
        

        llm_response = await provider.client.generate_completion(llm_request)
        
        if not llm_response.success:
            return {
                "success": False,
                "error": f"LLM completion failed: {llm_response.error_message}",
                "confidence": 0.0
            }
        

        analysis_result = _parse_analysis_response(
            llm_response.text,
            expected_format,
            fields
        )
        
        if not analysis_result:
            return {
                "success": False,
                "error": "No analysis result extracted from LLM response",
                "confidence": 0.0
            }
        
        return {
            "success": True,
            "analysis": analysis_result,
            "confidence": 0.8,
            "model_used": llm_response.model
        }
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        return {
            "success": False,
            "error": f"Content analysis failed: {str(e)}",
            "confidence": 0.0
        }


def _parse_analysis_response(
    response_text: str,
    expected_format: str,
    fields: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Parse structured analysis response from LLM."""
    try:
        text = response_text.strip()
        
        if expected_format == "dict":

            import json
            import re
            

            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    

                    if fields:
                        validated_result = {}
                        for field in fields:
                            validated_result[field] = result.get(field, f"No {field} analysis available")
                        return validated_result
                    else:
                        return result
                except json.JSONDecodeError:
                    pass
            

            if fields:
                result = {}
                for field in fields:

                    field_pattern = rf'{field}[:\-]\s*([^\n]+)'
                    match = re.search(field_pattern, text, re.IGNORECASE)
                    if match:
                        result[field] = match.group(1).strip()
                    else:
                        result[field] = f"No {field} analysis found"
                return result
            else:

                return {"analysis": text}
        else:

            return {"analysis": text}
        
    except Exception as e:
        logger.error(f"Failed to parse analysis response: {e}")
        return {"error": f"Failed to parse analysis: {str(e)}", "raw_response": response_text}


if __name__ == "__main__":
    import uvicorn
    

    uvicorn.run(
        "src.services.provider_services.llm_provider_service:app",
        host="0.0.0.0",
        port=settings.LLM_SERVICE_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )