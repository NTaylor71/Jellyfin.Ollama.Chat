"""
Movie Summary Enhancer Plugin

Generates LLM-enhanced summaries optimized for search matching.
Creates rich, searchable summaries that capture user search patterns
without being visible in the UI.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
import httpx
import structlog

from ..base import EmbedDataEmbellisherPlugin, PluginMetadata, PluginResourceRequirements, PluginExecutionContext, PluginExecutionResult
from ...shared.hardware_config import get_hardware_profile

logger = structlog.get_logger(__name__)


class MovieSummaryEnhancerPlugin(EmbedDataEmbellisherPlugin):
    """
    Enhances movie metadata with LLM-generated searchable summaries.
    
    Takes raw Jellyfin movie data and creates enhanced summaries that:
    - Use common search terms and patterns
    - Fill gaps in sparse official descriptions  
    - Handle synonyms and natural language queries
    - Boost search relevance without affecting UI
    """
    
    def __init__(self):
        super().__init__()
        self.name = "MovieSummaryEnhancer"
        self.version = "1.0.0"
        self.description = "Generates LLM-enhanced summaries for better movie search"
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 2.0
        self.timeout = 30.0
        self.max_summary_length = 300
        
        # Hardware adaptation - will be loaded async
        self.hardware_config = None
        self.processing_strategy = "standard"  # Default until hardware is loaded
        
        logger.info("Initialized MovieSummaryEnhancer", 
                   strategy=self.processing_strategy)
    
    # Required abstract method implementations
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata for registration."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author="System",
            plugin_type="embed_data_embellisher"
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """Resource requirements for this plugin."""
        reqs = self.get_resource_requirements()
        
        # Map our strategy to resource requirements
        if self.processing_strategy == "enhanced":
            return PluginResourceRequirements(
                min_cpu_cores=2.0,
                preferred_cpu_cores=4.0,
                min_memory_mb=256.0,
                preferred_memory_mb=512.0,
                max_execution_time_seconds=30.0
            )
        elif self.processing_strategy == "standard":
            return PluginResourceRequirements(
                min_cpu_cores=1.0,
                preferred_cpu_cores=2.0,
                min_memory_mb=128.0,
                preferred_memory_mb=256.0,
                max_execution_time_seconds=20.0
            )
        else:  # minimal
            return PluginResourceRequirements(
                min_cpu_cores=1.0,
                preferred_cpu_cores=1.0,
                min_memory_mb=64.0,
                preferred_memory_mb=128.0,
                max_execution_time_seconds=15.0
            )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            await self._initialize_hardware()
            logger.info("Plugin initialized successfully", strategy=self.processing_strategy)
            return True
        except Exception as e:
            logger.error("Plugin initialization failed", error=str(e))
            return False
    
    async def execute(self, data: Any, context: PluginExecutionContext) -> PluginExecutionResult:
        """Execute the plugin with given data and context."""
        try:
            # Convert dataclass to dict
            from dataclasses import asdict
            context_dict = asdict(context) if context else {}
            result_data = await self.process_data(data, context_dict)
            return PluginExecutionResult(
                success=True,
                data=result_data,
                metadata={"processing_strategy": self.processing_strategy}
            )
        except Exception as e:
            logger.error("Plugin execution failed", error=str(e))
            return PluginExecutionResult(
                success=False,
                data=data,
                error=str(e)
            )
    
    async def embellish_query(self, query: str, context: PluginExecutionContext) -> str:
        """Embellish the input query and return enhanced version."""
        # This plugin focuses on data embellishment, not query embellishment
        return query
    
    async def embellish_embed_data(self, data: Dict[str, Any], context: PluginExecutionContext) -> Dict[str, Any]:
        """Embellish data before embedding."""
        # Convert dataclass to dict
        from dataclasses import asdict
        context_dict = asdict(context) if context else {}
        return await self.process_data(data, context_dict)
    
    async def handle_faiss_operation(self, operation: str, data: Dict[str, Any], context: PluginExecutionContext) -> Any:
        """Handle FAISS CRUD operations."""
        # This plugin doesn't handle FAISS operations directly
        return data
    
    async def _initialize_hardware(self):
        """Initialize hardware configuration asynchronously."""
        if self.hardware_config is None:
            try:
                self.hardware_config = await get_hardware_profile()
                self.processing_strategy = self._determine_strategy()
                logger.info("Hardware configuration loaded", 
                           strategy=self.processing_strategy,
                           cpu_cores=self.hardware_config.local_cpu_cores)
            except Exception as e:
                logger.warning("Failed to load hardware config, using defaults", error=str(e))
                # Create minimal fallback config
                from ...shared.hardware_config import HardwareProfile
                self.hardware_config = HardwareProfile()  # Uses defaults
                self.processing_strategy = "minimal"
    
    def _determine_strategy(self) -> str:
        """Determine processing strategy based on available hardware."""
        if not self.hardware_config:
            return "standard"
        
        cpu_cores = getattr(self.hardware_config, 'local_cpu_cores', 4)
        if cpu_cores >= 16:
            return "enhanced"  # Use advanced prompting
        elif cpu_cores >= 8:
            return "standard"  # Standard processing
        else:
            return "minimal"   # Simple processing
    
    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return resource requirements based on strategy."""
        requirements = {
            "minimal": {"cpu_cores": 1, "memory_mb": 128, "priority": "low"},
            "standard": {"cpu_cores": 2, "memory_mb": 256, "priority": "medium"},
            "enhanced": {"cpu_cores": 4, "memory_mb": 512, "priority": "high"}
        }
        return requirements.get(self.processing_strategy, requirements["minimal"])
    
    def _extract_movie_info(self, movie_data: Dict[str, Any]) -> str:
        """Extract and format movie information for LLM processing."""
        # Core details
        name = movie_data.get("name", "Unknown Movie")
        year = movie_data.get("production_year", "Unknown")
        overview = movie_data.get("overview", "No description available")
        
        # People
        people = movie_data.get("people", [])
        directors = [p["name"] for p in people if p.get("type") == "Director"]
        actors = [p["name"] for p in people if p.get("type") == "Actor"]
        
        # Classification
        genres = movie_data.get("genres", [])
        rating = movie_data.get("official_rating", "Not Rated")
        
        # Runtime
        runtime_ticks = movie_data.get("run_time_ticks")
        runtime = f"{int(runtime_ticks / 600000000)} minutes" if runtime_ticks else "Unknown runtime"
        
        # Format for LLM
        formatted = f"""Movie: {name} ({year})
Genres: {', '.join(genres) if genres else 'Unknown'}
Rating: {rating}
Runtime: {runtime}
Directors: {', '.join(directors[:2]) if directors else 'Unknown'}
Main Cast: {', '.join(actors[:4]) if actors else 'Unknown'}
Official Description: {overview[:200]}{'...' if len(overview) > 200 else ''}"""
        
        return formatted
    
    def _create_enhancement_prompt(self, movie_info: str) -> str:
        """Create LLM prompt optimized for the processing strategy."""
        base_prompt = f"""You are a movie search optimization expert.

Given this movie information:
{movie_info}

Create a enhanced summary optimized for search matching. Focus on:
- Natural language people use when searching
- Alternative ways to describe themes and concepts  
- Common search patterns and synonyms
- Genre crossovers and sub-categories

Return VALID JSON ONLY:
{{
  "enhanced_summary": "2-3 sentence summary using searchable language"
}}

RULES:
- NO markdown, code blocks, or explanations
- Use terms people actually search for
- Include genre variations (e.g., 'psychological thriller', 'buddy cop')
- Mention mood/tone if relevant (dark, comedy, action-packed)
- 2-3 sentences maximum
- JSON only"""

        if self.processing_strategy == "enhanced":
            base_prompt += """
- Include subtle search hooks (e.g., 'cult classic', 'hidden gem')
- Consider decade/era associations  
- Reference similar well-known movies if appropriate"""
        
        return base_prompt
    
    async def _call_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make request to Ollama with retry logic."""
        # Get Ollama settings from config
        from ...shared.config import get_settings
        settings = get_settings()
        
        url = f"{settings.ollama_chat_url}/api/generate"
        
        payload = {
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower for more consistent results
                "top_p": 0.9,
                "num_predict": 150   # Limit response length
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    raw_response = result.get("response", "").strip()
                    
                    # Parse JSON response
                    try:
                        parsed = json.loads(raw_response)
                        if "enhanced_summary" in parsed:
                            return parsed
                        else:
                            logger.warning("Missing enhanced_summary field", 
                                         response=raw_response[:100])
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON", 
                                     attempt=attempt + 1, 
                                     error=str(e),
                                     response=raw_response[:100])
                
            except Exception as e:
                logger.warning("Ollama request failed", 
                             attempt=attempt + 1, 
                             error=str(e))
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        logger.error("All Ollama attempts failed")
        return None
    
    def _create_fallback_summary(self, movie_data: Dict[str, Any]) -> str:
        """Create fallback summary when LLM fails."""
        name = movie_data.get("name", "Unknown Movie")
        genres = movie_data.get("genres", [])
        year = movie_data.get("production_year")
        overview = movie_data.get("overview", "")
        
        # Extract key actors/directors
        people = movie_data.get("people", [])
        directors = [p["name"] for p in people if p.get("type") == "Director"]
        actors = [p["name"] for p in people if p.get("type") == "Actor"]
        
        # Build enhanced summary from available data
        summary_parts = []
        
        if year and year >= 2020:
            summary_parts.append("Recent")
        elif year and year >= 2010:
            summary_parts.append("Modern")
        elif year and year >= 2000:
            summary_parts.append("2000s")
        elif year and year >= 1990:
            summary_parts.append("90s")
        elif year and year < 1990:
            summary_parts.append("Classic")
        
        if genres:
            if "Action" in genres and "Comedy" in genres:
                summary_parts.append("action comedy")
            elif "Horror" in genres and "Comedy" in genres:
                summary_parts.append("horror comedy")
            elif "Science Fiction" in genres:
                summary_parts.append("sci-fi")
            elif len(genres) > 1:
                summary_parts.append(f"{genres[0].lower()}-{genres[1].lower()}")
            else:
                summary_parts.append(genres[0].lower())
        
        summary_parts.append("film")
        
        if directors:
            summary_parts.append(f"directed by {directors[0]}")
        
        if actors:
            summary_parts.append(f"starring {actors[0]}")
            if len(actors) > 1:
                summary_parts.append(f"and {actors[1]}")
        
        # Create summary
        summary = " ".join(summary_parts).capitalize()
        
        # Add brief plot if available
        if overview and len(overview) > 20:
            plot_snippet = overview[:80].split('.')[0]
            summary += f". {plot_snippet}."
        
        return summary[:self.max_summary_length]
    
    async def process_data(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process movie data to generate enhanced summary.
        
        Args:
            data: Movie data dictionary
            context: Processing context
            
        Returns:
            Enhanced data with summary in enhanced_fields
        """
        start_time = time.time()
        
        # Initialize hardware config if needed
        await self._initialize_hardware()
        
        try:
            # Extract movie information
            movie_info = self._extract_movie_info(data)
            
            # Create prompt
            prompt = self._create_enhancement_prompt(movie_info)
            
            # Call LLM
            llm_result = await self._call_ollama(prompt)
            
            if llm_result and "enhanced_summary" in llm_result:
                enhanced_summary = llm_result["enhanced_summary"]
                
                # Validate length
                if len(enhanced_summary) > self.max_summary_length:
                    enhanced_summary = enhanced_summary[:self.max_summary_length].rsplit(' ', 1)[0] + "..."
                
                logger.info("Generated enhanced summary", 
                           movie=data.get("name", "Unknown"),
                           strategy=self.processing_strategy,
                           summary_length=len(enhanced_summary))
            else:
                # Fallback to rule-based summary
                enhanced_summary = self._create_fallback_summary(data)
                logger.info("Used fallback summary", 
                           movie=data.get("name", "Unknown"))
            
            # Add to enhanced fields
            if "enhanced_fields" not in data:
                data["enhanced_fields"] = {}
            
            data["enhanced_fields"]["summary"] = enhanced_summary
            
            # Update processing metrics
            processing_time = time.time() - start_time
            self._update_metrics("summary_generation", processing_time, True)
            
            return data
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics("summary_generation", processing_time, False)
            
            logger.error("Summary enhancement failed", 
                        movie=data.get("name", "Unknown"),
                        error=str(e))
            
            # Return original data unchanged on error
            return data
    
    def _update_metrics(self, operation: str, duration: float, success: bool):
        """Update plugin performance metrics."""
        # This would integrate with the monitoring system
        # For now, just log the metrics
        logger.debug("Plugin metrics", 
                    operation=operation,
                    duration_ms=int(duration * 1000),
                    success=success,
                    strategy=self.processing_strategy)
    


# Register the plugin
def get_plugin():
    """Plugin factory function."""
    return MovieSummaryEnhancerPlugin()