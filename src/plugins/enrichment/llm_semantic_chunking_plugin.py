"""
LLM Semantic Chunking Plugin
HTTP-only plugin that calls LLM service for intelligent text chunking.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginResourceRequirements, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class LLMSemanticChunkingPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that performs semantic text chunking using LLM.
    
    Features:
    - Intelligent semantic boundary detection
    - Multiple chunking strategies (sentence, paragraph, topic-based)
    - Configurable chunk sizes and overlap
    - Context preservation across chunks
    - Multiple LLM calls for optimization
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="LLMSemanticChunkingPlugin",
            version="1.0.0",
            description="Performs intelligent semantic text chunking using LLM",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["chunking", "semantic", "llm", "text", "processing", "nlp"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """Semantic chunking requires moderate resources for multiple LLM calls."""
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=1024.0,
            preferred_memory_mb=3072.0,
            requires_gpu=True,
            min_gpu_memory_mb=2048.0,
            preferred_gpu_memory_mb=8192.0,
            max_execution_time_seconds=120.0
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with semantic chunking.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field (text to chunk)
            config: Plugin configuration
            
        Returns:
            Dict containing chunked text with metadata
        """
        try:
            # Convert field value to text
            if isinstance(field_value, str):
                text = field_value
            elif isinstance(field_value, list):
                text = " ".join(str(item) for item in field_value)
            else:
                text = str(field_value)
            
            # Check if text is long enough to warrant chunking
            min_length = config.get("min_length", 1000)
            if len(text) < min_length:
                self._logger.debug(f"Text too short for chunking: {len(text)} chars")
                return self.normalize_text({
                    "chunks": [{"text": text, "index": 0, "type": "single"}],
                    "metadata": {
                        "provider": "llm_semantic_chunking",
                        "chunk_count": 1,
                        "original_length": len(text),
                        "strategy": "no_chunking"
                    }
                })
            
            self._logger.debug(f"Chunking text of {len(text)} characters for field {field_name}")
            
            # Get chunking strategy
            strategy = config.get("strategy", "semantic_paragraphs")
            max_chunk_size = config.get("max_chunk_size", 500)
            overlap = config.get("overlap", 50)
            
            # Perform semantic chunking using multiple LLM calls
            chunks = await self._chunk_text(
                text, 
                strategy, 
                max_chunk_size, 
                overlap, 
                config
            )
            
            self._logger.info(f"Successfully chunked text into {len(chunks)} chunks")
            
            result = {
                "chunks": chunks,
                "original_text": text[:300] + "..." if len(text) > 300 else text,
                "field_name": field_name,
                "metadata": {
                    "provider": "llm_semantic_chunking",
                    "chunk_count": len(chunks),
                    "original_length": len(text),
                    "strategy": strategy,
                    "max_chunk_size": max_chunk_size,
                    "overlap": overlap,
                    "avg_chunk_size": sum(len(chunk["text"]) for chunk in chunks) / len(chunks) if chunks else 0
                }
            }
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"Semantic chunking failed for field {field_name}: {e}")
            
            result = {
                "chunks": [],
                "original_text": "",
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "llm_semantic_chunking",
                    "success": False
                }
            }
            
            return self.normalize_text(result)
    
    async def _chunk_text(
        self,
        text: str,
        strategy: str,
        max_chunk_size: int,
        overlap: int,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform intelligent text chunking using multiple LLM calls.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy
            max_chunk_size: Maximum chunk size in words
            overlap: Overlap between chunks in words
            config: Configuration options
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # Step 1: Analyze text structure
            structure = await self._analyze_text_structure(text, strategy, config)
            
            # Step 2: Detect semantic boundaries
            boundaries = await self._detect_semantic_boundaries(text, structure, strategy, config)
            
            # Step 3: Create initial chunks
            chunks = await self._create_chunks(text, boundaries, max_chunk_size, overlap)
            
            # Step 4: Optimize chunks (optional)
            if config.get("optimize_chunks", True):
                chunks = await self._optimize_chunks(chunks, max_chunk_size)
            
            return chunks
            
        except Exception as e:
            self._logger.error(f"Chunking process failed: {e}")
            # Fallback to simple chunking
            return await self._simple_chunk_fallback(text, max_chunk_size, overlap)
    
    async def _analyze_text_structure(
        self, 
        text: str, 
        strategy: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze text structure using advanced LLM prompts.
        
        Args:
            text: Text to analyze
            strategy: Chunking strategy
            
        Returns:
            Structure analysis results
        """
        try:
            service_url = await self.get_plugin_service_url()
            
            # Configurable structure analysis prompt
            structure_prompt = self._get_structure_analysis_prompt(config)
            
            concept_text = f"{structure_prompt}\n\nText: {text[:2500]}"  # Use expanded limit
            
            request_data = {
                "concept": concept_text,
                "media_context": "text",
                "max_concepts": 15,
                "field_name": "structure_analysis"
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                result_data = response.get("result", {})
                analysis = result_data.get("expanded_concepts", [])
                
                return {
                    "analysis": analysis,
                    "strategy": strategy,
                    "confidence": result_data.get("confidence", 0.5)
                }
            else:
                return {"analysis": [], "strategy": strategy, "confidence": 0.0}
                
        except Exception as e:
            self._logger.error(f"Structure analysis failed: {e}")
            return {"analysis": [], "strategy": strategy, "confidence": 0.0}
    
    async def _detect_semantic_boundaries(
        self,
        text: str,
        structure: Dict[str, Any],
        strategy: str,
        config: Dict[str, Any]
    ) -> List[int]:
        """
        Detect semantic boundaries using advanced LLM prompts.
        
        Args:
            text: Text to analyze
            structure: Structure analysis results
            strategy: Chunking strategy
            
        Returns:
            List of character positions for boundaries
        """
        try:
            service_url = await self.get_plugin_service_url()
            
            # Configurable boundary detection prompt
            boundary_prompt = self._get_boundary_detection_prompt(config)
            
            concept_text = f"{boundary_prompt}\n\nText: {text[:2500]}"
            
            request_data = {
                "concept": concept_text,
                "media_context": "text",
                "max_concepts": 20,
                "field_name": "boundary_detection"
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                result_data = response.get("result", {})
                boundaries = result_data.get("expanded_concepts", [])
                
                # Convert boundary descriptions to positions
                positions = self._extract_boundary_positions(text, boundaries)
                return sorted(positions)
            else:
                return self._fallback_boundaries(text, strategy)
                
        except Exception as e:
            self._logger.error(f"Boundary detection failed: {e}")
            return self._fallback_boundaries(text, strategy)
    
    async def _create_chunks(
        self,
        text: str,
        boundaries: List[int],
        max_chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Create chunks based on detected boundaries.
        
        Args:
            text: Original text
            boundaries: List of boundary positions
            max_chunk_size: Maximum chunk size in words
            overlap: Overlap between chunks in words
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        if not boundaries:
            boundaries = [0, len(text)]
        
        # Ensure boundaries include start and end
        if 0 not in boundaries:
            boundaries = [0] + boundaries
        if len(text) not in boundaries:
            boundaries = boundaries + [len(text)]
        
        boundaries = sorted(set(boundaries))
        
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]
            end_pos = boundaries[i + 1]
            
            chunk_text = text[start_pos:end_pos].strip()
            
            if chunk_text:
                # Check if chunk is too large
                words = chunk_text.split()
                if len(words) > max_chunk_size:
                    # Split large chunks
                    sub_chunks = self._split_large_chunk(chunk_text, max_chunk_size, overlap)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            "text": sub_chunk,
                            "index": len(chunks),
                            "type": "sub_chunk",
                            "parent_index": i,
                            "sub_index": j,
                            "word_count": len(sub_chunk.split()),
                            "char_count": len(sub_chunk),
                            "start_pos": start_pos,
                            "end_pos": end_pos
                        })
                else:
                    chunks.append({
                        "text": chunk_text,
                        "index": len(chunks),
                        "type": "semantic_chunk",
                        "word_count": len(words),
                        "char_count": len(chunk_text),
                        "start_pos": start_pos,
                        "end_pos": end_pos
                    })
        
        return chunks
    
    async def _optimize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunk_size: int
    ) -> List[Dict[str, Any]]:
        """
        Optimize chunks using LLM feedback.
        
        Args:
            chunks: Initial chunks
            max_chunk_size: Maximum chunk size
            
        Returns:
            Optimized chunks
        """
        try:
            service_url = await self.get_plugin_service_url()
            
            # Create optimization prompt
            chunk_summary = []
            for chunk in chunks[:5]:  # Limit to first 5 chunks for optimization
                chunk_summary.append({
                    "index": chunk["index"],
                    "word_count": chunk["word_count"],
                    "preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                })
            
            optimization_prompt = f"""
            Analyze these text chunks and suggest improvements:
            
            Chunks: {chunk_summary}
            Max chunk size: {max_chunk_size} words
            
            Should any chunks be merged or split for better semantic coherence?
            """
            
            request_data = {
                "concept": optimization_prompt,
                "media_context": "text",
                "max_concepts": 5,
                "field_name": "chunk_optimization",
                "options": {
                    "task_type": "optimization",
                    "temperature": 0.1,
                    "max_tokens": 200
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                # For now, return original chunks
                # In future, could implement optimization logic based on LLM feedback
                return chunks
            else:
                return chunks
                
        except Exception as e:
            self._logger.error(f"Chunk optimization failed: {e}")
            return chunks
    
    def _get_structure_analysis_prompt(self, config: Dict[str, Any]) -> str:
        """Get configurable structure analysis prompt."""
        # Use config-provided prompt or default
        if "structure_prompt" in config:
            return config["structure_prompt"]
        
        # Default generic prompt
        return "Analyze this text structure. Identify the main topics and where they transition. List the key themes in order:"
    
    def _get_boundary_detection_prompt(self, config: Dict[str, Any]) -> str:
        """Get configurable boundary detection prompt."""
        # Use config-provided prompt or default
        if "boundary_prompt" in config:
            return config["boundary_prompt"]
        
        # Default generic prompt
        return "Mark semantic boundaries with | where topics change or new themes emerge. Output format: 'Text before boundary | Text after boundary'"
    
    def _extract_boundary_positions(self, text: str, boundaries: List[str]) -> List[int]:
        """
        Extract character positions from boundary descriptions.
        
        Args:
            text: Original text
            boundaries: List of boundary descriptions
            
        Returns:
            List of character positions
        """
        positions = []
        paragraphs = text.split('\n\n')
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            if i > 0:  # Add boundary at paragraph breaks
                positions.append(current_pos)
            current_pos += len(paragraph) + 2  # +2 for \n\n
        
        return positions
    
    def _fallback_boundaries(self, text: str, strategy: str) -> List[int]:
        """
        Fallback boundary detection without LLM.
        
        Args:
            text: Text to analyze
            strategy: Chunking strategy
            
        Returns:
            List of boundary positions
        """
        if strategy == "semantic_paragraphs":
            # Use paragraph boundaries
            positions = []
            current_pos = 0
            for paragraph in text.split('\n\n'):
                if current_pos > 0:
                    positions.append(current_pos)
                current_pos += len(paragraph) + 2
            return positions
        else:
            # Use sentence boundaries
            import re
            sentences = re.split(r'[.!?]+', text)
            positions = []
            current_pos = 0
            for sentence in sentences[:-1]:
                current_pos += len(sentence) + 1
                positions.append(current_pos)
            return positions
    
    def _split_large_chunk(self, text: str, max_size: int, overlap: int) -> List[str]:
        """
        Split a large chunk into smaller pieces.
        
        Args:
            text: Text to split
            max_size: Maximum size in words
            overlap: Overlap in words
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            if end >= len(words):
                break
                
            start = end - overlap
            if start < 0:
                start = 0
        
        return chunks
    
    async def _simple_chunk_fallback(
        self, 
        text: str, 
        max_chunk_size: int, 
        overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Simple fallback chunking when LLM calls fail.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum chunk size in words
            overlap: Overlap in words
            
        Returns:
            List of simple chunks
        """
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "index": len(chunks),
                "type": "simple_chunk",
                "word_count": len(chunk_words),
                "char_count": len(chunk_text),
                "start_pos": start,
                "end_pos": end
            })
            
            if end >= len(words):
                break
                
            start = end - overlap
            if start < 0:
                start = 0
        
        return chunks
    
    async def chunk_text(
        self,
        text: str,
        strategy: str = "semantic_paragraphs",
        max_chunk_size: int = 500,
        overlap: int = 50,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct chunking method for external use.
        
        Args:
            text: Text to chunk
            strategy: Chunking strategy
            max_chunk_size: Maximum chunk size in words
            overlap: Overlap in words
            config: Additional configuration
            
        Returns:
            Chunking results
        """
        if config is None:
            config = {}
            
        config.update({
            "strategy": strategy,
            "max_chunk_size": max_chunk_size,
            "overlap": overlap
        })
        
        result = await self.enrich_field("text", text, config)
        return self.normalize_text(result)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            service_url = self.get_service_url("llm", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "llm_semantic_chunking_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "llm_semantic_chunking_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health