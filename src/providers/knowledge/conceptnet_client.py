"""
ConceptNet API Client with rate limiting and error handling.
Used by ConceptExpander for actual ConceptNet API calls.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConceptNetResponse:
    """Structured response from ConceptNet API."""
    concepts: List[str]
    confidence_scores: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    api_call_time_ms: float = 0.0


class ConceptNetClient:
    """
    ConceptNet API client with rate limiting and graceful error handling.
    
    Provides simple interface for concept expansion while respecting
    ConceptNet's usage guidelines and handling network failures.
    """
    
    BASE_URL = "https://api.conceptnet.io"
    DEFAULT_LIMIT = 20
    MIN_WEIGHT = 1.0  # Minimum edge weight to consider
    
    def __init__(self, rate_limit_per_second: float = 3.0, timeout_seconds: int = 10):
        """
        Initialize ConceptNet client.
        
        Args:
            rate_limit_per_second: Maximum requests per second (ConceptNet guideline: ~3/sec)
            timeout_seconds: HTTP request timeout
        """
        self.rate_limit_per_second = rate_limit_per_second
        self.timeout_seconds = timeout_seconds
        self._last_request_time = datetime.min
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self.rate_limit_per_second <= 0:
            return
        
        time_since_last = datetime.now() - self._last_request_time
        min_interval = timedelta(seconds=1.0 / self.rate_limit_per_second)
        
        if time_since_last < min_interval:
            sleep_time = (min_interval - time_since_last).total_seconds()
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = datetime.now()
    
    async def expand_concept(
        self,
        concept: str,
        media_context: str = "movie",
        limit: int = DEFAULT_LIMIT
    ) -> ConceptNetResponse:
        """
        Expand a concept using ConceptNet /related endpoint.
        
        Args:
            concept: Term to expand (e.g., "action")
            media_context: Context for expansion (used in cache but not API)
            limit: Maximum number of related concepts to return
            
        Returns:
            ConceptNetResponse with expanded concepts and confidence scores
        """
        start_time = datetime.now()
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Build API URL  
            # Try compound term first, then fall back to first word if no results
            concept_normalized = concept.lower().replace(' ', '_')
            url = f"{self.BASE_URL}/query"
            params = {
                "node": f"/c/en/{concept_normalized}",
                "rel": "/r/RelatedTo",
                "limit": limit
            }
            
            session = await self._get_session()
            
            logger.debug(f"ConceptNet API call: {url}")
            async with session.get(url, params=params) as response:
                
                if response.status != 200:
                    error_msg = f"ConceptNet API returned status {response.status}"
                    logger.warning(error_msg)
                    return ConceptNetResponse(
                        concepts=[],
                        confidence_scores={},
                        success=False,
                        error_message=error_msg,
                        api_call_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
                
                data = await response.json()
                
                # If no results and concept has multiple words, try first word
                if not data.get("edges") and " " in concept:
                    logger.debug(f"No results for compound term '{concept}', trying first word")
                    first_word = concept.split()[0]
                    fallback_params = {
                        "node": f"/c/en/{first_word.lower()}",
                        "rel": "/r/RelatedTo", 
                        "limit": limit
                    }
                    
                    async with session.get(url, params=fallback_params) as fallback_response:
                        if fallback_response.status == 200:
                            fallback_data = await fallback_response.json()
                            if fallback_data.get("edges"):
                                logger.debug(f"Using fallback results for '{first_word}'")
                                return self._parse_conceptnet_response(fallback_data, first_word, start_time)
                
                return self._parse_conceptnet_response(data, concept, start_time)
        
        except asyncio.TimeoutError:
            error_msg = f"ConceptNet API timeout after {self.timeout_seconds}s"
            logger.warning(error_msg)
            return ConceptNetResponse(
                concepts=[],
                confidence_scores={},
                success=False,
                error_message=error_msg,
                api_call_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        except Exception as e:
            error_msg = f"ConceptNet API error: {str(e)}"
            logger.error(error_msg)
            return ConceptNetResponse(
                concepts=[],
                confidence_scores={},
                success=False,
                error_message=error_msg,
                api_call_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _parse_conceptnet_response(self, data: Dict[str, Any], input_concept: str, start_time: datetime) -> ConceptNetResponse:
        """
        Parse ConceptNet API response into structured format.
        
        Extracts related concepts and their confidence scores from the response.
        """
        try:
            concepts = []
            confidence_scores = {}
            
            # Parse edges from query response
            edges = data.get("edges", [])
            
            for edge in edges:
                # Extract weight
                weight = edge.get("weight", 0.0)
                
                # Skip if weight too low
                if weight < self.MIN_WEIGHT:
                    continue
                
                # Extract related concept from 'start' node
                start_node = edge.get("start", {})
                start_uri = start_node.get("@id", "")
                start_label = start_node.get("label", "")
                start_lang = start_node.get("language", "")
                
                # Also check 'end' node in case the structure is reversed
                end_node = edge.get("end", {}) 
                end_uri = end_node.get("@id", "")
                end_label = end_node.get("label", "")
                end_lang = end_node.get("language", "")
                
                # Prefer English concepts, extract from both start and end nodes
                concept_candidates = []
                
                # Check start node
                if start_lang == "en" and start_uri.startswith("/c/en/"):
                    concept_candidates.append((start_label or self._extract_term_from_uri(start_uri), start_uri))
                
                # Check end node  
                if end_lang == "en" and end_uri.startswith("/c/en/"):
                    concept_candidates.append((end_label or self._extract_term_from_uri(end_uri), end_uri))
                
                # Process valid English concepts
                for concept_term, uri in concept_candidates:
                    if not concept_term or len(concept_term.strip()) < 2:
                        continue
                    
                    # Clean up the concept term
                    concept_term = concept_term.strip().lower()
                    
                    # Skip if it's the same as input concept  
                    input_normalized = input_concept.lower().replace(" ", "_").replace("-", "_")
                    term_normalized = concept_term.replace(" ", "_").replace("-", "_")
                    if term_normalized == input_normalized:
                        continue
                    
                    # Convert weight to confidence score (0.0 to 1.0)
                    # ConceptNet weights typically range from 1.0 to 10.0+
                    confidence = min(weight / 10.0, 1.0)
                    
                    concepts.append(concept_term)
                    confidence_scores[concept_term] = confidence
            
            api_call_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.debug(f"ConceptNet returned {len(concepts)} concepts in {api_call_time:.1f}ms")
            
            return ConceptNetResponse(
                concepts=concepts,
                confidence_scores=confidence_scores,
                success=True,
                api_call_time_ms=api_call_time
            )
        
        except Exception as e:
            error_msg = f"Failed to parse ConceptNet response: {str(e)}"
            logger.error(error_msg)
            return ConceptNetResponse(
                concepts=[],
                confidence_scores={},
                success=False,
                error_message=error_msg,
                api_call_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _extract_term_from_uri(self, uri: str) -> str:
        """
        Extract concept term from ConceptNet URI.
        
        Handles complex URIs like:
        - /c/en/example -> "example"
        - /c/en/example/n/wn/cognition -> "example"
        - /c/en/dark_comedy -> "dark comedy"
        """
        if not uri.startswith("/c/en/"):
            return ""
        
        # Remove /c/en/ prefix
        term_part = uri[6:]
        
        # Split on / and take first part (the actual term)
        term = term_part.split("/")[0]
        
        # Convert underscores to spaces
        term = term.replace("_", " ")
        
        return term.strip()
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global client instance for reuse
_conceptnet_client: Optional[ConceptNetClient] = None


def get_conceptnet_client() -> ConceptNetClient:
    """Get singleton ConceptNet client instance."""
    global _conceptnet_client
    if _conceptnet_client is None:
        _conceptnet_client = ConceptNetClient()
    return _conceptnet_client