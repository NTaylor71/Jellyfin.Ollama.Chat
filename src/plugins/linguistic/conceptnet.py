"""
ConceptNet expansion plugin for concept-based search enhancement.
Extracts concepts from text and expands them via ConceptNet API.
"""

import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Set
import re
import logging
from collections import defaultdict, deque
import time
import threading

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from .base import DualUsePlugin


class ConceptNetRateLimiter:
    """
    Rate limiter for ConceptNet API with 90% safety margin.
    
    Limits:
    - 3600 requests/hour → 3240 requests/hour (90%)
    - 120 requests/minute → 108 requests/minute (90%)
    - Some endpoints count as 2 requests
    """
    
    def __init__(self):
        self.hourly_limit = 3240  # 90% of 3600
        self.minute_limit = 108   # 90% of 120
        
        # Track requests with timestamps
        self.hourly_requests = deque()  # (timestamp, cost)
        self.minute_requests = deque()  # (timestamp, cost)
        self.lock = threading.Lock()
        
        # Endpoint costs (some count as 2 requests)
        self.endpoint_costs = {
            '/c/': 1,           # Basic concept lookup
            '/related': 2,      # Related concepts (costs 2)
            '/relatedness': 2,  # Relatedness score (costs 2)
        }
    
    def _get_endpoint_cost(self, url: str) -> int:
        """Determine the cost of an API endpoint."""
        for endpoint, cost in self.endpoint_costs.items():
            if endpoint in url:
                return cost
        return 1  # Default cost
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove requests older than the time windows."""
        # Remove requests older than 1 hour
        hour_ago = current_time - 3600
        while self.hourly_requests and self.hourly_requests[0][0] < hour_ago:
            self.hourly_requests.popleft()
        
        # Remove requests older than 1 minute
        minute_ago = current_time - 60
        while self.minute_requests and self.minute_requests[0][0] < minute_ago:
            self.minute_requests.popleft()
    
    def _get_current_usage(self, current_time: float) -> tuple[int, int]:
        """Get current hourly and minute usage."""
        self._cleanup_old_requests(current_time)
        
        hourly_usage = sum(cost for _, cost in self.hourly_requests)
        minute_usage = sum(cost for _, cost in self.minute_requests)
        
        return hourly_usage, minute_usage
    
    async def can_make_request(self, url: str) -> bool:
        """Check if we can make a request without exceeding limits."""
        cost = self._get_endpoint_cost(url)
        current_time = time.time()
        
        with self.lock:
            hourly_usage, minute_usage = self._get_current_usage(current_time)
            
            # Check if request would exceed limits
            would_exceed_hourly = (hourly_usage + cost) > self.hourly_limit
            would_exceed_minute = (minute_usage + cost) > self.minute_limit
            
            return not (would_exceed_hourly or would_exceed_minute)
    
    async def record_request(self, url: str):
        """Record a successful request."""
        cost = self._get_endpoint_cost(url)
        current_time = time.time()
        
        with self.lock:
            self.hourly_requests.append((current_time, cost))
            self.minute_requests.append((current_time, cost))
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiting status."""
        current_time = time.time()
        
        with self.lock:
            hourly_usage, minute_usage = self._get_current_usage(current_time)
            
            return {
                "hourly_usage": hourly_usage,
                "hourly_limit": self.hourly_limit,
                "hourly_remaining": max(0, self.hourly_limit - hourly_usage),
                "minute_usage": minute_usage,
                "minute_limit": self.minute_limit,
                "minute_remaining": max(0, self.minute_limit - minute_usage),
                "hourly_percent": (hourly_usage / self.hourly_limit) * 100,
                "minute_percent": (minute_usage / self.minute_limit) * 100
            }


class ConceptNetExpansionPlugin(DualUsePlugin):
    """Expand concepts using ConceptNet for better semantic understanding."""
    
    # Global rate limiter shared across all instances
    _rate_limiter = None
    _rate_limiter_lock = threading.Lock()
    
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.api_base = "http://api.conceptnet.io"
        self.max_relations = 5
        self.min_weight = 1.0
        
        # Initialize global rate limiter if needed
        with ConceptNetExpansionPlugin._rate_limiter_lock:
            if ConceptNetExpansionPlugin._rate_limiter is None:
                ConceptNetExpansionPlugin._rate_limiter = ConceptNetRateLimiter()
        
        self.rate_limiter = ConceptNetExpansionPlugin._rate_limiter
        
    def _initialize_models(self):
        """Initialize spaCy model if available."""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model for concept extraction")
            except OSError:
                self.logger.warning("spaCy model not found, using regex fallback")
                self.nlp = None
        else:
            self.nlp = None
            self.logger.warning("spaCy not available, using regex fallback")
    
    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract and expand concepts from text."""
        try:
            # Extract concepts
            concepts = self._extract_concepts(text)
            
            # Expand concepts via ConceptNet with rate limiting
            expanded_concepts = {}
            concept_graph = {"nodes": set(), "edges": []}
            
            # Check rate limiting status first
            rate_status = self.rate_limiter.get_status()
            max_concepts = min(10, rate_status["minute_remaining"] // 2)  # Conservative estimate
            
            if max_concepts <= 0:
                self.logger.warning("ConceptNet rate limit reached, skipping expansions")
            else:
                self.logger.debug(f"ConceptNet rate status: {rate_status['minute_usage']}/{rate_status['minute_limit']} per minute")
                
                for concept in concepts[:max_concepts]:
                    expansion = await self._expand_concept(concept)
                    if expansion:
                        expanded_concepts[concept] = expansion["related"]
                        concept_graph["nodes"].add(concept)
                        concept_graph["nodes"].update(expansion["related"])
                        concept_graph["edges"].extend(expansion["edges"])
            
            # Convert sets to lists for JSON serialization
            concept_graph["nodes"] = list(concept_graph["nodes"])
            
            # Include rate limiting status in results
            rate_status = self.rate_limiter.get_status()
            
            return {
                "primary_concepts": concepts,
                "expanded_concepts": expanded_concepts,
                "concept_graph": concept_graph,
                "expansion_count": len(expanded_concepts),
                "rate_limit_status": {
                    "minute_remaining": rate_status["minute_remaining"],
                    "hourly_remaining": rate_status["hourly_remaining"],
                    "minute_percent_used": round(rate_status["minute_percent"], 1),
                    "hourly_percent_used": round(rate_status["hourly_percent"], 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in concept analysis: {e}")
            return {"error": str(e), "primary_concepts": []}
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text using spaCy or regex."""
        concepts = []
        
        if self.nlp:
            # Use spaCy for better concept extraction
            doc = self.nlp(text.lower())
            
            # Extract nouns and proper nouns
            for token in doc:
                if (token.pos_ in ["NOUN", "PROPN"] and 
                    len(token.text) > 2 and 
                    not token.is_stop and 
                    token.is_alpha):
                    concepts.append(token.lemma_)
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2:
                    # Clean and normalize
                    clean_phrase = re.sub(r'[^\w\s]', '', chunk.text.lower())
                    if clean_phrase and not any(word in clean_phrase for word in ['the', 'a', 'an']):
                        concepts.append(clean_phrase.replace(' ', '_'))
        
        else:
            # Regex fallback
            # Extract potential concepts (nouns, compound words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter common stopwords manually
            stopwords = {'the', 'and', 'but', 'for', 'are', 'with', 'this', 'that', 
                        'have', 'from', 'they', 'been', 'said', 'each', 'which', 
                        'their', 'time', 'will', 'about', 'would', 'there', 'could'}
            
            concepts = [word for word in words if word not in stopwords]
        
        # Remove duplicates and return top concepts
        unique_concepts = list(dict.fromkeys(concepts))
        return unique_concepts[:15]  # Limit for performance
    
    async def _expand_concept(self, concept: str) -> Optional[Dict[str, Any]]:
        """Expand a concept using ConceptNet API with rate limiting."""
        # Check cache first
        cache_key = f"concept_{concept}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                self.logger.debug(f"Using cached expansion for '{concept}'")
                return cached["data"]
        
        # Check rate limits before making request
        url = f"{self.api_base}/c/en/{concept}"
        
        if not await self.rate_limiter.can_make_request(url):
            rate_status = self.rate_limiter.get_status()
            self.logger.warning(
                f"ConceptNet rate limit would be exceeded for '{concept}'. "
                f"Current usage: {rate_status['minute_usage']}/{rate_status['minute_limit']} per minute, "
                f"{rate_status['hourly_usage']}/{rate_status['hourly_limit']} per hour"
            )
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        # Record the request for rate limiting
                        await self.rate_limiter.record_request(url)
                        
                        data = await response.json()
                        result = self._process_conceptnet_response(data, concept)
                        
                        # Cache result
                        self.cache[cache_key] = {
                            "data": result,
                            "timestamp": time.time()
                        }
                        
                        self.logger.debug(f"Successfully expanded concept '{concept}' with {len(result.get('related', []))} related terms")
                        return result
                    
                    elif response.status == 429:  # Rate limited by server
                        self.logger.warning(f"ConceptNet server rate limited request for '{concept}'")
                        return None
                    
                    else:
                        self.logger.debug(f"ConceptNet API returned status {response.status} for '{concept}'")
                        return None
        
        except asyncio.TimeoutError:
            self.logger.debug(f"ConceptNet API timeout for '{concept}'")
        except Exception as e:
            self.logger.debug(f"ConceptNet API error for '{concept}': {e}")
        
        return None
    
    def _process_conceptnet_response(self, data: Dict, concept: str) -> Dict[str, Any]:
        """Process ConceptNet API response to extract related concepts."""
        related_concepts = []
        edges = []
        
        # Get edges (relationships)
        edges_data = data.get("edges", [])
        
        # Priority order for relationship types (most useful first)
        relation_priority = {
            "Synonym": 3,
            "RelatedTo": 2,
            "IsA": 2,
            "PartOf": 1,
            "HasA": 1,
            "UsedFor": 1,
            "CapableOf": 1,
            "Causes": 1
        }
        
        # Sort edges by relationship priority and weight
        sorted_edges = sorted(
            edges_data,
            key=lambda e: (
                relation_priority.get(e.get("rel", {}).get("label", ""), 0),
                e.get("weight", 0)
            ),
            reverse=True
        )
        
        for edge in sorted_edges[:self.max_relations * 2]:  # Process more to get better results
            try:
                # Get relationship info
                rel_info = edge.get("rel", {})
                rel_type = rel_info.get("label", "")
                weight = edge.get("weight", 0)
                
                if weight < self.min_weight:
                    continue
                
                # Skip low-priority relationships unless they're high weight
                if relation_priority.get(rel_type, 0) == 0 and weight < 2.5:
                    continue
                
                # Extract start and end nodes
                start_node = edge.get("start", {})
                end_node = edge.get("end", {})
                
                start_label = start_node.get("label", "")
                end_label = end_node.get("label", "")
                start_lang = start_node.get("language", "")
                end_lang = end_node.get("language", "")
                
                # Focus on English concepts for better relevance
                start_is_english = start_lang == "en"
                end_is_english = end_lang == "en"
                
                # Clean labels
                start_clean = self._clean_concept_label(start_label)
                end_clean = self._clean_concept_label(end_label)
                
                if not start_clean or not end_clean:
                    continue
                
                # Determine which concept is the target concept and which is related
                target_concept = None
                related_concept = None
                
                if start_clean.lower() == concept.lower():
                    target_concept = start_clean
                    related_concept = end_clean
                    related_is_english = end_is_english
                elif end_clean.lower() == concept.lower():
                    target_concept = end_clean
                    related_concept = start_clean
                    related_is_english = start_is_english
                else:
                    # Neither matches exactly, skip
                    continue
                
                # Prefer English concepts, but allow high-quality non-English ones
                if not related_is_english and weight < 2.5:
                    continue
                
                # Filter out poor quality concepts
                if (len(related_concept) <= 2 or 
                    related_concept.lower() == target_concept.lower() or
                    related_concept.lower() in ['a', 'an', 'the', 'and', 'or', 'but']):
                    continue
                
                related_concepts.append(related_concept)
                
                # Add edge to graph
                edges.append({
                    "source": target_concept,
                    "target": related_concept,
                    "relation": rel_type,
                    "weight": weight,
                    "source_language": start_lang if target_concept == start_clean else end_lang,
                    "target_language": end_lang if target_concept == start_clean else start_lang
                })
            
            except Exception as e:
                self.logger.debug(f"Error processing edge: {e}")
                continue
        
        # Remove duplicates while preserving order (first occurrence wins)
        seen = set()
        unique_related = []
        for concept in related_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen:
                seen.add(concept_lower)
                unique_related.append(concept)
        
        # Filter and sort by quality
        filtered_related = []
        for rel_concept in unique_related:
            # Additional quality filters
            if (len(rel_concept) > 2 and 
                not rel_concept.isdigit() and
                not rel_concept.startswith(('http', 'www')) and
                ' ' not in rel_concept or len(rel_concept.split()) <= 3):  # Avoid very long phrases
                filtered_related.append(rel_concept)
        
        return {
            "related": filtered_related[:10],  # Limit results
            "edges": edges[:self.max_relations],  # Limit edges
            "total_edges_processed": len(edges_data),
            "edges_after_filtering": len(edges)
        }
    
    def _clean_concept_label(self, label: str) -> str:
        """Clean and normalize concept labels from ConceptNet."""
        if not label:
            return ""
        
        # The label field in the API response is already clean,
        # but we may need to handle some edge cases
        clean_label = label.strip()
        
        # Handle multi-word concepts with hyphens or underscores
        if "_" in clean_label or "-" in clean_label:
            # Keep meaningful compound words, but clean up technical formatting
            if not any(char.isspace() for char in clean_label):
                # Replace underscores/hyphens with spaces for single compound words
                clean_label = clean_label.replace("_", " ").replace("-", " ")
                # But preserve hyphenated words that should stay together
                clean_label = re.sub(r'\b(\w+)\s+(\w+)\b', 
                                   lambda m: f"{m.group(1)}-{m.group(2)}" 
                                   if len(m.group(1)) <= 4 or len(m.group(2)) <= 4 
                                   else f"{m.group(1)} {m.group(2)}", 
                                   clean_label)
        
        # Remove leading articles (but be careful with proper nouns)
        if clean_label.lower().startswith(("a ", "an ", "the ")) and not clean_label[0].isupper():
            parts = clean_label.split(" ", 1)
            if len(parts) > 1:
                clean_label = parts[1]
        
        # Capitalize properly for consistency
        if clean_label and not clean_label[0].isupper():
            # Only capitalize if it's not a proper noun or already capitalized
            clean_label = clean_label.lower()
        
        # Clean up extra spaces
        clean_label = re.sub(r'\s+', ' ', clean_label).strip()
        
        return clean_label
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current ConceptNet rate limiting status."""
        return self.rate_limiter.get_status()
    
    def reset_rate_limiter(self):
        """Reset the rate limiter (useful for testing)."""
        with ConceptNetExpansionPlugin._rate_limiter_lock:
            ConceptNetExpansionPlugin._rate_limiter = ConceptNetRateLimiter()
        self.rate_limiter = ConceptNetExpansionPlugin._rate_limiter