"""
Question Expansion Plugin - Handles natural language questions about media.

Expands questions like "Suggest 5 sci-fi movies" or "What are the censorship 
issues in Goodfellas?" using LLM understanding with queue integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import re
from enum import Enum

from src.plugins.base_concept import BaseConceptPlugin, ProcessingStrategy
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.providers.llm.llm_provider import LLMProvider
from src.providers.knowledge.conceptnet_provider import ConceptNetProvider
from src.providers.nlp.base_provider import ExpansionRequest
from src.shared.plugin_contracts import PluginResult, CacheType
from src.storage.cache_manager import CacheStrategy

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions we can handle."""
    RECOMMENDATION = "recommendation"      # "Suggest X movies"
    INFORMATION = "information"           # "Tell me about X"
    COMPARISON = "comparison"             # "Compare X and Y"
    FACTUAL = "factual"                  # "What year was X released?"
    ANALYTICAL = "analytical"             # "Why is X considered good?"
    SEARCH = "search"                     # "Find movies with X"


class QuestionExpansionPlugin(BaseConceptPlugin):
    """
    Plugin that handles natural language questions about media content.
    
    Features:
    - Question intent classification
    - Query expansion for recommendations
    - Information extraction from questions
    - Context-aware answer structuring
    - Queue integration for complex questions
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="QuestionExpansionPlugin",
            version="2.0.0",
            description="Handles natural language questions with LLM understanding",
            author="System",
            plugin_type=PluginType.QUERY_EMBELLISHER,
            tags=["question", "nlp", "understanding", "recommendation"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=300.0,
            preferred_memory_mb=800.0,
            requires_gpu=False,  # Benefits from Ollama GPU
            max_execution_time_seconds=25.0,
            can_use_distributed_resources=True
        )
    
    async def _initialize_providers(self) -> None:
        """Initialize question handling providers."""
        # Primary: LLM for understanding
        self.providers["llm"] = LLMProvider()
        await self.providers["llm"].initialize()
        
        # Secondary: ConceptNet for keyword expansion
        self.providers["conceptnet"] = ConceptNetProvider()
        await self.providers["conceptnet"].initialize()
        
        self._logger.info("Initialized question handling providers")
    
    async def embellish_query(
        self,
        query: str,
        context: PluginExecutionContext
    ) -> str:
        """
        Transform natural language questions into enhanced search queries.
        """
        start_time = datetime.now()
        
        try:
            # Detect if this is a question
            if not self._is_question(query):
                # Not a question - pass through
                return query
            
            # Select processing strategy
            strategy = self._select_processing_strategy(context)
            
            # Classify question type
            question_type = await self._classify_question(query, strategy)
            
            self._logger.info(
                f"Processing {question_type.value} question with {strategy.value} strategy"
            )
            
            # Process based on question type and strategy
            if question_type == QuestionType.RECOMMENDATION:
                enhanced = await self._handle_recommendation_question(query, strategy, context)
            elif question_type == QuestionType.SEARCH:
                enhanced = await self._handle_search_question(query, strategy, context)
            elif question_type == QuestionType.INFORMATION:
                enhanced = await self._handle_information_question(query, strategy, context)
            else:
                # Other types need different handling
                enhanced = await self._handle_generic_question(query, strategy, context)
            
            self._logger.info(
                f"Question expansion complete in "
                f"{(datetime.now() - start_time).total_seconds() * 1000:.0f}ms"
            )
            
            return enhanced
            
        except Exception as e:
            self._logger.error(f"Question expansion failed: {e}")
            return query  # Return original on error
    
    def _is_question(self, text: str) -> bool:
        """Detect if the text is a question."""
        # Check for question marks
        if "?" in text:
            return True
        
        # Check for question words
        question_starters = [
            "what", "who", "where", "when", "why", "how",
            "can", "could", "would", "should", "is", "are",
            "suggest", "recommend", "find", "show", "tell",
            "give me", "list", "name"
        ]
        
        first_word = text.lower().split()[0] if text else ""
        return any(text.lower().startswith(starter) for starter in question_starters)
    
    async def _classify_question(
        self,
        query: str,
        strategy: ProcessingStrategy
    ) -> QuestionType:
        """Classify the type of question."""
        query_lower = query.lower()
        
        # Pattern-based classification (fast)
        if any(word in query_lower for word in ["suggest", "recommend", "give me", "list"]):
            return QuestionType.RECOMMENDATION
        elif any(word in query_lower for word in ["find", "search", "show", "with", "featuring"]):
            return QuestionType.SEARCH
        elif any(word in query_lower for word in ["tell me about", "what is", "who is", "explain"]):
            return QuestionType.INFORMATION
        elif any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return QuestionType.COMPARISON
        elif any(word in query_lower for word in ["when", "year", "date", "how many", "how much"]):
            return QuestionType.FACTUAL
        elif any(word in query_lower for word in ["why", "how come", "reason"]):
            return QuestionType.ANALYTICAL
        
        # LLM classification for ambiguous cases (if high resource)
        if strategy == ProcessingStrategy.HIGH_RESOURCE and self.providers.get("llm"):
            # Queue LLM classification
            # For now, default to SEARCH
            pass
        
        return QuestionType.SEARCH
    
    async def _handle_recommendation_question(
        self,
        query: str,
        strategy: ProcessingStrategy,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle recommendation questions like "Suggest 5 sci-fi movies".
        """
        # Extract key parameters
        params = self._extract_recommendation_params(query)
        
        if strategy == ProcessingStrategy.HIGH_RESOURCE:
            # Queue LLM for deep understanding
            if self.queue_manager:
                task_data = {
                    "question": query,
                    "question_type": "recommendation",
                    "params": params
                }
                task_id = await self._queue_task(
                    "question_understanding",
                    task_data,
                    ExecutionPriority.HIGH
                )
                self._logger.debug(f"Queued recommendation analysis: {task_id}")
            
            # Parallel concept expansion for immediate response
            expanded_concepts = await self._expand_recommendation_concepts(params)
            
            # Build enhanced query
            enhanced_parts = []
            if params["genre"]:
                enhanced_parts.extend(expanded_concepts.get(params["genre"], [params["genre"]]))
            if params["count"]:
                enhanced_parts.append(f"top {params['count']}")
            if params["modifiers"]:
                enhanced_parts.extend(params["modifiers"])
            
            return " ".join(enhanced_parts)
            
        else:
            # Simple extraction and expansion
            genre = params.get("genre", "")
            if genre and "conceptnet" in self.providers:
                try:
                    result = await self._expand_with_provider("conceptnet", genre, "movie")
                    if result and result.success:
                        expansions = result.enhanced_data.get("expanded_concepts", [])[:3]
                        return f"{genre} {' '.join(expansions)}"
                except Exception:
                    pass
            
            return query
    
    async def _handle_search_question(
        self,
        query: str,
        strategy: ProcessingStrategy,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle search questions like "Find movies with robots".
        """
        # Extract search terms
        search_terms = self._extract_search_terms(query)
        
        if not search_terms:
            return query
        
        expanded_terms = []
        
        if strategy in [ProcessingStrategy.HIGH_RESOURCE, ProcessingStrategy.MEDIUM_RESOURCE]:
            # Expand each search term
            for term in search_terms[:5]:  # Limit
                if "conceptnet" in self.providers:
                    try:
                        result = await self._expand_with_provider("conceptnet", term, "movie")
                        if result and result.success:
                            expansions = result.enhanced_data.get("expanded_concepts", [])[:2]
                            expanded_terms.append(term)
                            expanded_terms.extend(expansions)
                        else:
                            expanded_terms.append(term)
                    except Exception:
                        expanded_terms.append(term)
                else:
                    expanded_terms.append(term)
        else:
            expanded_terms = search_terms
        
        # Remove duplicates while preserving order
        unique_terms = list(dict.fromkeys(expanded_terms))
        return " ".join(unique_terms)
    
    async def _handle_information_question(
        self,
        query: str,
        strategy: ProcessingStrategy,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle information questions like "Tell me about censorship in Goodfellas".
        """
        # Extract subject and topic
        subject, topic = self._extract_information_components(query)
        
        if strategy == ProcessingStrategy.HIGH_RESOURCE and self.queue_manager:
            # Queue for detailed analysis
            task_data = {
                "question": query,
                "subject": subject,
                "topic": topic,
                "question_type": "information"
            }
            task_id = await self._queue_task(
                "information_extraction",
                task_data,
                ExecutionPriority.HIGH
            )
            self._logger.debug(f"Queued information extraction: {task_id}")
        
        # Build search query from components
        search_parts = []
        if subject:
            search_parts.append(subject)
        if topic:
            search_parts.append(topic)
            # Expand topic if possible
            if "conceptnet" in self.providers:
                try:
                    result = await self._expand_with_provider("conceptnet", topic, "movie")
                    if result and result.success:
                        expansions = result.enhanced_data.get("expanded_concepts", [])[:2]
                        search_parts.extend(expansions)
                except Exception:
                    pass
        
        return " ".join(search_parts) if search_parts else query
    
    async def _handle_generic_question(
        self,
        query: str,
        strategy: ProcessingStrategy,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle other question types with basic keyword extraction.
        """
        # Extract key concepts
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return query
        
        # Basic expansion for top keywords
        expanded = []
        for keyword in keywords[:3]:
            expanded.append(keyword)
            if "conceptnet" in self.providers and strategy != ProcessingStrategy.LOW_RESOURCE:
                try:
                    result = await self._expand_with_provider("conceptnet", keyword, "movie")
                    if result and result.success:
                        expansions = result.enhanced_data.get("expanded_concepts", [])[:1]
                        expanded.extend(expansions)
                except Exception:
                    pass
        
        return " ".join(expanded)
    
    def _extract_recommendation_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters from recommendation questions."""
        params = {
            "count": None,
            "genre": None,
            "modifiers": []
        }
        
        # Extract count
        count_match = re.search(r'\b(\d+)\s+(movie|film)', query.lower())
        if count_match:
            params["count"] = int(count_match.group(1))
        
        # Extract genre
        genres = [
            "action", "comedy", "drama", "horror", "sci-fi", "science fiction",
            "romance", "thriller", "documentary", "animation", "fantasy"
        ]
        for genre in genres:
            if genre in query.lower():
                params["genre"] = genre
                break
        
        # Extract modifiers
        modifiers = ["recent", "classic", "popular", "underrated", "new", "old"]
        for modifier in modifiers:
            if modifier in query.lower():
                params["modifiers"].append(modifier)
        
        return params
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from search questions."""
        # Remove question words
        clean_query = query.lower()
        for word in ["find", "search", "show", "movies", "films", "with", "featuring", "about"]:
            clean_query = clean_query.replace(word, " ")
        
        # Extract remaining meaningful words
        words = clean_query.split()
        terms = [w for w in words if len(w) > 2 and w not in ["the", "and", "for"]]
        
        return terms
    
    def _extract_information_components(self, query: str) -> Tuple[str, str]:
        """Extract subject and topic from information questions."""
        # Pattern: "tell me about TOPIC in SUBJECT"
        match = re.search(r'about\s+(.+?)\s+in\s+(.+)', query.lower())
        if match:
            return match.group(2).strip(), match.group(1).strip()
        
        # Pattern: "what is TOPIC"
        match = re.search(r'what\s+is\s+(.+)', query.lower())
        if match:
            return "", match.group(1).strip("?")
        
        # Pattern: "SUBJECT information/details"
        match = re.search(r'(.+?)\s+(information|details|info)', query.lower())
        if match:
            return match.group(1).strip(), "information"
        
        return "", ""
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from generic questions."""
        # Remove common question words
        stopwords = {
            "what", "who", "where", "when", "why", "how", "is", "are",
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "about"
        }
        
        words = query.lower().split()
        keywords = [w for w in words if len(w) > 2 and w not in stopwords]
        
        return keywords
    
    async def _expand_recommendation_concepts(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Expand concepts from recommendation parameters."""
        expanded = {}
        
        # Expand genre if present
        if params["genre"] and "conceptnet" in self.providers:
            try:
                result = await self._expand_with_provider(
                    "conceptnet",
                    params["genre"],
                    "movie"
                )
                if result and result.success:
                    expanded[params["genre"]] = result.enhanced_data.get(
                        "expanded_concepts", []
                    )[:5]
            except Exception:
                pass
        
        # Expand modifiers
        for modifier in params.get("modifiers", []):
            if "conceptnet" in self.providers:
                try:
                    result = await self._expand_with_provider(
                        "conceptnet",
                        modifier,
                        "movie"
                    )
                    if result and result.success:
                        expanded[modifier] = result.enhanced_data.get(
                            "expanded_concepts", []
                        )[:3]
                except Exception:
                    pass
        
        return expanded
    
    async def _expand_with_provider(
        self,
        provider_name: str,
        concept: str,
        media_context: str
    ) -> Optional[PluginResult]:
        """Expand concept using specified provider."""
        provider = self.providers.get(provider_name)
        if not provider:
            return None
        
        try:
            request = ExpansionRequest(
                concept=concept,
                media_context=media_context,
                max_concepts=5
            )
            
            # Use cache
            cache_key = self.cache_manager.generate_cache_key(
                cache_type=CacheType.CONCEPTNET if provider_name == "conceptnet" else CacheType.LLM_CONCEPT,
                field_name="question_concept",
                input_value=concept,
                media_context=media_context
            )
            
            result = await self.cache_manager.get_or_compute(
                cache_key=cache_key,
                compute_func=lambda: provider.expand_concept(request),
                strategy=CacheStrategy.CACHE_FIRST
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Provider '{provider_name}' failed: {e}")
            return None