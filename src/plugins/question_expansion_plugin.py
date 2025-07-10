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

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import (
    PluginMetadata, PluginResourceRequirements, PluginExecutionContext,
    PluginExecutionResult, PluginType, ExecutionPriority
)
from src.shared.plugin_contracts import PluginResult

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions we can handle."""
    RECOMMENDATION = "recommendation"    
    INFORMATION = "information"         
    COMPARISON = "comparison"           
    FACTUAL = "factual"                
    ANALYTICAL = "analytical"           
    SEARCH = "search"                   


class QuestionExpansionPlugin(HTTPBasePlugin):
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
            requires_gpu=False,
            max_execution_time_seconds=25.0,
            can_use_distributed_resources=True
        )
    
    
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

            if not self._is_question(query):

                return query
            

            question_type = await self._classify_question(query)
            
            self._logger.info(f"Processing {question_type.value} question")
            

            if question_type == QuestionType.RECOMMENDATION:
                enhanced = await self._handle_recommendation_question(query, context)
            elif question_type == QuestionType.SEARCH:
                enhanced = await self._handle_search_question(query, context)
            elif question_type == QuestionType.INFORMATION:
                enhanced = await self._handle_information_question(query, context)
            else:

                enhanced = await self._handle_generic_question(query, context)
            
            self._logger.info(
                f"Question expansion complete in "
                f"{(datetime.now() - start_time).total_seconds() * 1000:.0f}ms"
            )
            
            return enhanced
            
        except Exception as e:
            self._logger.error(f"Question expansion failed: {e}")
            return query
    
    def _is_question(self, text: str) -> bool:
        """Detect if the text is a question."""
        
        if "?" in text:
            return True
        
        
        question_starters = [
            "what", "who", "where", "when", "why", "how",
            "can", "could", "would", "should", "is", "are",
            "suggest", "recommend", "find", "show", "tell",
            "give me", "list", "name"
        ]
        
        first_word = text.lower().split()[0] if text else ""
        return any(text.lower().startswith(starter) for starter in question_starters)
    
    async def _classify_question(self, query: str) -> QuestionType:
        """Classify the type of question."""
        query_lower = query.lower()
        

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
        

        
        return QuestionType.SEARCH
    
    async def _handle_recommendation_question(
        self,
        query: str,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle recommendation questions like "Suggest 5 sci-fi movies".
        """

        params = self._extract_recommendation_params(query)
        

        expanded_concepts = await self._expand_recommendation_concepts(params)
        
        
        enhanced_parts = []
        if params["genre"]:
            enhanced_parts.extend(expanded_concepts.get(params["genre"], [params["genre"]]))
        if params["count"]:
            enhanced_parts.append(f"top {params['count']}")
        if params["modifiers"]:
            enhanced_parts.extend(params["modifiers"])
        
        return " ".join(enhanced_parts) if enhanced_parts else query
    
    async def _handle_search_question(
        self,
        query: str,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle search questions like "Find movies with robots".
        """

        search_terms = self._extract_search_terms(query)
        
        if not search_terms:
            return query
        
        expanded_terms = []
        

        for term in search_terms[:5]:
            expanded_terms.append(term)
            try:
                result = await self._expand_with_conceptnet_service(term, "movie")
                if result:
                    expansions = result[:2]
                    expanded_terms.extend(expansions)
            except Exception as e:
                self._logger.debug(f"Failed to expand term '{term}': {e}")
        
        
        unique_terms = list(dict.fromkeys(expanded_terms))
        return " ".join(unique_terms)
    
    async def _handle_information_question(
        self,
        query: str,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle information questions like "Tell me about censorship in Goodfellas".
        """

        subject, topic = self._extract_information_components(query)
        
        
        search_parts = []
        if subject:
            search_parts.append(subject)
        if topic:
            search_parts.append(topic)

            try:
                expansions = await self._expand_with_conceptnet_service(topic, "movie")
                if expansions:
                    search_parts.extend(expansions[:2])
            except Exception as e:
                self._logger.debug(f"Failed to expand topic '{topic}': {e}")
        
        return " ".join(search_parts) if search_parts else query
    
    async def _handle_generic_question(
        self,
        query: str,
        context: PluginExecutionContext
    ) -> str:
        """
        Handle other question types with basic keyword extraction.
        """

        keywords = self._extract_keywords(query)
        
        if not keywords:
            return query
        

        expanded = []
        for keyword in keywords[:3]:
            expanded.append(keyword)
            try:
                expansions = await self._expand_with_conceptnet_service(keyword, "movie")
                if expansions:
                    expanded.extend(expansions[:1])
            except Exception as e:
                self._logger.debug(f"Failed to expand keyword '{keyword}': {e}")
        
        return " ".join(expanded)
    
    def _extract_recommendation_params(self, query: str) -> Dict[str, Any]:
        """Extract parameters from recommendation questions."""
        params = {
            "count": None,
            "genre": None,
            "modifiers": []
        }
        

        count_match = re.search(r'\b(\d+)\s+(movie|film)', query.lower())
        if count_match:
            params["count"] = int(count_match.group(1))
        

        genres = [
            "action", "comedy", "drama", "horror", "sci-fi", "science fiction",
            "romance", "thriller", "documentary", "animation", "fantasy"
        ]
        for genre in genres:
            if genre in query.lower():
                params["genre"] = genre
                break
        

        modifiers = ["recent", "classic", "popular", "underrated", "new", "old"]
        for modifier in modifiers:
            if modifier in query.lower():
                params["modifiers"].append(modifier)
        
        return params
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from search questions."""
        
        clean_query = query.lower()
        for word in ["find", "search", "show", "movies", "films", "with", "featuring", "about"]:
            clean_query = clean_query.replace(word, " ")
        

        words = clean_query.split()
        terms = [w for w in words if len(w) > 2 and w not in ["the", "and", "for"]]
        
        return terms
    
    def _extract_information_components(self, query: str) -> Tuple[str, str]:
        """Extract subject and topic from information questions."""

        match = re.search(r'about\s+(.+?)\s+in\s+(.+)', query.lower())
        if match:
            return match.group(2).strip(), match.group(1).strip()
        

        match = re.search(r'what\s+is\s+(.+)', query.lower())
        if match:
            return "", match.group(1).strip("?")
        

        match = re.search(r'(.+?)\s+(information|details|info)', query.lower())
        if match:
            return match.group(1).strip(), "information"
        
        return "", ""
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from generic questions."""
        
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
        

        if params["genre"]:
            try:
                expansions = await self._expand_with_conceptnet_service(params["genre"], "movie")
                if expansions:
                    expanded[params["genre"]] = expansions[:5]
            except Exception as e:
                self._logger.debug(f"Failed to expand genre '{params['genre']}': {e}")
        

        for modifier in params.get("modifiers", []):
            try:
                expansions = await self._expand_with_conceptnet_service(modifier, "movie")
                if expansions:
                    expanded[modifier] = expansions[:3]
            except Exception as e:
                self._logger.debug(f"Failed to expand modifier '{modifier}': {e}")
        
        return expanded
    
    async def _expand_with_conceptnet_service(
        self,
        concept: str,
        media_context: str
    ) -> Optional[List[str]]:
        """Expand concept using ConceptNet service via HTTP."""
        try:
            
            service_url = self.get_service_url("conceptnet", "expand")
            
            request_data = {
                "concept": concept,
                "media_context": media_context,
                "max_concepts": 5,
                "field_name": "question_concept",
                "options": {
                    "relation_types": ["RelatedTo", "IsA", "PartOf"],
                    "language": "en"
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            
            if response.get("success", False):
                result_data = response.get("result", {})
                expanded_concepts = result_data.get("expanded_concepts", [])
                return expanded_concepts if isinstance(expanded_concepts, list) else []
            else:
                self._logger.debug(f"ConceptNet service failed for '{concept}': {response.get('error', 'Unknown error')}")
                return []
                
        except Exception as e:
            self._logger.debug(f"ConceptNet service call failed for '{concept}': {e}")
            return []