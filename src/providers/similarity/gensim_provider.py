"""
Gensim provider for statistical concept expansion via vectorization.
Implements the BaseProvider interface using Gensim for similarity-based expansion.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import numpy as np

from src.providers.nlp.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)


from gensim.models import Word2Vec, KeyedVectors
from gensim.models.keyedvectors import KeyedVectors as KV
from gensim.downloader import load as gensim_load


class GensimProvider(BaseProvider):
    """
    Gensim provider for statistical concept expansion via vectorization.
    
    Uses pre-trained word embeddings to find semantically similar concepts
    based on vector space similarity rather than explicit relationships.
    """
    
    def __init__(self, model_name: str = "word2vec-google-news-300"):
        super().__init__()
        self.model_name = model_name
        self.model: Optional[KeyedVectors] = None
        self._model_loaded = False
        self._available_models = {
            "word2vec-google-news-300": "Google News Word2Vec (300d)",
            "glove-wiki-gigaword-300": "GloVe Wiki Gigaword (300d)", 
            "fasttext-wiki-news-subwords-300": "FastText Wiki News (300d)"
        }
    
    @property
    def metadata(self) -> ProviderMetadata:
        """Get Gensim provider metadata."""
        return ProviderMetadata(
            name="Gensim",
            provider_type="statistical", 
            context_aware=False,
            strengths=[
                "corpus similarity",
                "statistical patterns",
                "fast lookup once loaded",
                "vector space semantics",
                "synonym discovery"
            ],
            weaknesses=[
                "limited to training data",
                "no semantic understanding",
                "large model downloads",
                "context-blind",
                "may lack domain-specific terms"
            ],
            best_for=[
                "text similarity",
                "synonym expansion",
                "document similarity",
                "lexical relationships",
                "corpus-based patterns"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the Gensim provider by loading the word embedding model."""
        
        try:
            if not self._model_loaded:
                logger.info(f"Loading Gensim model: {self.model_name}")
                logger.info("This may take several minutes for large models...")
                

                self.model = gensim_load(self.model_name)
                self._model_loaded = True
                
                vocab_size = len(self.model.key_to_index) if self.model else 0
                logger.info(f"Gensim model loaded successfully. Vocabulary size: {vocab_size}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gensim provider: {e}")
            logger.error("Consider using a different model or installing gensim-data")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using Gensim word embeddings.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with Gensim similarity expansions
        """
        start_time = datetime.now()
        
        try:

            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("Gensim provider not available", "Gensim")
            
            if not self.model:
                logger.error("Gensim model not loaded")
                return None
            

            concept = request.concept.lower().strip()
            

            similar_concepts = []
            confidence_scores = {}
            
            if " " in concept:

                words = concept.split()
                all_similarities = []
                
                for word in words:
                    word_similarities = self._get_word_similarities(word, request.max_concepts)
                    all_similarities.extend(word_similarities)
                
                
                unique_similarities = {}
                for word, similarity in all_similarities:
                    if word not in unique_similarities:
                        unique_similarities[word] = similarity
                    else:

                        unique_similarities[word] = max(unique_similarities[word], similarity)
                

                sorted_similarities = sorted(
                    unique_similarities.items(),
                    key=lambda x: x[1], 
                    reverse=True
                )[:request.max_concepts]
                
                similar_concepts = [word for word, _ in sorted_similarities]
                confidence_scores = {word: float(sim) for word, sim in sorted_similarities}
                
            else:

                word_similarities = self._get_word_similarities(concept, request.max_concepts)
                similar_concepts = [word for word, _ in word_similarities]
                confidence_scores = {word: float(sim) for word, sim in word_similarities}
            
            if not similar_concepts:
                logger.warning(f"No similar concepts found for: {concept}")
                return None
            

            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result={
                    "expanded_concepts": similar_concepts,
                    "original_concept": request.concept,
                    "expansion_method": "gensim",
                    "model_name": self.model_name,
                    "vocabulary_size": len(self.model.key_to_index),
                    "provider_type": "statistical"
                },
                confidence_scores=confidence_scores,
                plugin_name="GensimProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.GENSIM_SIMILARITY,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint=f"gensim:{self.model_name}",
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Gensim expansion failed: {e}")
            return None
    
    def _get_word_similarities(self, word: str, max_results: int) -> List[tuple]:
        """
        Get similar words for a single word using Gensim.
        
        Args:
            word: Word to find similarities for
            max_results: Maximum number of results
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if not self.model:
            return []
        
        try:
            
            if word not in self.model.key_to_index:
                logger.debug(f"Word '{word}' not in vocabulary")
                return []
            
            
            similarities = self.model.most_similar(
                word, 
                topn=max_results * 2  
            )
            

            filtered_similarities = []
            seen_words = set()
            
            for similar_word, similarity in similarities:

                clean_word = similar_word.lower().strip()
                

                if clean_word in seen_words or clean_word == word:
                    continue
                

                if "_" in clean_word or any(char.isdigit() for char in clean_word):
                    continue
                

                if len(clean_word) < 2:
                    continue
                
                filtered_similarities.append((clean_word, similarity))
                seen_words.add(clean_word)
                
                if len(filtered_similarities) >= max_results:
                    break
            
            return filtered_similarities
            
        except Exception as e:
            logger.error(f"Error getting similarities for '{word}': {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Gensim provider health."""
        try:
            
            if not self._model_loaded:
                return {
                    "status": "unhealthy",
                    "provider": "Gensim",
                    "error": "Model not loaded"
                }
            
            if not self.model:
                return {
                    "status": "unhealthy",
                    "provider": "Gensim",
                    "error": "Model is None"
                }
            

            try:
                test_word = "test"
                if test_word in self.model.key_to_index:
                    similarities = self.model.most_similar(test_word, topn=1)
                    success = len(similarities) > 0
                else:

                    common_words = ["movie", "film", "good", "action", "the"]
                    success = False
                    for word in common_words:
                        if word in self.model.key_to_index:
                            similarities = self.model.most_similar(word, topn=1)
                            success = len(similarities) > 0
                            break
                
                return {
                    "status": "healthy" if success else "degraded",
                    "provider": "Gensim",
                    "model_name": self.model_name,
                    "vocabulary_size": len(self.model.key_to_index),
                    "test_successful": success
                }
                
            except Exception as e:
                return {
                    "status": "degraded",
                    "provider": "Gensim",
                    "model_name": self.model_name,
                    "error": f"Similarity test failed: {e}"
                }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "Gensim",
                "error": str(e)
            }
    
    def supports_concept(self, concept: str, media_context: str) -> bool:
        """
        Check if Gensim can handle this concept.
        
        Gensim works best with:
        - Words in the training vocabulary
        - Common English words
        - Single words or simple phrases
        
        Less effective with:
        - Very specific technical terms
        - Proper nouns not in training data
        - Very new slang or terminology
        """

            


        if not self.model:

            words = concept.lower().split()

            for word in words:

                if len(word) < 2 or word.isdigit() or not word.isalpha():
                    continue

                return True
            return False
        

        words = concept.lower().split()
        

        for word in words:
            if word in self.model.key_to_index:
                return True
        
        return False
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for Gensim expansion."""
        params = {
            "max_concepts": 10
        }
        

        if " " in concept:

            params["max_concepts"] = 15
        

        if len(concept.split()) == 1:
            params["max_concepts"] = 20
        
        return params
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available Gensim models."""
        return self._available_models
    
    async def close(self) -> None:
        """Clean up Gensim provider resources."""
        if self.model:

            self.model = None
            self._model_loaded = False
            logger.info("Gensim model unloaded")