"""
Comprehensive SpaCy provider for NER, linguistic analysis, and temporal processing.
Implements the BaseProvider interface using SpaCy for full NLP capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

from src.providers.nlp.base_provider import (
    BaseProvider, ProviderMetadata, ExpansionRequest, ProviderError, ProviderNotAvailableError
)
from src.shared.plugin_contracts import (
    PluginResult, CacheKey, CacheType, ConfidenceScore, PluginMetadata, PluginType,
    create_field_expansion_result
)

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available - install with: pip install spacy")


class SpacyProvider(BaseProvider):
    """
    Comprehensive SpaCy provider for NLP tasks.
    
    Capabilities:
    - Named Entity Recognition (all entity types)
    - Temporal concept extraction
    - Linguistic analysis (POS, dependencies, lemmatization)
    - Noun chunk extraction
    - Similarity computation
    - Pattern matching
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        super().__init__()
        self.model_name = model_name
        self.nlp = None
        self._spacy_initialized = False
        
        # Entity type mapping for structured output
        self.entity_type_mapping = {
            "PERSON": "people",
            "ORG": "organizations", 
            "GPE": "locations",
            "LOC": "locations",
            "FAC": "locations",
            "WORK_OF_ART": "works_of_art",
            "EVENT": "events",
            "PRODUCT": "products",
            "DATE": "dates",
            "TIME": "dates",
            "MONEY": "money",
            "QUANTITY": "quantities",
            "PERCENT": "quantities",
            "LANGUAGE": "languages",
            "LAW": "other",
            "NORP": "other"
        }
        
    @property
    def metadata(self) -> ProviderMetadata:
        """Get SpaCy provider metadata."""
        return ProviderMetadata(
            name="SpacyProvider",
            provider_type="nlp",
            context_aware=True,
            strengths=[
                "comprehensive named entity recognition",
                "linguistic analysis capabilities",
                "temporal concept extraction",
                "Python 3.12 compatible",
                "reliable and fast processing",
                "multiple model sizes available",
                "multilingual support"
            ],
            weaknesses=[
                "requires model download",
                "memory usage scales with model size",
                "primarily English-focused in practice",
                "limited domain-specific knowledge"
            ],
            best_for=[
                "entity extraction from descriptions",
                "people and organization identification",
                "location and work references",
                "temporal information extraction",
                "linguistic feature analysis",
                "text preprocessing"
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize the SpaCy provider."""
        if not SPACY_AVAILABLE:
            logger.error("SpaCy not available - cannot initialize")
            return False
            
        try:
            if not self._spacy_initialized:
                logger.info(f"Initializing SpaCy with model: {self.model_name}")
                
                # Load spaCy model
                self.nlp = spacy.load(self.model_name)
                
                # Test the model
                test_doc = self.nlp("Test text with John Doe from Microsoft in 2024.")
                self._spacy_initialized = True
                
                logger.info(f"SpaCy initialized successfully with model: {self.model_name}")
                logger.info(f"Available pipeline components: {self.nlp.pipe_names}")
                
            return True
        except OSError as e:
            logger.error(f"SpaCy model '{self.model_name}' not found: {e}")
            logger.error(f"Install with: python -m spacy download {self.model_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SpaCy provider: {e}")
            return False
    
    async def expand_concept(self, request: ExpansionRequest) -> Optional[PluginResult]:
        """
        Expand concept using comprehensive SpaCy analysis.
        
        Args:
            request: Expansion request
            
        Returns:
            PluginResult with NLP analysis results
        """
        start_time = datetime.now()
        
        try:
            if not await self._ensure_initialized():
                raise ProviderNotAvailableError("SpaCy provider not available", "SpacyProvider")
            
            if not self.nlp:
                logger.error("SpaCy model not loaded")
                return None
            
            text = request.concept.strip()
            extraction_type = request.options.get("extraction_type", "entities")
            
            # Perform the requested type of extraction
            if extraction_type == "entities":
                result_data = await self._extract_entities(text, request.options)
            elif extraction_type == "temporal":
                result_data = await self._extract_temporal(text, request.options)
            elif extraction_type == "linguistic":
                result_data = await self._extract_linguistic(text, request.options)
            elif extraction_type == "similarity":
                result_data = await self._compute_similarity(text, request.options)
            else:
                # Default to entity extraction
                result_data = await self._extract_entities(text, request.options)
            
            if not result_data:
                logger.warning(f"No results found for: {text}")
                return None
            
            # Calculate execution time
            total_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create confidence scores
            confidence_scores = self._calculate_confidence_scores(result_data)
            
            return create_field_expansion_result(
                field_name=request.field_name,
                input_value=request.concept,
                expansion_result=result_data,
                confidence_scores=confidence_scores,
                plugin_name="SpacyProvider",
                plugin_version="1.0.0",
                cache_type=CacheType.SPACY_NER,
                execution_time_ms=total_time_ms,
                media_context=request.media_context,
                plugin_type=PluginType.CONCEPT_EXPANSION,
                api_endpoint="spacy:nlp",
                model_used=f"spacy-{self.model_name}"
            )
            
        except Exception as e:
            logger.error(f"SpaCy analysis failed: {e}")
            return None
    
    async def _extract_entities(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        
        # Group entities by type
        entities_by_type = {}
        structured_entities = {category: [] for category in set(self.entity_type_mapping.values())}
        
        entity_types_filter = options.get("entity_types", "ALL")
        confidence_threshold = options.get("confidence_threshold", 0.0)
        include_lemmas = options.get("include_lemmas", True)
        
        for ent in doc.ents:
            # Filter by entity type if specified
            if entity_types_filter != "ALL" and ent.label_ not in entity_types_filter:
                continue
            
            # Basic confidence based on entity length and capitalization
            confidence = self._calculate_entity_confidence(ent)
            
            if confidence < confidence_threshold:
                continue
            
            entity_data = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": confidence
            }
            
            if include_lemmas:
                entity_data["lemma"] = ent.lemma_
            
            # Add to type-specific list
            if ent.label_ not in entities_by_type:
                entities_by_type[ent.label_] = []
            entities_by_type[ent.label_].append(entity_data)
            
            # Add to structured categories
            category = self.entity_type_mapping.get(ent.label_, "other")
            if category not in structured_entities:
                structured_entities[category] = []
            structured_entities[category].append(entity_data)
        
        # Remove empty categories
        structured_entities = {k: v for k, v in structured_entities.items() if v}
        
        # Extract noun chunks if requested
        noun_chunks = []
        if options.get("extract_noun_chunks", False):
            for chunk in doc.noun_chunks:
                noun_chunks.append({
                    "text": chunk.text,
                    "root": chunk.root.text,
                    "start": chunk.start_char,
                    "end": chunk.end_char
                })
        
        return {
            "entities_by_type": entities_by_type,
            "structured_entities": structured_entities,
            "noun_chunks": noun_chunks,
            "total_entities": sum(len(ents) for ents in entities_by_type.values()),
            "extraction_type": "entities",
            "model": self.model_name
        }
    
    async def _extract_temporal(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal information from text."""
        doc = self.nlp(text)
        
        temporal_entities = []
        temporal_concepts = []
        
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME', 'ORDINAL', 'CARDINAL']:
                entity_text = ent.text.lower()
                confidence = self._calculate_entity_confidence(ent)
                
                temporal_entities.append({
                    "text": ent.text,
                    "normalized": entity_text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": confidence
                })
                
                # Add semantic temporal concepts
                if ent.label_ == 'DATE':
                    temporal_concepts.extend(self._generate_temporal_concepts(entity_text))
        
        return {
            "temporal_entities": temporal_entities,
            "temporal_concepts": temporal_concepts,
            "extraction_type": "temporal",
            "model": self.model_name
        }
    
    async def _extract_linguistic(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract linguistic features from text."""
        doc = self.nlp(text)
        
        # POS tags
        pos_tags = []
        if options.get("extract_pos", True):
            pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
        
        # Dependencies
        dependencies = []
        if options.get("extract_dependencies", True):
            for token in doc:
                dependencies.append({
                    "text": token.text,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "children": [child.text for child in token.children]
                })
        
        # Lemmas
        lemmas = []
        if options.get("extract_lemmas", True):
            lemmas = [(token.text, token.lemma_) for token in doc if token.lemma_ != token.text]
        
        # Sentences
        sentences = [sent.text for sent in doc.sents] if options.get("extract_sentences", True) else []
        
        return {
            "pos_tags": pos_tags,
            "dependencies": dependencies,
            "lemmas": lemmas,
            "sentences": sentences,
            "token_count": len(doc),
            "sentence_count": len(sentences),
            "extraction_type": "linguistic",
            "model": self.model_name
        }
    
    async def _compute_similarity(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Compute similarity metrics."""
        doc = self.nlp(text)
        
        # Compare with reference text if provided
        reference_text = options.get("reference_text")
        similarity_score = None
        
        if reference_text and doc.has_vector:
            ref_doc = self.nlp(reference_text)
            if ref_doc.has_vector:
                similarity_score = doc.similarity(ref_doc)
        
        return {
            "has_vector": doc.has_vector,
            "similarity_score": similarity_score,
            "vector_norm": float(doc.vector_norm) if doc.has_vector else None,
            "extraction_type": "similarity",
            "model": self.model_name
        }
    
    def _calculate_entity_confidence(self, entity) -> float:
        """Calculate confidence score for an entity."""
        base_confidence = 0.7
        
        # Boost confidence for capitalized entities
        if entity.text[0].isupper():
            base_confidence += 0.1
        
        # Boost confidence for longer entities
        if len(entity.text) > 5:
            base_confidence += 0.1
        
        # Boost confidence for proper nouns
        if any(token.pos_ == "PROPN" for token in entity):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_temporal_concepts(self, entity_text: str) -> List[str]:
        """Generate semantic temporal concepts."""
        concepts = []
        
        if any(word in entity_text for word in ['week', 'weekly']):
            concepts.append("weekly")
        if any(word in entity_text for word in ['month', 'monthly']):
            concepts.append("monthly")
        if any(word in entity_text for word in ['year', 'yearly', 'annual']):
            concepts.append("yearly")
        if any(word in entity_text for word in ['day', 'daily']):
            concepts.append("daily")
        if any(word in entity_text for word in ['recent', 'new', 'latest']):
            concepts.append("recent")
        if any(word in entity_text for word in ['old', 'classic', 'vintage']):
            concepts.append("classic")
        
        return concepts
    
    def _calculate_confidence_scores(self, result_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for results."""
        scores = {}
        
        # Base confidence based on extraction type and results
        extraction_type = result_data.get("extraction_type", "unknown")
        
        if extraction_type == "entities":
            total_entities = result_data.get("total_entities", 0)
            base_score = min(0.9, 0.5 + (total_entities * 0.05))
            scores["entity_extraction"] = base_score
            
        elif extraction_type == "temporal":
            temporal_count = len(result_data.get("temporal_entities", []))
            base_score = min(0.9, 0.6 + (temporal_count * 0.1))
            scores["temporal_extraction"] = base_score
            
        elif extraction_type == "linguistic":
            token_count = result_data.get("token_count", 0)
            base_score = min(0.9, 0.7 + min(0.2, token_count * 0.01))
            scores["linguistic_analysis"] = base_score
            
        else:
            scores["general"] = 0.7
        
        return scores
    
    async def health_check(self) -> Dict[str, Any]:
        """Check SpaCy provider health."""
        try:
            if not self._spacy_initialized:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyProvider",
                    "error": "SpaCy not initialized"
                }
            
            if not self.nlp:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyProvider", 
                    "error": "SpaCy model not loaded"
                }
            
            # Test entity extraction
            try:
                test_doc = self.nlp("John Smith works at Apple Inc. in California since 2020.")
                entities = [(ent.text, ent.label_) for ent in test_doc.ents]
                
                return {
                    "status": "healthy",
                    "provider": "SpacyProvider",
                    "model": self.model_name,
                    "pipeline_components": self.nlp.pipe_names,
                    "test_entities": entities,
                    "has_vectors": self.nlp.meta.get("vectors", {}).get("keys", 0) > 0
                }
            except Exception as test_error:
                return {
                    "status": "unhealthy",
                    "provider": "SpacyProvider",
                    "error": f"Test failed: {test_error}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "provider": "SpacyProvider",
                "error": str(e)
            }
    
    def can_handle_concept(self, concept: str, media_context: str) -> bool:
        """Check if SpaCy can handle this concept."""
        if not concept or len(concept.strip()) == 0:
            return False
        
        # SpaCy can handle most text for entity extraction
        return len(concept.strip()) >= 3
    
    def get_recommended_parameters(self, concept: str, media_context: str) -> Dict[str, Any]:
        """Get recommended parameters for SpaCy analysis."""
        return {
            "extraction_type": "entities",
            "entity_types": "ALL",
            "confidence_threshold": 0.7,
            "include_lemmas": True,
            "extract_noun_chunks": True,
            "model": self.model_name
        }
    
    async def close(self) -> None:
        """Clean up SpaCy provider resources."""
        if self.nlp:
            self.nlp = None
            self._spacy_initialized = False
            logger.info("SpaCy provider closed")