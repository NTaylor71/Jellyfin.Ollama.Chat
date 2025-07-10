"""
SpaCy Linguistic Analysis Plugin
HTTP-only plugin that calls SpaCy service for comprehensive linguistic feature extraction.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class SpacyLinguisticPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that extracts linguistic features using SpaCy.
    
    Features:
    - Part-of-speech tagging and analysis
    - Dependency parsing and grammatical relationships
    - Lemmatization and morphological analysis
    - Sentence structure analysis
    - Readability and complexity metrics
    - Noun phrase extraction
    - Adjective clustering for sentiment/mood
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="SpacyLinguisticPlugin",
            version="1.0.0",
            description="Comprehensive linguistic analysis using SpaCy NLP",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["linguistic", "pos", "dependencies", "spacy", "nlp", "grammar"],
            execution_priority=ExecutionPriority.NORMAL
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with comprehensive linguistic analysis.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field
            config: Plugin configuration
            
        Returns:
            Dict containing linguistic analysis results
        """
        try:
            if isinstance(field_value, str):
                text = field_value
            elif isinstance(field_value, list):
                text = " ".join(str(item) for item in field_value)
            else:
                text = str(field_value)
            
            if not text.strip():
                self._logger.debug(f"No text found in field {field_name}")
                return self._create_empty_result(field_name)
            
            self._logger.debug(f"Performing linguistic analysis on {len(text)} characters using SpaCy")
            
            service_url = await self.get_plugin_service_url()
            request_data = {
                "text": text,
                "field_name": field_name,
                "options": {
                    "model": config.get("model", "en_core_web_sm"),
                    "extract_pos": config.get("extract_pos", True),
                    "extract_dependencies": config.get("extract_dependencies", True),
                    "extract_lemmas": config.get("extract_lemmas", True),
                    "extract_sentences": config.get("extract_sentences", True),
                    "extract_noun_chunks": config.get("extract_noun_chunks", True),
                    "analyze_readability": config.get("analyze_readability", True),
                    "extract_adjectives": config.get("extract_adjectives", True)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            if response.get("success", False):
                result_data = response.get("result", {})
                linguistic_features = self._structure_linguistic_features(result_data)
                metadata = response.get("metadata", {})
                
                result = {
                    "spacy_linguistic": linguistic_features,
                    "linguistic_summary": self._create_linguistic_summary(linguistic_features),
                    "original_text": text[:200] + "..." if len(text) > 200 else text,
                    "field_name": field_name,
                    "metadata": {
                        "provider": "spacy_linguistic",
                        "model_used": request_data["options"]["model"],
                        "token_count": result_data.get("token_count", 0),
                        "sentence_count": result_data.get("sentence_count", 0),
                        "text_length": len(text),
                        "service_metadata": metadata,
                        "analysis_features": list(linguistic_features.keys())
                    }
                }
            else:
                result = self._create_error_result(
                    field_name, 
                    response.get("error_message", "Unknown error"),
                    text
                )
            
            self._logger.info(
                f"SpaCy linguistic analysis completed for field {field_name}: "
                f"{result['metadata']['token_count']} tokens, {result['metadata']['sentence_count']} sentences"
            )
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"SpaCy linguistic analysis failed for field {field_name}: {e}")
            return self.normalize_text(self._create_error_result(field_name, str(e)))
    
    def _structure_linguistic_features(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structure linguistic features for better organization."""
        structured = {}
        
        # Part-of-speech analysis
        if "pos_tags" in result_data:
            pos_analysis = self._analyze_pos_tags(result_data["pos_tags"])
            structured["pos_analysis"] = pos_analysis
        
        # Dependency analysis
        if "dependencies" in result_data:
            dep_analysis = self._analyze_dependencies(result_data["dependencies"])
            structured["dependency_analysis"] = dep_analysis
        
        # Lemmatization results
        if "lemmas" in result_data:
            structured["lemmatization"] = {
                "lemma_pairs": result_data["lemmas"][:20],  # Limit for readability
                "total_lemmas": len(result_data["lemmas"])
            }
        
        # Sentence analysis
        if "sentences" in result_data:
            structured["sentence_analysis"] = self._analyze_sentences(result_data["sentences"])
        
        # Readability metrics (if available)
        if "readability" in result_data:
            structured["readability"] = result_data["readability"]
        
        # Adjective extraction for mood/sentiment
        if "pos_tags" in result_data:
            adjectives = self._extract_adjectives(result_data["pos_tags"])
            structured["adjective_analysis"] = adjectives
        
        return structured
    
    def _analyze_pos_tags(self, pos_tags: List) -> Dict[str, Any]:
        """Analyze part-of-speech distribution."""
        pos_counts = {}
        total_tokens = len(pos_tags)
        
        for token, pos, tag in pos_tags:
            if pos not in pos_counts:
                pos_counts[pos] = {"count": 0, "examples": []}
            pos_counts[pos]["count"] += 1
            if len(pos_counts[pos]["examples"]) < 3:
                pos_counts[pos]["examples"].append(token)
        
        # Calculate percentages
        for pos in pos_counts:
            pos_counts[pos]["percentage"] = round((pos_counts[pos]["count"] / total_tokens) * 100, 2)
        
        return {
            "total_tokens": total_tokens,
            "pos_distribution": pos_counts,
            "complexity_indicators": {
                "noun_density": pos_counts.get("NOUN", {}).get("percentage", 0),
                "verb_density": pos_counts.get("VERB", {}).get("percentage", 0),
                "adjective_density": pos_counts.get("ADJ", {}).get("percentage", 0),
                "adverb_density": pos_counts.get("ADV", {}).get("percentage", 0)
            }
        }
    
    def _analyze_dependencies(self, dependencies: List) -> Dict[str, Any]:
        """Analyze dependency relationships."""
        dep_counts = {}
        root_count = 0
        
        for dep_info in dependencies:
            dep = dep_info.get("dep", "")
            if dep == "ROOT":
                root_count += 1
            
            if dep not in dep_counts:
                dep_counts[dep] = {"count": 0, "examples": []}
            dep_counts[dep]["count"] += 1
            if len(dep_counts[dep]["examples"]) < 2:
                dep_counts[dep]["examples"].append(dep_info.get("text", ""))
        
        return {
            "total_dependencies": len(dependencies),
            "root_sentences": root_count,
            "dependency_distribution": dict(sorted(dep_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:10]),
            "complexity_score": self._calculate_dependency_complexity(dep_counts)
        }
    
    def _analyze_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentence structure and complexity."""
        if not sentences:
            return {"sentence_count": 0}
        
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        return {
            "sentence_count": len(sentences),
            "avg_sentence_length": round(sum(sentence_lengths) / len(sentence_lengths), 2),
            "min_sentence_length": min(sentence_lengths),
            "max_sentence_length": max(sentence_lengths),
            "short_sentences": sum(1 for length in sentence_lengths if length <= 10),
            "long_sentences": sum(1 for length in sentence_lengths if length >= 20),
            "complexity_level": "high" if sum(sentence_lengths) / len(sentence_lengths) > 15 else "medium" if sum(sentence_lengths) / len(sentence_lengths) > 10 else "low"
        }
    
    def _extract_adjectives(self, pos_tags: List) -> Dict[str, Any]:
        """Extract and analyze adjectives for mood/sentiment indicators."""
        adjectives = []
        
        for token, pos, tag in pos_tags:
            if pos == "ADJ":
                adjectives.append(token.lower())
        
        # Categorize adjectives (basic sentiment/mood analysis)
        positive_indicators = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "brilliant", "outstanding"]
        negative_indicators = ["bad", "terrible", "awful", "horrible", "disappointing", "poor", "weak"]
        intensity_indicators = ["very", "extremely", "highly", "incredibly", "exceptionally", "remarkably"]
        
        positive_count = sum(1 for adj in adjectives if adj in positive_indicators)
        negative_count = sum(1 for adj in adjectives if adj in negative_indicators)
        intensity_count = sum(1 for adj in adjectives if adj in intensity_indicators)
        
        return {
            "total_adjectives": len(adjectives),
            "unique_adjectives": len(set(adjectives)),
            "adjective_list": list(set(adjectives))[:15],  # Top 15 unique
            "sentiment_indicators": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "intensity_count": intensity_count,
                "sentiment_score": positive_count - negative_count
            }
        }
    
    def _calculate_dependency_complexity(self, dep_counts: Dict) -> float:
        """Calculate a complexity score based on dependency types."""
        complex_deps = ["ccomp", "xcomp", "advcl", "acl", "nmod", "amod"]
        simple_deps = ["det", "aux", "cop", "case"]
        
        complex_count = sum(dep_counts.get(dep, {}).get("count", 0) for dep in complex_deps)
        simple_count = sum(dep_counts.get(dep, {}).get("count", 0) for dep in simple_deps)
        total_deps = sum(info["count"] for info in dep_counts.values())
        
        if total_deps == 0:
            return 0.0
        
        complexity_ratio = complex_count / total_deps
        return round(complexity_ratio * 10, 2)  # Scale to 0-10
    
    def _create_linguistic_summary(self, linguistic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of linguistic analysis."""
        summary = {}
        
        # Overall complexity
        pos_analysis = linguistic_features.get("pos_analysis", {})
        sentence_analysis = linguistic_features.get("sentence_analysis", {})
        dep_analysis = linguistic_features.get("dependency_analysis", {})
        
        complexity_indicators = []
        if sentence_analysis.get("complexity_level") == "high":
            complexity_indicators.append("long sentences")
        if pos_analysis.get("complexity_indicators", {}).get("adjective_density", 0) > 15:
            complexity_indicators.append("descriptive language")
        if dep_analysis.get("complexity_score", 0) > 5:
            complexity_indicators.append("complex grammar")
        
        summary["overall_complexity"] = "high" if len(complexity_indicators) >= 2 else "medium" if len(complexity_indicators) == 1 else "low"
        summary["complexity_factors"] = complexity_indicators
        
        # Writing style indicators
        adj_analysis = linguistic_features.get("adjective_analysis", {})
        sentiment_score = adj_analysis.get("sentiment_indicators", {}).get("sentiment_score", 0)
        
        summary["writing_style"] = {
            "sentiment_tendency": "positive" if sentiment_score > 2 else "negative" if sentiment_score < -2 else "neutral",
            "descriptiveness": "high" if adj_analysis.get("total_adjectives", 0) > 10 else "medium" if adj_analysis.get("total_adjectives", 0) > 5 else "low"
        }
        
        return summary
    
    def _create_empty_result(self, field_name: str) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            "spacy_linguistic": {},
            "linguistic_summary": {},
            "original_text": "",
            "field_name": field_name,
            "metadata": {
                "provider": "spacy_linguistic",
                "token_count": 0,
                "sentence_count": 0,
                "success": True
            }
        }
    
    def _create_error_result(self, field_name: str, error: str, text: str = "") -> Dict[str, Any]:
        """Create error result structure."""
        return {
            "spacy_linguistic": {},
            "linguistic_summary": {},
            "original_text": text[:200] + "..." if len(text) > 200 else text,
            "field_name": field_name,
            "error": error,
            "metadata": {
                "provider": "spacy_linguistic",
                "success": False,
                "token_count": 0,
                "sentence_count": 0
            }
        }
    
    async def analyze_readability(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze text readability and complexity.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Readability analysis results
        """
        if config is None:
            config = {}
        
        config["analyze_readability"] = True
        result = await self.enrich_field("text", text, config)
        
        readability = result.get("spacy_linguistic", {}).get("readability", {})
        sentence_analysis = result.get("spacy_linguistic", {}).get("sentence_analysis", {})
        
        return self.normalize_text({
            "readability": readability,
            "sentence_complexity": sentence_analysis.get("complexity_level", "unknown"),
            "avg_sentence_length": sentence_analysis.get("avg_sentence_length", 0),
            "metadata": result.get("metadata", {})
        })
    
    async def extract_adjectives(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract adjectives for mood/sentiment analysis.
        
        Args:
            text: Text to extract adjectives from
            config: Extraction configuration
            
        Returns:
            Adjective extraction results
        """
        if config is None:
            config = {}
        
        config["extract_adjectives"] = True
        result = await self.enrich_field("text", text, config)
        
        adjectives = result.get("spacy_linguistic", {}).get("adjective_analysis", {})
        
        return self.normalize_text({
            "adjectives": adjectives.get("adjective_list", []),
            "sentiment_indicators": adjectives.get("sentiment_indicators", {}),
            "total_count": adjectives.get("total_adjectives", 0),
            "metadata": result.get("metadata", {})
        })
    
    async def analyze_sentence_structure(
        self, 
        text: str, 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentence structure and complexity.
        
        Args:
            text: Text to analyze
            config: Analysis configuration
            
        Returns:
            Sentence structure analysis results
        """
        if config is None:
            config = {}
        
        config["extract_sentences"] = True
        config["extract_dependencies"] = True
        result = await self.enrich_field("text", text, config)
        
        sentence_analysis = result.get("spacy_linguistic", {}).get("sentence_analysis", {})
        dependency_analysis = result.get("spacy_linguistic", {}).get("dependency_analysis", {})
        
        return self.normalize_text({
            "sentence_structure": sentence_analysis,
            "grammatical_complexity": dependency_analysis.get("complexity_score", 0),
            "dependency_types": len(dependency_analysis.get("dependency_distribution", {})),
            "metadata": result.get("metadata", {})
        })
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            service_url = self.get_service_url("linguistic", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "spacy_linguistic_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "spacy_linguistic_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health