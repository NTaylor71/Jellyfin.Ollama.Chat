"""
Merge Keywords Plugin
HTTP-only plugin that merges keyword results from multiple provider plugins.
"""

import logging
from typing import Dict, Any, List, Set
from collections import Counter

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class MergeKeywordsPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that merges keyword results from multiple providers.
    
    Features:
    - Combines results from ConceptNet, LLM, and Gensim plugins
    - Multiple merge strategies (union, intersection, weighted)
    - Deduplication and ranking
    - No HTTP calls - just local processing
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MergeKeywordsPlugin",
            version="1.0.0", 
            description="Merges keyword results from multiple provider plugins",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["keyword", "merge", "fusion", "aggregation"],
            execution_priority=ExecutionPriority.LOW
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge keyword enrichments from multiple sources.
        
        Note: This plugin doesn't make HTTP calls. It processes existing
        enrichment results from other plugins.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Original field value (for reference)
            config: Merge configuration including input results
            
        Returns:
            Dict containing merged keyword results
        """
        try:
            
            input_enrichments = config.get("input_enrichments", [])
            if not input_enrichments:
                self._logger.warning(f"No input enrichments provided for merging field {field_name}")
                result = {"merged_keywords": []}
                
                return self.normalize_text(result)
            
            merge_strategy = config.get("strategy", "union")
            max_results = config.get("max_results", 20)
            
            self._logger.debug(
                f"Merging {len(input_enrichments)} enrichments using {merge_strategy} strategy"
            )
            

            keyword_sources = self._extract_keywords_from_enrichments(input_enrichments)
            
            if not keyword_sources:
                result = {"merged_keywords": []}
                
                return self.normalize_text(result)
            

            if merge_strategy == "union":
                merged_keywords = self._merge_union(keyword_sources, max_results)
            elif merge_strategy == "intersection":
                merged_keywords = self._merge_intersection(keyword_sources, max_results)
            elif merge_strategy == "weighted":
                weights = config.get("weights", {})
                merged_keywords = self._merge_weighted(keyword_sources, weights, max_results)
            elif merge_strategy == "ranked":
                merged_keywords = self._merge_ranked(keyword_sources, max_results)
            else:

                merged_keywords = self._merge_union(keyword_sources, max_results)
            

            total_input_keywords = sum(len(source["keywords"]) for source in keyword_sources)
            
            self._logger.info(
                f"Merged {total_input_keywords} keywords from {len(keyword_sources)} sources "
                f"to {len(merged_keywords)} final keywords"
            )
            
            result = {
                "merged_keywords": merged_keywords,
                "original_field": field_name,
                "merge_metadata": {
                    "strategy": merge_strategy,
                    "source_count": len(keyword_sources),
                    "total_input_keywords": total_input_keywords,
                    "final_keyword_count": len(merged_keywords),
                    "sources": [source["provider"] for source in keyword_sources]
                }
            }
            
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"Keyword merging failed for field {field_name}: {e}")
            result = {
                "merged_keywords": [],
                "original_field": field_name,
                "error": str(e),
                "merge_metadata": {
                    "strategy": "failed",
                    "success": False
                }
            }
            
            
            return self.normalize_text(result)
    
    def _extract_keywords_from_enrichments(self, enrichments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract keywords and metadata from enrichment results."""
        keyword_sources = []
        
        for enrichment in enrichments:
            provider = "unknown"
            keywords = []
            

            if "conceptnet_keywords" in enrichment:
                provider = "conceptnet"
                keywords = enrichment["conceptnet_keywords"]
            

            elif "llm_keywords" in enrichment:
                provider = "llm"
                keywords = enrichment["llm_keywords"]
            

            elif "gensim_similar" in enrichment:
                provider = "gensim"
                similar_terms = enrichment["gensim_similar"]
                

                if similar_terms and isinstance(similar_terms[0], dict):
                    keywords = [item.get("term", str(item)) for item in similar_terms]
                else:
                    keywords = similar_terms
            

            elif "keywords" in enrichment:
                keywords = enrichment["keywords"]
                provider = enrichment.get("metadata", {}).get("provider", "unknown")
            
            if keywords:
                keyword_sources.append({
                    "provider": provider,
                    "keywords": [str(kw).strip().lower() for kw in keywords if str(kw).strip()],
                    "metadata": enrichment.get("metadata", {})
                })
        
        return keyword_sources
    
    def _merge_union(self, sources: List[Dict[str, Any]], max_results: int) -> List[str]:
        """Merge using union strategy - combine all unique keywords."""
        all_keywords = set()
        
        for source in sources:
            all_keywords.update(source["keywords"])
        

        result = list(all_keywords)[:max_results]
        return result
    
    def _merge_intersection(self, sources: List[Dict[str, Any]], max_results: int) -> List[str]:
        """Merge using intersection strategy - only keywords present in multiple sources."""
        if not sources:
            return []
        
        if len(sources) == 1:
            return sources[0]["keywords"][:max_results]
        

        common_keywords = set(sources[0]["keywords"])
        

        for source in sources[1:]:
            common_keywords &= set(source["keywords"])
        
        result = list(common_keywords)[:max_results]
        return result
    
    def _merge_weighted(
        self, 
        sources: List[Dict[str, Any]], 
        weights: Dict[str, float], 
        max_results: int
    ) -> List[str]:
        """Merge using weighted strategy - weight keywords by provider importance."""
        keyword_scores = Counter()
        

        default_weights = {
            "llm": 1.0,     
            "conceptnet": 0.8,
            "gensim": 0.6    
        }
        
        for source in sources:
            provider = source["provider"]
            weight = weights.get(provider, default_weights.get(provider, 0.5))
            
            for keyword in source["keywords"]:
                keyword_scores[keyword] += weight
        

        sorted_keywords = keyword_scores.most_common(max_results)
        result = [keyword for keyword, score in sorted_keywords]
        return result
    
    def _merge_ranked(self, sources: List[Dict[str, Any]], max_results: int) -> List[str]:
        """Merge using ranking strategy - prioritize by frequency across sources."""
        keyword_counts = Counter()
        
        for source in sources:
            for keyword in source["keywords"]:
                keyword_counts[keyword] += 1
        

        sorted_keywords = keyword_counts.most_common(max_results)
        result = [keyword for keyword, count in sorted_keywords]
        return result
    
    async def merge_enrichments(
        self,
        enrichments: List[Dict[str, Any]],
        strategy: str = "union",
        max_results: int = 20,
        weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Direct merge method for backwards compatibility.
        
        Args:
            enrichments: List of enrichment results to merge
            strategy: Merge strategy ("union", "intersection", "weighted", "ranked")
            max_results: Maximum number of keywords to return
            weights: Weights for providers (used with "weighted" strategy)
            
        Returns:
            Merged results
        """
        config = {
            "input_enrichments": enrichments,
            "strategy": strategy,
            "max_results": max_results
        }
        
        if weights:
            config["weights"] = weights
        
        
        result = await self.enrich_field("merged", "", config)
        response = {
            "merged_keywords": result.get("merged_keywords", []),
            "metadata": result.get("merge_metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health - no HTTP dependencies."""
        base_health = await super().health_check()
        

        base_health["service_dependencies"] = "none"
        base_health["merge_strategies"] = ["union", "intersection", "weighted", "ranked"]
        
        return base_health