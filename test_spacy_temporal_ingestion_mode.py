#!/usr/bin/env python3
"""
Test suite for SpacyWithFallbackIngestionAndQueryPlugin in INGESTION MODE.
Tests temporal analysis of movie content for MongoDB storage.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.temporal import SpacyWithFallbackIngestionAndQueryPlugin
from src.plugins.base import PluginExecutionContext


class TestSpacyTemporalIngestionMode:
    """Test spaCy temporal plugin in ingestion mode."""
    
    def __init__(self):
        self.plugin = SpacyWithFallbackIngestionAndQueryPlugin()
        
    async def setup(self):
        """Initialize the plugin."""
        print("🚀 Initializing SpacyWithFallbackIngestionAndQueryPlugin for INGESTION MODE testing...")
        success = await self.plugin.initialize({})
        if success:
            print("✅ Plugin initialized successfully")
        else:
            print("❌ Plugin initialization failed")
        print(f"📊 Available analysis methods: {self._get_available_methods()}")
        
    def _get_available_methods(self):
        """Check which analysis methods are available."""
        methods = []
        if hasattr(self.plugin, 'nlp') and self.plugin.nlp:
            methods.append("spacy")
        if hasattr(self.plugin, 'ner_pipeline') and self.plugin.ner_pipeline:
            methods.append("transformers")
        
        try:
            from dateutil.parser import parse
            methods.append("dateutil")
        except ImportError:
            pass
            
        try:
            import arrow
            methods.append("arrow")
        except ImportError:
            pass
            
        return methods
    
    async def test_movie_content_ingestion(self):
        """Test temporal analysis of movie content for ingestion."""
        print("\n🎬 Testing Movie Content Temporal Analysis for Ingestion")
        print("=" * 70)
        
        # Test cases: movie content with temporal references
        movie_test_cases = [
            {
                "title": "Blade Runner",
                "overview": "Set in dystopian Los Angeles in 2019, this neo-noir follows a blade runner tasked with hunting down replicants.",
                "plot": "In the year 2019, synthetic humans called replicants are created to work on off-world colonies. When four escape to Earth, blade runner Rick Deckard is called out of retirement to hunt them down.",
                "genres": ["sci-fi", "thriller"],
                "description": "Classic 1980s cyberpunk film"
            },
            {
                "title": "Casablanca", 
                "overview": "Set during World War II, this romance follows an American expatriate in Casablanca who must choose between love and virtue.",
                "plot": "During World War II in December 1941, American Rick Blaine runs a nightclub in Casablanca. When his former lover Ilsa arrives with her husband, a Czech resistance leader, Rick faces a difficult choice.",
                "genres": ["romance", "drama"],
                "description": "Classic 1940s wartime romance"
            },
            {
                "title": "Back to the Future",
                "overview": "A teenager travels back to the 1950s in a time machine and must ensure his parents fall in love.",
                "plot": "In 1985, teenager Marty McFly accidentally travels back to 1955 in a time machine built by his eccentric scientist friend Doc Brown. He must ensure his teenage parents fall in love or risk erasing his own existence.",
                "genres": ["sci-fi", "comedy"],
                "description": "1980s time travel comedy"
            },
            {
                "title": "Saving Private Ryan",
                "overview": "Set during World War II, a group of soldiers search for a paratrooper whose brothers have been killed in action.",
                "plot": "Following the Normandy landings in June 1944 during World War II, Captain Miller leads a squad to find Private Ryan, whose three brothers have been killed in action.",
                "genres": ["war", "drama"],
                "description": "1990s World War II epic"
            },
            {
                "title": "The Matrix",
                "overview": "A computer hacker discovers reality is a simulation and joins a rebellion against the machines.",
                "plot": "In the late 1990s, computer programmer Neo discovers that reality as he knows it is actually a computer simulation called the Matrix. He joins a rebellion led by Morpheus to free humanity from their digital prison.",
                "genres": ["sci-fi", "action"],
                "description": "Late 1990s cyberpunk action film"
            }
        ]
        
        for i, movie_data in enumerate(movie_test_cases, 1):
            print(f"\n📽️ Test {i}: {movie_data['title']}")
            print(f"   Overview: {movie_data['overview'][:80]}...")
            
            try:
                # Create mock context
                context = PluginExecutionContext(
                    user_id="test_user",
                    metadata={"media_type": "movie", "test_mode": True}
                )
                
                # Test INGESTION MODE: embellish_embed_data
                enriched_data = await self.plugin.embellish_embed_data(movie_data, context)
                
                print(f"   ✅ Ingestion analysis completed")
                
                # Check if temporal analysis was added
                if 'enhanced_fields' in enriched_data:
                    enhanced = enriched_data['enhanced_fields']
                    
                    # Check spacy_temporal_analysis
                    if 'spacy_temporal_analysis' in enhanced:
                        analysis = enhanced['spacy_temporal_analysis']
                        expressions = len(analysis.get('expressions', []))
                        normalized = len(analysis.get('normalized', []))
                        methods = analysis.get('analysis_methods', [])
                        confidence = analysis.get('confidence_level', 'unknown')
                        scope = analysis.get('temporal_scope', 'unknown')
                        
                        print(f"   📊 Temporal Analysis: {expressions} expressions, {normalized} normalized")
                        print(f"   🔧 Methods: {', '.join(methods)}")
                        print(f"   📈 Confidence: {confidence}, Scope: {scope}")
                        
                        # Show some results
                        if normalized > 0:
                            print(f"   🎯 Normalized Results:")
                            for norm in analysis['normalized'][:3]:  # Show first 3
                                start = norm.get('start')
                                end = norm.get('end')
                                precision = norm.get('precision')
                                method = norm.get('method', 'unknown')
                                print(f"      - '{norm['text']}' → {start}-{end} ({precision}) via {method}")
                    
                    # Check spacy_temporal_metadata
                    if 'spacy_temporal_metadata' in enhanced:
                        metadata = enhanced['spacy_temporal_metadata']
                        periods = len(metadata.get('detected_time_periods', []))
                        scope = metadata.get('temporal_scope', 'unknown')
                        confidence = metadata.get('confidence', 'unknown')
                        
                        print(f"   📋 Temporal Metadata: {periods} periods, scope: {scope}, confidence: {confidence}")
                        
                        # Show periods
                        for period in metadata.get('detected_time_periods', [])[:2]:
                            print(f"      - {period.get('text')} ({period.get('precision')})")
                    
                    # Check spacy_temporal_search_tags
                    if 'spacy_temporal_search_tags' in enhanced:
                        tags = enhanced['spacy_temporal_search_tags']
                        print(f"   🏷️  Search Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
                    
                else:
                    print(f"   ⚠️  No enhanced_fields added")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def test_genre_specific_content(self):
        """Test temporal analysis with different movie genres."""
        print("\n🎭 Testing Genre-Specific Temporal Analysis")
        print("=" * 70)
        
        genre_test_cases = [
            {
                "genre": "Western",
                "content": "Set in the American frontier during the 1870s, this tale follows a gunslinger in the Old West as he faces outlaws in a dusty frontier town.",
                "expected_periods": ["1870s", "old_west"]
            },
            {
                "genre": "Film Noir", 
                "content": "In post-war 1947 Los Angeles, a private detective investigates a murder case that leads him through the dark underbelly of the city.",
                "expected_periods": ["1940s", "post-war"]
            },
            {
                "genre": "Space Opera",
                "content": "Set in the distant future of 2387, humanity has colonized the galaxy and faces an alien threat that could destroy all known civilizations.",
                "expected_periods": ["future", "2387"]
            },
            {
                "genre": "Period Drama",
                "content": "During the Victorian era in 1889 London, a young woman challenges social conventions while navigating the complexities of high society.",
                "expected_periods": ["1889", "victorian_era"]
            }
        ]
        
        for test_case in genre_test_cases:
            print(f"\n🎬 Genre: {test_case['genre']}")
            print(f"   Content: {test_case['content'][:80]}...")
            
            try:
                context = PluginExecutionContext(metadata={"media_type": "movie"})
                movie_data = {
                    "overview": test_case['content'],
                    "genres": [test_case['genre'].lower()]
                }
                
                enriched_data = await self.plugin.embellish_embed_data(movie_data, context)
                
                if 'enhanced_fields' in enriched_data and 'spacy_temporal_analysis' in enriched_data['enhanced_fields']:
                    analysis = enriched_data['enhanced_fields']['spacy_temporal_analysis']
                    normalized = analysis.get('normalized', [])
                    
                    if normalized:
                        print(f"   ✅ Found {len(normalized)} temporal references:")
                        for norm in normalized:
                            print(f"      - {norm.get('text')} → {norm.get('start')}-{norm.get('end')}")
                    else:
                        print(f"   ⚠️  No temporal references detected")
                else:
                    print(f"   ❌ No temporal analysis generated")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")


async def main():
    """Run ingestion mode tests."""
    print("🧪 SpacyWithFallbackIngestionAndQueryPlugin - INGESTION MODE Tests")
    print("=" * 80)
    print("Testing temporal analysis of movie content for MongoDB storage")
    
    tester = TestSpacyTemporalIngestionMode()
    
    try:
        await tester.setup()
        await tester.test_movie_content_ingestion()
        await tester.test_genre_specific_content()
        
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Ingestion mode testing completed!")
    print("📝 Run with: python test_spacy_temporal_ingestion_mode.py")


if __name__ == "__main__":
    asyncio.run(main())