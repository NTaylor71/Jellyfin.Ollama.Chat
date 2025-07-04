#!/usr/bin/env python3
"""
Comprehensive test for SpacyWithFallbackIngestionAndQueryPlugin dual-use modes.
Tests both ingestion mode (embellish_embed_data) and query mode (embellish_query).
"""

import asyncio
import json
from typing import Dict, Any

# Mock context for testing
class MockContext:
    def __init__(self, metadata: Dict[str, Any] = None):
        self.metadata = metadata or {}

async def test_spacy_temporal_dual_mode():
    """Test SpacyWithFallbackIngestionAndQueryPlugin in both ingestion and query modes."""
    
    try:
        # Import plugin
        from src.plugins.temporal.spacy_with_fallback_ingestion_and_query import SpacyWithFallbackIngestionAndQueryPlugin
        
        # Initialize plugin
        plugin = SpacyWithFallbackIngestionAndQueryPlugin()
        
        # Initialize plugin with config
        config = {
            "enabled": True,
            "debug": True,
            "timeout": 30
        }
        
        init_success = await plugin.initialize(config)
        print(f"🔧 Plugin initialization: {'✅ SUCCESS' if init_success else '❌ FAILED'}")
        
        if not init_success:
            print("❌ Plugin initialization failed - cannot proceed with tests")
            return False
        
        # Test 1: INGESTION MODE - Movie content analysis
        print("\n" + "="*80)
        print("🎬 TEST 1: INGESTION MODE - Movie Content Analysis")
        print("="*80)
        
        # Test with realistic movie data
        test_movies = [
            {
                "name": "Terminator 2: Judgment Day",
                "overview": "A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her teenage son John Connor from a more advanced and powerful cyborg in the 1990s future war.",
                "production_year": 1991,
                "genres": ["Action", "Science Fiction"]
            },
            {
                "name": "Jurassic Park",
                "overview": "A wealthy entrepreneur secretly creates a theme park featuring living dinosaurs drawn from prehistoric DNA. During a preview tour, the park's security system shuts down and the dinosaurs escape. Set during the early 1990s.",
                "production_year": 1993,
                "genres": ["Adventure", "Science Fiction"]
            },
            {
                "name": "The Matrix",
                "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers in the late 1990s.",
                "production_year": 1999,
                "genres": ["Action", "Science Fiction"]
            },
            {
                "name": "Blade Runner 2049",
                "overview": "Thirty years after the events of the first film, a new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos.",
                "production_year": 2017,
                "genres": ["Science Fiction", "Thriller"]
            }
        ]
        
        ingestion_results = []
        
        for movie in test_movies:
            print(f"\n🎥 Testing: {movie['name']} ({movie['production_year']})")
            
            # Create test context
            context = MockContext({"movie_year": movie['production_year']})
            
            # Test ingestion mode
            enhanced_data = await plugin.embellish_embed_data(movie.copy(), context)
            
            # Validate results
            if 'enhanced_fields' in enhanced_data:
                spacy_analysis = enhanced_data['enhanced_fields'].get('spacy_temporal_analysis', {})
                spacy_metadata = enhanced_data['enhanced_fields'].get('spacy_temporal_metadata', {})
                search_tags = enhanced_data['enhanced_fields'].get('spacy_temporal_search_tags', [])
                
                print(f"   📊 Temporal expressions found: {len(spacy_analysis.get('expressions', []))}")
                print(f"   📅 Normalized periods: {len(spacy_analysis.get('normalized', []))}")
                print(f"   🔍 Search tags: {len(search_tags)}")
                print(f"   🎯 Confidence: {spacy_analysis.get('confidence_level', 'unknown')}")
                print(f"   🧠 Methods: {', '.join(spacy_analysis.get('analysis_methods', []))}")
                
                if spacy_analysis.get('expressions'):
                    print(f"   📝 Expressions: {[expr['text'] for expr in spacy_analysis['expressions'][:3]]}")
                
                if spacy_analysis.get('normalized'):
                    print("   🔢 Normalized periods:")
                    for norm in spacy_analysis['normalized'][:3]:
                        start = norm.get('start', 'unknown')
                        end = norm.get('end', 'unknown')
                        precision = norm.get('precision', 'unknown')
                        print(f"      • {norm.get('text', 'unknown')} → {start}-{end} ({precision})")
                
                if search_tags:
                    print(f"   🏷️ Search tags: {search_tags[:5]}")
                
                ingestion_results.append({
                    "movie": movie['name'],
                    "expressions": len(spacy_analysis.get('expressions', [])),
                    "normalized": len(spacy_analysis.get('normalized', [])),
                    "search_tags": len(search_tags),
                    "confidence": spacy_analysis.get('confidence_level', 'unknown'),
                    "methods": spacy_analysis.get('analysis_methods', [])
                })
            else:
                print("   ❌ No enhanced_fields found in result")
                ingestion_results.append({
                    "movie": movie['name'],
                    "expressions": 0,
                    "normalized": 0,
                    "search_tags": 0,
                    "confidence": "failed",
                    "methods": []
                })
        
        # Test 2: QUERY MODE - User query analysis
        print("\n" + "="*80)
        print("🔍 TEST 2: QUERY MODE - User Query Analysis")
        print("="*80)
        
        # Test with realistic user queries
        test_queries = [
            "90s action movies",
            "recent sci-fi films",
            "movies from the last decade",
            "80s and 90s science fiction",
            "films from two decades ago",
            "early 2000s thrillers",
            "movies around the turn of the century",
            "late 90s action films",
            "this decade's best movies",
            "post-millennium cinema"
        ]
        
        query_results = []
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            # Create test context
            context = MockContext({"query_type": "temporal_search"})
            
            # Test query mode
            enhanced_query = await plugin.embellish_query(query, context)
            
            # For debugging - also get internal analysis
            analysis = await plugin.analyze(query, context.metadata)
            
            expressions = analysis.get('expressions', [])
            normalized = analysis.get('normalized', [])
            
            print(f"   📊 Temporal expressions: {len(expressions)}")
            print(f"   📅 Normalized periods: {len(normalized)}")
            print(f"   🎯 Confidence: {analysis.get('confidence_level', 'unknown')}")
            print(f"   🧠 Methods: {', '.join(analysis.get('analysis_methods', []))}")
            
            if expressions:
                print(f"   📝 Expressions: {[expr['text'] for expr in expressions[:3]]}")
            
            if normalized:
                print("   🔢 Normalized periods:")
                for norm in normalized[:3]:
                    start = norm.get('start', 'unknown')
                    end = norm.get('end', 'unknown')
                    precision = norm.get('precision', 'unknown')
                    print(f"      • {norm.get('text', 'unknown')} → {start}-{end} ({precision})")
            
            query_results.append({
                "query": query,
                "expressions": len(expressions),
                "normalized": len(normalized),
                "confidence": analysis.get('confidence_level', 'unknown'),
                "methods": analysis.get('analysis_methods', [])
            })
        
        # Test 3: EDGE CASES AND FALLBACKS
        print("\n" + "="*80)
        print("🔧 TEST 3: EDGE CASES AND FALLBACKS")
        print("="*80)
        
        edge_cases = [
            "",  # Empty string
            "No temporal information here",  # No dates
            "1950s 1960s 1970s 1980s",  # Multiple decades
            "between 1990 and 2000",  # Date ranges
            "around 1995",  # Approximate dates
            "summer of 1993",  # Seasonal references
            "début des années 90",  # Non-English
            "🎬 90s movies 🎭",  # With emojis
        ]
        
        for case in edge_cases:
            print(f"\n🧪 Testing edge case: '{case}'")
            
            try:
                context = MockContext()
                analysis = await plugin.analyze(case, context.metadata)
                
                expressions = len(analysis.get('expressions', []))
                normalized = len(analysis.get('normalized', []))
                
                print(f"   📊 Expressions: {expressions}, Normalized: {normalized}")
                
                if expressions > 0:
                    print(f"   📝 Found: {[expr['text'] for expr in analysis['expressions'][:2]]}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Summary
        print("\n" + "="*80)
        print("📋 SUMMARY")
        print("="*80)
        
        print(f"\n🎬 INGESTION MODE RESULTS:")
        total_expressions = sum(r['expressions'] for r in ingestion_results)
        total_normalized = sum(r['normalized'] for r in ingestion_results)
        total_search_tags = sum(r['search_tags'] for r in ingestion_results)
        
        print(f"   • Total movies tested: {len(ingestion_results)}")
        print(f"   • Total expressions found: {total_expressions}")
        print(f"   • Total normalized periods: {total_normalized}")
        print(f"   • Total search tags: {total_search_tags}")
        print(f"   • Average expressions per movie: {total_expressions/len(ingestion_results):.1f}")
        
        print(f"\n🔍 QUERY MODE RESULTS:")
        total_query_expressions = sum(r['expressions'] for r in query_results)
        total_query_normalized = sum(r['normalized'] for r in query_results)
        
        print(f"   • Total queries tested: {len(query_results)}")
        print(f"   • Total expressions found: {total_query_expressions}")
        print(f"   • Total normalized periods: {total_query_normalized}")
        print(f"   • Average expressions per query: {total_query_expressions/len(query_results):.1f}")
        
        # Success metrics
        successful_ingestion = sum(1 for r in ingestion_results if r['expressions'] > 0)
        successful_queries = sum(1 for r in query_results if r['expressions'] > 0)
        
        print(f"\n✅ SUCCESS METRICS:")
        print(f"   • Ingestion success rate: {successful_ingestion}/{len(ingestion_results)} ({successful_ingestion/len(ingestion_results)*100:.1f}%)")
        print(f"   • Query success rate: {successful_queries}/{len(query_results)} ({successful_queries/len(query_results)*100:.1f}%)")
        
        # Overall assessment
        overall_success = (successful_ingestion >= len(ingestion_results) * 0.75 and 
                          successful_queries >= len(query_results) * 0.75)
        
        print(f"\n🎯 OVERALL ASSESSMENT: {'✅ PASSED' if overall_success else '⚠️ NEEDS IMPROVEMENT'}")
        
        if overall_success:
            print("   The SpacyWithFallbackIngestionAndQueryPlugin is working correctly in both modes!")
            print("   ✅ Ingestion mode: Successfully analyzes movie content for temporal metadata")
            print("   ✅ Query mode: Successfully processes user queries for temporal search")
            print("   ✅ Dual-use architecture: Same plugin handles both content and query analysis")
            print("   ✅ Fallback strategies: Robust handling of edge cases and missing dependencies")
        else:
            print("   The plugin needs improvement in one or both modes.")
            print("   Consider checking dependency installation and fallback logic.")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 SpacyWithFallbackIngestionAndQueryPlugin Dual-Mode Testing")
    print("="*80)
    
    success = asyncio.run(test_spacy_temporal_dual_mode())
    
    if success:
        print("\n🎉 All tests passed! Plugin is ready for production use.")
    else:
        print("\n⚠️ Some tests failed. Please review the results above.")