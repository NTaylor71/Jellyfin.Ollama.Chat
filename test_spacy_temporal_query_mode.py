#!/usr/bin/env python3
"""
Test suite for SpacyWithFallbackIngestionAndQueryPlugin in QUERY MODE.
Tests temporal enhancement of user search queries.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.temporal import SpacyWithFallbackIngestionAndQueryPlugin
from src.plugins.base import PluginExecutionContext


class TestSpacyTemporalQueryMode:
    """Test spaCy temporal plugin in query mode."""
    
    def __init__(self):
        self.plugin = SpacyWithFallbackIngestionAndQueryPlugin()
        
    async def setup(self):
        """Initialize the plugin."""
        print("🚀 Initializing SpacyWithFallbackIngestionAndQueryPlugin for QUERY MODE testing...")
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
    
    async def test_user_query_enhancement(self):
        """Test temporal enhancement of user search queries."""
        print("\n🔍 Testing User Query Temporal Enhancement")
        print("=" * 70)
        
        # Test cases: user search queries with temporal expressions
        query_test_cases = [
            {
                "query": "90s action movies",
                "description": "Simple decade reference",
                "expected_enhancement": "temporal decade detection"
            },
            {
                "query": "recent sci-fi films",
                "description": "Recent temporal reference",
                "expected_enhancement": "relative temporal detection"
            },
            {
                "query": "movies from the last decade",
                "description": "Relative decade reference",
                "expected_enhancement": "relative decade calculation"
            },
            {
                "query": "classic films from the 1950s",
                "description": "Specific decade with qualifier",
                "expected_enhancement": "decade + era detection"
            },
            {
                "query": "horror movies from two decades ago",
                "description": "Complex relative expression",
                "expected_enhancement": "relative calculation"
            },
            {
                "query": "sci-fi movies like Blade Runner but newer",
                "description": "Comparative temporal reference",
                "expected_enhancement": "comparative temporal analysis"
            },
            {
                "query": "summer blockbusters from 1999",
                "description": "Seasonal + year reference",
                "expected_enhancement": "seasonal temporal detection"
            },
            {
                "query": "early 2000s thriller movies",
                "description": "Early decade modifier",
                "expected_enhancement": "decade + modifier detection"
            },
            {
                "query": "films between 1995 and 2005",
                "description": "Year range query",
                "expected_enhancement": "range temporal detection"
            },
            {
                "query": "this decade's superhero movies",
                "description": "Current decade reference",
                "expected_enhancement": "current temporal detection"
            }
        ]
        
        for i, test_case in enumerate(query_test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case['description']}")
            print(f"   Query: '{test_case['query']}'")
            
            try:
                # Create mock context for query processing
                context = PluginExecutionContext(
                    user_id="test_user",
                    session_id="test_session",
                    metadata={"query_type": "search", "test_mode": True}
                )
                
                # Test QUERY MODE: embellish_query
                enhanced_query = await self.plugin.embellish_query(test_case['query'], context)
                
                print(f"   ✅ Query processing completed")
                print(f"   📝 Enhanced Query: '{enhanced_query}'")
                
                # Test the underlying temporal analysis (what the query enhancement uses)
                analysis = await self.plugin.analyze(test_case['query'], context.metadata)
                
                expressions = len(analysis.get('expressions', []))
                normalized = len(analysis.get('normalized', []))
                methods = analysis.get('analysis_methods', [])
                confidence = analysis.get('confidence_level', 'unknown')
                scope = analysis.get('temporal_scope', 'unknown')
                
                print(f"   📊 Temporal Analysis: {expressions} expressions, {normalized} normalized")
                print(f"   🔧 Methods: {', '.join(methods)}")
                print(f"   📈 Confidence: {confidence}, Scope: {scope}")
                
                # Show normalized results for query enhancement
                if normalized > 0:
                    print(f"   🎯 Temporal Context Available for Search:")
                    for norm in analysis['normalized'][:3]:  # Show first 3
                        start = norm.get('start')
                        end = norm.get('end')
                        precision = norm.get('precision')
                        method = norm.get('method', 'unknown')
                        print(f"      - '{norm['text']}' → {start}-{end} ({precision}) via {method}")
                        
                        # Show how this could enhance search
                        if precision == "decade" and isinstance(start, int):
                            decade_term = f"{start}s"
                            print(f"        → Could add search term: '{decade_term}'")
                        elif precision == "year":
                            print(f"        → Could add search term: '{start}'")
                else:
                    print(f"   ℹ️  No temporal context detected for search enhancement")
                
                # Check if query was actually enhanced
                if enhanced_query != test_case['query']:
                    print(f"   🚀 Query was enhanced!")
                else:
                    print(f"   ℹ️  Query returned unchanged (may be enhanced internally)")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def test_complex_query_expressions(self):
        """Test complex temporal expressions in user queries."""
        print("\n🧠 Testing Complex Temporal Query Expressions")
        print("=" * 70)
        
        complex_queries = [
            "movies from around the turn of the century",
            "sci-fi films from two decades before the millennium", 
            "action movies from approximately 2010",
            "horror films from the past few years",
            "classic movies from before color was common",
            "films from the golden age of Hollywood",
            "post-9/11 action movies",
            "pre-digital era monster movies",
            "movies from when DVDs were popular",
            "silent films from the early days"
        ]
        
        for query in complex_queries:
            print(f"\n🔬 Testing: '{query}'")
            
            try:
                context = PluginExecutionContext(metadata={"query_type": "complex_search"})
                
                # Test query enhancement
                enhanced_query = await self.plugin.embellish_query(query, context)
                
                # Analyze temporal content
                analysis = await self.plugin.analyze(query, context.metadata)
                
                expressions = len(analysis.get('expressions', []))
                normalized = len(analysis.get('normalized', []))
                methods = analysis.get('analysis_methods', [])
                
                print(f"   📊 Results: {expressions} expressions, {normalized} normalized")
                print(f"   🔧 Methods: {', '.join(methods) if methods else 'none'}")
                
                if expressions > 0:
                    print(f"   ✅ Temporal expressions detected")
                    for expr in analysis.get('expressions', [])[:2]:  # Show first 2
                        method = expr.get('method', 'unknown')
                        confidence = expr.get('confidence', 0)
                        print(f"      - '{expr['text']}' ({method}, conf: {confidence:.2f})")
                else:
                    print(f"   ℹ️  No temporal expressions detected (may be too complex for current methods)")
                
                if normalized > 0:
                    print(f"   🎯 Usable temporal data for search:")
                    for norm in analysis.get('normalized', [])[:2]:
                        print(f"      - {norm.get('text')} → {norm.get('start')}-{norm.get('end')}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def test_query_fallback_behavior(self):
        """Test query processing with missing dependencies."""
        print("\n⚠️  Testing Query Processing Fallback Behavior")
        print("=" * 70)
        
        fallback_queries = [
            "90s movies",
            "recent films", 
            "last decade movies",
            "2000s sci-fi"
        ]
        
        for query in fallback_queries:
            print(f"\n🔄 Testing fallback for: '{query}'")
            
            try:
                context = PluginExecutionContext(metadata={"test_fallback": True})
                
                # Test that query processing doesn't break even with limited capabilities
                enhanced_query = await self.plugin.embellish_query(query, context)
                analysis = await self.plugin.analyze(query, context.metadata)
                
                print(f"   ✅ Fallback processing successful")
                print(f"   📝 Enhanced Query: '{enhanced_query}'")
                
                # Check if fallback normalization worked
                normalized = analysis.get('normalized', [])
                if normalized:
                    print(f"   🎯 Fallback normalization: {len(normalized)} results")
                    for norm in normalized[:2]:
                        method = norm.get('method', 'unknown')
                        print(f"      - {norm.get('text')} via {method}")
                else:
                    print(f"   ℹ️  No normalization in fallback mode")
                
            except Exception as e:
                print(f"   ❌ Fallback error: {e}")


async def main():
    """Run query mode tests."""
    print("🧪 SpacyWithFallbackIngestionAndQueryPlugin - QUERY MODE Tests")
    print("=" * 80)
    print("Testing temporal enhancement of user search queries")
    
    tester = TestSpacyTemporalQueryMode()
    
    try:
        await tester.setup()
        await tester.test_user_query_enhancement()
        await tester.test_complex_query_expressions()
        await tester.test_query_fallback_behavior()
        
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Query mode testing completed!")
    print("📝 Run with: python test_spacy_temporal_query_mode.py")


if __name__ == "__main__":
    asyncio.run(main())