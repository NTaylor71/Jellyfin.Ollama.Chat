#!/usr/bin/env python3
"""
Debug script to identify why spaCy temporal patterns aren't working
"""

import asyncio
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

async def debug_spacy_temporal():
    """Debug the spaCy temporal plugin initialization"""
    
    try:
        from src.plugins.temporal.spacy_with_fallback_ingestion_and_query import SpacyWithFallbackIngestionAndQueryPlugin
        
        plugin = SpacyWithFallbackIngestionAndQueryPlugin()
        
        print("🔍 Debugging SpacyWithFallbackIngestionAndQueryPlugin")
        print("="*60)
        
        # Check initial state
        print(f"Initial nlp: {plugin.nlp}")
        print(f"Initial matcher: {plugin.matcher}")
        
        # Try initialization
        config = {"enabled": True, "debug": True}
        success = await plugin.initialize(config)
        print(f"Initialization success: {success}")
        
        # Check post-initialization state
        print(f"Post-init nlp: {plugin.nlp}")
        print(f"Post-init matcher: {plugin.matcher}")
        
        if plugin.matcher:
            print(f"Matcher patterns: {list(plugin.matcher._patterns.keys()) if hasattr(plugin.matcher, '_patterns') else 'No _patterns attr'}")
            
            # Try to get pattern info
            try:
                import spacy.matcher
                print(f"Matcher type: {type(plugin.matcher)}")
                if hasattr(plugin.matcher, 'get'):
                    for key in ["DECADE", "RELATIVE", "SEASON"]:
                        patterns = plugin.matcher.get(key)
                        print(f"Patterns for {key}: {patterns}")
            except Exception as e:
                print(f"Error getting patterns: {e}")
        
        # Test a simple query
        test_query = "90s action movies"
        print(f"\n🧪 Testing query: '{test_query}'")
        
        try:
            result = await plugin.analyze(test_query, {})
            print(f"Analysis result: {result}")
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test the pattern setup directly
        print(f"\n🔧 Testing pattern setup directly...")
        if plugin.nlp and plugin.matcher:
            try:
                plugin._setup_temporal_patterns()
                print("Pattern setup called successfully")
                
                # Test with a document
                doc = plugin.nlp(test_query)
                print(f"Doc created: {doc}")
                
                matches = plugin.matcher(doc)
                print(f"Matches found: {matches}")
                
            except Exception as e:
                print(f"Pattern setup error: {e}")
                import traceback
                traceback.print_exc()
        
        return success
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_spacy_temporal())
    print(f"\nDebug result: {'SUCCESS' if success else 'FAILED'}")