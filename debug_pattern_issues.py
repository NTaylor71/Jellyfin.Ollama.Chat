#!/usr/bin/env python3
"""
Debug remaining pattern issues in SpacyWithFallbackIngestionAndQueryPlugin
"""

import asyncio

async def debug_specific_issues():
    """Debug specific failing cases"""
    
    from src.plugins.temporal.spacy_with_fallback_ingestion_and_query import SpacyWithFallbackIngestionAndQueryPlugin
    
    plugin = SpacyWithFallbackIngestionAndQueryPlugin()
    config = {"enabled": True, "debug": True}
    await plugin.initialize(config)
    
    # Test specific failing cases
    failing_cases = [
        "80s and 90s science fiction",
        "post-millennium cinema", 
        "recent sci-fi films",
        "🎬 90s movies 🎭"  # This one worked but let's see why
    ]
    
    print("🔍 Debugging specific failing patterns")
    print("="*60)
    
    for case in failing_cases:
        print(f"\n🧪 Testing: '{case}'")
        
        # Test each analysis method individually
        try:
            # spaCy analysis
            if plugin.nlp:
                doc = plugin.nlp(case)
                print(f"   spaCy entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
                
                if plugin.matcher:
                    matches = plugin.matcher(doc)
                    print(f"   spaCy matches: {matches}")
                    for match_id, start, end in matches:
                        span = doc[start:end]
                        label = plugin.nlp.vocab.strings[match_id]
                        print(f"      - {label}: '{span.text}'")
            
            # Full analysis
            result = await plugin.analyze(case, {})
            print(f"   Total expressions: {len(result.get('expressions', []))}")
            print(f"   Total normalized: {len(result.get('normalized', []))}")
            
            if result.get('expressions'):
                for expr in result['expressions']:
                    print(f"      - '{expr['text']}' ({expr['method']})")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test pattern setup directly
    print(f"\n🔧 Pattern Setup Debug")
    print("="*30)
    
    if plugin.matcher:
        try:
            # Try to manually check patterns
            print(f"Matcher vocab size: {len(plugin.matcher.vocab)}")
            
            # Test a simple pattern manually
            from spacy.matcher import Matcher
            test_matcher = Matcher(plugin.nlp.vocab)
            
            # Add a simple test pattern
            test_pattern = [[{"LIKE_NUM": True}, {"LOWER": "s"}]]
            test_matcher.add("TEST_DECADE", test_pattern)
            
            test_doc = plugin.nlp("90s movies")
            test_matches = test_matcher(test_doc)
            print(f"Test pattern matches: {test_matches}")
            
        except Exception as e:
            print(f"Pattern debug error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_specific_issues())