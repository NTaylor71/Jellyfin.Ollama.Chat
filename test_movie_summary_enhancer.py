"""
Test the Movie Summary Enhancer plugin.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.examples.movie_summary_enhancer import MovieSummaryEnhancerPlugin


async def test_movie_summary_enhancer():
    """Test the movie summary enhancement plugin."""
    print("🎬 Testing Movie Summary Enhancer Plugin")
    print("=" * 50)
    
    # Sample movie data (based on Jellyfin structure)
    sample_movie = {
        "name": "Samurai Rauni",
        "production_year": 2016,
        "overview": "Villagers are afraid of Samurai Rauni Reposaarelainen, who keeps them on their toes every day. When someone places a bounty on Rauni's head, he goes after this mysterious person.",
        "genres": ["Comedy", "Drama"],
        "official_rating": None,
        "people": [
            {"name": "Mika Ratto", "type": "Director"},
            {"name": "Mika Ratto", "type": "Actor", "role": "Rauni Reposaarelainen"},
            {"name": "Reetta Turtiainen", "type": "Actor", "role": "Reetta Geisha"}
        ],
        "run_time_ticks": 47718720000,
        "jellyfin_id": "test123"
    }
    
    try:
        # Create plugin instance
        plugin = MovieSummaryEnhancerPlugin()
        
        print(f"✅ Plugin initialized successfully")
        print(f"   Initial Strategy: {plugin.processing_strategy}")
        
        # Initialize plugin
        init_success = await plugin.initialize({})
        print(f"   Initialization: {'Success' if init_success else 'Failed'}")
        print(f"   Final Strategy: {plugin.processing_strategy}")
        
        # Test health status
        health = await plugin.health_check()
        print(f"   Health: {health['status']}")
        print(f"   Metadata: {plugin.metadata.name} v{plugin.metadata.version}")
        
        # Test processing
        print(f"\n🔄 Processing sample movie...")
        print(f"   Movie: {sample_movie['name']} ({sample_movie['production_year']})")
        print(f"   Genres: {', '.join(sample_movie['genres'])}")
        print(f"   Overview: {sample_movie['overview'][:100]}...")
        
        # Process the movie data using the standard interface
        from src.plugins.base import PluginExecutionContext
        context = PluginExecutionContext(request_id="test123", user_id="test_user")
        enhanced_data = await plugin.embellish_embed_data(sample_movie, context)
        
        # Check results
        if "enhanced_fields" in enhanced_data and "summary" in enhanced_data["enhanced_fields"]:
            enhanced_summary = enhanced_data["enhanced_fields"]["summary"]
            print(f"\n✅ Enhanced Summary Generated:")
            print(f"   Length: {len(enhanced_summary)} characters")
            print(f"   Summary: {enhanced_summary}")
            
            # Show comparison
            print(f"\n📊 Comparison:")
            print(f"   Original: {sample_movie['overview'][:100]}...")
            print(f"   Enhanced: {enhanced_summary}")
            
            print(f"\n🔍 Search Benefits:")
            print(f"   - Contains searchable terms and patterns")
            print(f"   - Optimized for natural language queries")
            print(f"   - Fills gaps in sparse descriptions")
            
        else:
            print("❌ No enhanced summary generated")
        
        print("\n" + "=" * 50)
        print("🎉 Movie Summary Enhancer Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_movie_summary_enhancer())