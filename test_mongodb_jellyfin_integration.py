"""
Test MongoDB + Jellyfin integration with real credentials.
Run this after dev_setup.ps1 and docker-compose up to validate Phase 5.1.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.config import get_settings
from src.data.mongo_client import get_mongo_client, close_mongo_client
from src.ingestion.jellyfin_connector import JellyfinConnector


async def test_configuration():
    """Test that configuration is loaded correctly."""
    print("🔧 Testing Configuration...")
    
    settings = get_settings()
    
    print(f"   MongoDB URL: {settings.mongodb_url}")
    print(f"   Jellyfin URL: {settings.JELLYFIN_URL}")
    print(f"   Jellyfin API Key: {'✅ Set' if settings.JELLYFIN_API_KEY else '❌ Missing'}")
    print(f"   Jellyfin User ID: {settings.JELLYFIN_USER_ID or 'Not set'}")
    
    if not settings.JELLYFIN_API_KEY:
        print("❌ JELLYFIN_API_KEY is required. Please add to .env file.")
        return False
    
    print("✅ Configuration loaded successfully")
    return True


async def test_mongodb_connection():
    """Test MongoDB connection."""
    print("\n🍃 Testing MongoDB Connection...")
    
    try:
        mongo_client = await get_mongo_client()
        
        # Test basic operations
        count = await mongo_client.count_movies()
        print(f"   Current movie count: {count}")
        
        print("✅ MongoDB connection successful")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False


async def test_jellyfin_connection():
    """Test Jellyfin connection."""
    print("\n🎬 Testing Jellyfin Connection...")
    
    try:
        connector = JellyfinConnector()
        await connector.connect()
        
        # Test getting a small batch of movies  
        movies = await connector.get_movies(limit=5, start_index=0)
        print(f"   Found {len(movies)} movies in Jellyfin (sample batch)")
        
        if movies:
            sample_movie = movies[0]
            print(f"   Sample movie: {sample_movie.get('Name', 'Unknown')}")
            print(f"   Sample year: {sample_movie.get('ProductionYear', 'Unknown')}")
            print(f"   Sample genres: {sample_movie.get('Genres', [])}")
        
        print("✅ Jellyfin connection successful")
        return True, len(movies), connector
        
    except Exception as e:
        print(f"❌ Jellyfin connection failed: {e}")
        return False, 0, None


async def test_movie_ingestion(limit=5, connector=None):
    """Test ingesting a small number of movies."""
    print(f"\n📥 Testing Movie Ingestion (limit={limit})...")
    
    try:
        # Check existing count before ingestion
        mongo_client = await get_mongo_client()
        initial_count = await mongo_client.count_movies()
        print(f"   Movies in database before ingestion: {initial_count}")
        
        # Use provided connector or create new one
        if connector is None:
            connector = JellyfinConnector()
            await connector.connect()
            should_disconnect = True
        else:
            should_disconnect = False
        
        # Ingest a small batch
        ingested_movies = await connector.ingest_movies_batch(limit=limit, start_index=0)
        
        # Check final count
        final_count = await mongo_client.count_movies()
        new_movies = final_count - initial_count
        
        print(f"   Processed {len(ingested_movies)} movies from Jellyfin")
        print(f"   New movies inserted: {new_movies}")
        print(f"   Total movies in database: {final_count}")
        
        if new_movies < len(ingested_movies):
            updated = len(ingested_movies) - new_movies
            print(f"   ✅ {updated} existing movies updated with fresh data")
        
        print(f"\n   📋 Processed Movies:")
        for movie in ingested_movies:
            print(f"   - {movie.name} ({movie.production_year}) - {len(movie.genres)} genres, {len([p for p in movie.people if p.type == 'Actor'])} cast")
        
        if should_disconnect:
            await connector.disconnect()
        print("✅ Movie ingestion successful")
        return True, len(ingested_movies)
        
    except Exception as e:
        print(f"❌ Movie ingestion failed: {e}")
        return False, 0


async def test_mongodb_queries():
    """Test MongoDB query operations and display stored data."""
    print("\n🔍 Testing MongoDB Queries & Data Retrieval...")
    
    try:
        mongo_client = await get_mongo_client()
        
        # Test basic queries
        total_count = await mongo_client.count_movies()
        print(f"   Total movies in database: {total_count}")
        
        if total_count > 0:
            # Get all movies and show detailed data
            movies = await mongo_client.list_movies(limit=total_count)
            print(f"\n📋 All Movies Retrieved from MongoDB:")
            print("   " + "="*60)
            
            for i, movie in enumerate(movies, 1):
                print(f"   {i}. {movie.name} ({movie.production_year or 'Unknown'})")
                
                # Core info
                if movie.original_title and movie.original_title != movie.name:
                    print(f"      Original Title: {movie.original_title}")
                if movie.taglines:
                    print(f"      Taglines: {', '.join(movie.taglines)}")
                print(f"      Overview: {(movie.overview[:100] + '...') if movie.overview and len(movie.overview) > 100 else movie.overview or 'No overview'}")
                
                # People (extract from people list)
                directors = [p.name for p in movie.people if p.type == 'Director']
                actors = [p.name for p in movie.people if p.type == 'Actor']
                writers = [p.name for p in movie.people if p.type == 'Writer']
                producers = [p.name for p in movie.people if p.type == 'Producer']
                
                print(f"      Directors: {', '.join(directors) if directors else 'None'}")
                print(f"      Actors: {', '.join(actors[:4]) if actors else 'None'}{f' (+{len(actors)-4} more)' if len(actors) > 4 else ''}")
                if writers:
                    print(f"      Writers: {', '.join(writers[:3])}{f' (+{len(writers)-3} more)' if len(writers) > 3 else ''}")
                if producers:
                    print(f"      Producers: {', '.join(producers[:2])}{f' (+{len(producers)-2} more)' if len(producers) > 2 else ''}")
                
                # Classification & ratings
                print(f"      Genres: {', '.join(movie.genres) if movie.genres else 'None'}")
                print(f"      Official Rating: {movie.official_rating or 'Not rated'}")
                if movie.community_rating:
                    print(f"      Community Rating: {movie.community_rating}/10")
                if movie.critic_rating:
                    print(f"      Critic Rating: {movie.critic_rating}/10")
                if movie.run_time_ticks:
                    runtime_minutes = int(movie.run_time_ticks / 600000000)
                    print(f"      Runtime: {runtime_minutes} minutes")
                
                # External IDs
                print(f"      Jellyfin ID: {movie.jellyfin_id}")
                if movie.provider_ids.get('Imdb'):
                    print(f"      IMDB ID: {movie.provider_ids['Imdb']}")
                if movie.provider_ids.get('Tmdb'):
                    print(f"      TMDB ID: {movie.provider_ids['Tmdb']}")
                
                # Enhanced fields (LLM-generated)
                if movie.enhanced_fields:
                    print(f"      🤖 Enhanced Fields:")
                    for field_name, field_value in movie.enhanced_fields.items():
                        if isinstance(field_value, str):
                            display_value = field_value[:100] + '...' if len(field_value) > 100 else field_value
                            print(f"         {field_name}: {display_value}")
                        else:
                            print(f"         {field_name}: {field_value}")
                
                # Search & metadata
                print(f"      Search Keywords: {', '.join(movie.search_keywords[:6])}{' (+more)' if len(movie.search_keywords) > 6 else ''}")
                print(f"      Created: {movie.created_at.strftime('%Y-%m-%d %H:%M')}")
                print(f"      Updated: {movie.updated_at.strftime('%Y-%m-%d %H:%M')}")
                
                if i < len(movies):
                    print()
            
            print("   " + "="*60)
            
            # Test search functionality
            print(f"\n🔍 Testing Search Functionality:")
            
            # Text search
            search_results = await mongo_client.search_movies("movie", limit=5)
            print(f"   Text search for 'movie': {len(search_results)} results")
            
            # Genre search if we have movies with genres
            if any(movie.genres for movie in movies):
                first_genre = next((genre for movie in movies for genre in movie.genres), None)
                if first_genre:
                    genre_results = await mongo_client.get_movies_by_genre(first_genre, limit=5)
                    print(f"   Genre search for '{first_genre}': {len(genre_results)} results")
            
            # Year search if we have movies with years
            if any(movie.production_year for movie in movies):
                first_year = next((movie.production_year for movie in movies if movie.production_year), None)
                if first_year:
                    year_results = await mongo_client.get_movies_by_year(first_year, limit=5)
                    print(f"   Year search for {first_year}: {len(year_results)} results")
        
        print("✅ MongoDB queries and data retrieval successful")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB queries failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing MongoDB + Jellyfin Integration")
    print("=" * 50)
    
    try:
        # Test configuration
        config_ok = await test_configuration()
        if not config_ok:
            return
        
        # Test MongoDB
        mongo_ok = await test_mongodb_connection()
        if not mongo_ok:
            print("\n💡 Make sure MongoDB is running (docker-compose up should start it)")
            return
        
        # Test Jellyfin
        jellyfin_ok, movie_count, connector = await test_jellyfin_connection()
        if not jellyfin_ok:
            print("\n💡 Check your Jellyfin credentials in .env file")
            return
        
        if movie_count == 0:
            print("\n⚠️  No movies found in Jellyfin. Make sure your library has movies.")
            await connector.disconnect()
            return
        
        # Test movie ingestion (start small) - reuse the same connector
        print(f"\n📝 Ready to ingest movies. Starting with just 5 movies...")
        ingest_ok, ingested_count = await test_movie_ingestion(limit=5, connector=connector)
        
        # Now disconnect the connector
        await connector.disconnect()
        
        if ingest_ok and ingested_count > 0:
            # Test queries
            await test_mongodb_queries()
        
        print("\n" + "=" * 50)
        print("🎉 Phase 5.1 Integration Test Complete!")
        print(f"   Movies available in Jellyfin: {movie_count}")
        print(f"   Movies ingested to MongoDB: {ingested_count}")
        
        if ingest_ok:
            print("\n✅ Ready for larger ingestion. You can increase the limit in the script.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        
    finally:
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())