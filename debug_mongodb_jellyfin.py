"""
Debug script for MongoDB + Jellyfin integration.
Run this directly on the host to test connections before/after docker setup.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.shared.config import get_settings
    from src.data.mongo_client import MongoClient
    from src.ingestion.jellyfin_connector import JellyfinConnector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Run 'pip install -e .' or 'dev_setup.ps1' first")
    sys.exit(1)


def print_env_check():
    """Check environment variables."""
    print("🔧 Environment Variables Check:")
    print(f"   JELLYFIN_URL: {os.getenv('JELLYFIN_URL', 'Not set')}")
    print(f"   JELLYFIN_API_KEY: {'✅ Set' if os.getenv('JELLYFIN_API_KEY') else '❌ Missing'}")
    print(f"   JELLYFIN_USER_ID: {os.getenv('JELLYFIN_USER_ID', 'Not set')}")
    print(f"   ENV: {os.getenv('ENV', 'localhost')}")
    print()


async def test_config():
    """Test configuration loading."""
    print("📋 Testing Configuration...")
    try:
        settings = get_settings()
        print(f"   MongoDB URL: {settings.mongodb_url}")
        print(f"   Jellyfin URL: {settings.JELLYFIN_URL}")
        print(f"   Environment: {settings.ENV}")
        print("✅ Configuration loaded")
        return settings
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return None


async def test_jellyfin_only(settings):
    """Test just Jellyfin connection (no MongoDB)."""
    print("🎬 Testing Jellyfin Connection...")
    
    try:
        import httpx
        
        url = f"{settings.JELLYFIN_URL}/System/Info"
        headers = {"X-Emby-Token": settings.JELLYFIN_API_KEY}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            system_info = response.json()
            print(f"   Server: {system_info.get('ServerName', 'Unknown')}")
            print(f"   Version: {system_info.get('Version', 'Unknown')}")
            
            # Test getting movies
            movies_url = f"{settings.JELLYFIN_URL}/Items"
            params = {
                "IncludeItemTypes": "Movie",
                "Recursive": True,
                "Limit": 3,
                "StartIndex": 0,
                "Fields": "Overview,Genres,People,ProductionYear"
            }
            
            if settings.JELLYFIN_USER_ID:
                params["UserId"] = settings.JELLYFIN_USER_ID
            
            response = await client.get(movies_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            movies = data.get("Items", [])
            print(f"   Found {len(movies)} movies (sample)")
            
            for movie in movies:  # Show all movies
                print(f"   - {movie.get('Name', 'Unknown')} ({movie.get('ProductionYear', '?')})")
            
        print("✅ Jellyfin connection successful")
        return True, len(movies)
        
    except Exception as e:
        print(f"❌ Jellyfin connection failed: {e}")
        return False, 0


async def test_mongodb_only(settings):
    """Test just MongoDB connection."""
    print("🍃 Testing MongoDB Connection...")
    
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        
        client = AsyncIOMotorClient(settings.mongodb_url)
        
        # Test ping
        await client.admin.command('ping')
        
        # Test database access
        db = client[settings.MONGODB_DATABASE]
        collections = await db.list_collection_names()
        print(f"   Database: {settings.MONGODB_DATABASE}")
        print(f"   Collections: {collections if collections else 'None (new database)'}")
        
        client.close()
        print("✅ MongoDB connection successful")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("💡 Make sure MongoDB is running (docker-compose up or local MongoDB)")
        return False


async def main():
    """Main test function."""
    print("🔍 MongoDB + Jellyfin Debug Test")
    print("=" * 40)
    
    # Check environment
    print_env_check()
    
    # Test configuration
    settings = await test_config()
    if not settings:
        return
    
    print()
    
    # Test Jellyfin (works without MongoDB)
    jellyfin_ok, movie_count = await test_jellyfin_only(settings)
    
    print()
    
    # Test MongoDB (works without Jellyfin)
    mongodb_ok = await test_mongodb_only(settings)
    
    print()
    print("=" * 40)
    print("📊 Results Summary:")
    print(f"   Jellyfin: {'✅ Working' if jellyfin_ok else '❌ Failed'}")
    print(f"   MongoDB: {'✅ Working' if mongodb_ok else '❌ Failed'}")
    
    if jellyfin_ok:
        print(f"   Movies available: {movie_count}")
    
    if jellyfin_ok and mongodb_ok:
        print("\n🎉 Ready to test full integration!")
        print("   Next: Run test_mongodb_jellyfin_integration.py")
    else:
        print("\n🔧 Fix the failed connections before proceeding.")


if __name__ == "__main__":
    asyncio.run(main())