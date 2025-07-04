"""
Reset MongoDB database after schema changes.
Run this when field names change (like title -> name).
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.mongo_client import get_mongo_client, close_mongo_client


async def reset_database():
    """Clear all movies from MongoDB."""
    print("🗑️  Resetting MongoDB database...")
    
    try:
        mongo_client = await get_mongo_client()
        
        # Count existing movies
        count_before = await mongo_client.count_movies()
        print(f"   Movies before reset: {count_before}")
        
        if count_before > 0:
            # Clear all movies
            result = await mongo_client.db.movies.delete_many({})
            print(f"   Deleted {result.deleted_count} movies")
        
        # Verify empty
        count_after = await mongo_client.count_movies()
        print(f"   Movies after reset: {count_after}")
        
        print("✅ Database reset complete")
        
    except Exception as e:
        print(f"❌ Database reset failed: {e}")
    
    finally:
        await close_mongo_client()


if __name__ == "__main__":
    asyncio.run(reset_database())