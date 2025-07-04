"""
Debug script to see what fields are being extracted from Jellyfin.
"""

import asyncio
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion.jellyfin_connector import JellyfinConnector


async def debug_extraction():
    """Debug what fields we're getting from Jellyfin."""
    print("🔍 Debugging Jellyfin Data Extraction")
    print("=" * 50)
    
    try:
        connector = JellyfinConnector()
        await connector.connect()
        
        # Get one movie
        movies = await connector.get_movies(limit=1, start_index=0)
        
        if not movies:
            print("❌ No movies found")
            return
            
        raw_movie = movies[0]
        print(f"📋 Raw Jellyfin Data for: {raw_movie.get('Name', 'Unknown')}")
        print("=" * 50)
        
        # Show key fields we're interested in
        fields_to_check = [
            'Name', 'OriginalTitle', 'Overview', 'ProductionYear',
            'Taglines', 'People', 'Genres', 'OfficialRating', 
            'CommunityRating', 'CriticRating', 'RunTimeTicks',
            'MediaStreams', 'ProviderIds'
        ]
        
        for field in fields_to_check:
            value = raw_movie.get(field)
            if value is not None:
                if isinstance(value, list) and len(value) > 0:
                    print(f"{field}: {value[:2]} ({'...' if len(value) > 2 else ''})")
                elif isinstance(value, dict):
                    print(f"{field}: {dict(list(value.items())[:3])}")
                else:
                    print(f"{field}: {value}")
            else:
                print(f"{field}: None")
        
        print("\n" + "=" * 50)
        print("🔧 Extracted Movie Data:")
        print("=" * 50)
        
        # Extract using our function
        movie_data = connector._extract_movie_data(raw_movie)
        
        # Show extracted data
        extracted_dict = movie_data.dict()
        for key, value in extracted_dict.items():
            if value:
                if isinstance(value, list) and len(value) > 0:
                    print(f"{key}: {value[:3]} ({'...' if len(value) > 3 else ''})")
                else:
                    print(f"{key}: {value}")
            else:
                print(f"{key}: None/Empty")
        
        await connector.disconnect()
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")


if __name__ == "__main__":
    asyncio.run(debug_extraction())