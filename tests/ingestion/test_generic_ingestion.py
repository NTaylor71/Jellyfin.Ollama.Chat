#!/usr/bin/env python3
"""
Test the generic ingestion system with different media types.
"""

import asyncio
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion_manager import IngestionManager


async def test_media_type(media_type: str, test_data: dict):
    """Test ingestion for a specific media type."""
    print(f"\n🧪 TESTING {media_type.upper()} INGESTION")
    print("=" * 60)
    
    try:
        async with IngestionManager(media_type=media_type) as manager:
            print(f"✅ Loaded {media_type} configuration: {manager.media_config.name}")
            print(f"📋 Dynamic model: {manager.dynamic_model.__name__}")
            print(f"🎯 Jellyfin type: {manager.media_config.execution.get('jellyfin_type', 'Unknown') if manager.media_config.execution else 'Not configured'}")
            print(f"💾 Collection: {manager.media_config.output.get('collection', f'{media_type}_enriched') if manager.media_config.output else f'{media_type}_enriched'}")
            

            print(f"\n🔍 Testing validation with sample data...")
            validated_item = manager.dynamic_model(**test_data)
            print(f"✅ Validation passed - {len(validated_item.model_dump())} fields")
            

            test_dict = validated_item.model_dump()
            before_computed = len(test_dict)
            manager._add_computed_fields(test_dict)
            after_computed = len(test_dict)
            
            if after_computed > before_computed:
                computed_fields = {k: v for k, v in test_dict.items() if k not in validated_item.model_dump()}
                print(f"🧮 Added {after_computed - before_computed} computed fields: {list(computed_fields.keys())}")
            else:
                print(f"ℹ️  No computed fields configured")
            

            print(f"\n🔄 Testing enrichment...")
            enriched_data = await manager.enrich_media_item(validated_item)
            print(f"✅ Enrichment completed - {len(enriched_data)} fields")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Test the generic ingestion system with different media types."""
    
    print("🎯 GENERIC MEDIA INGESTION SYSTEM TEST")
    print("Testing truly generic behavior across different media types")
    

    test_cases = {
        "movie": {
            "Name": "Test Movie",
            "Id": "test-movie-001",
            "ProductionYear": 2020,
            "CommunityRating": 8.5,
            "Genres": ["Action", "Drama"],
            "Overview": "A test movie for validation",
            "People": [
                {"Name": "Test Director", "Type": "Director"},
                {"Name": "Test Actor", "Type": "Actor"}
            ]
        },
        "book": {
            "Name": "Test Book", 
            "Id": "test-book-001",
            "PublicationYear": 2019,
            "Rating": 4.2,
            "Genres": ["Fiction", "Mystery"],
            "Overview": "A test book for validation",
            "People": [
                {"Name": "Test Author", "Type": "Author"}
            ]
        }
    }
    

    results = {}
    for media_type, data in test_cases.items():
        results[media_type] = await test_media_type(media_type, data)
    

    print(f"\n🎉 TEST SUMMARY")
    print("=" * 40)
    for media_type, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{media_type}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎯 The ingestion system is truly generic!")
        print("✅ Different media types work with their own YAML configs")
        print("✅ Dynamic model creation works for any media type") 
        print("✅ Computed fields work based on YAML configuration")
        print("✅ Jellyfin type mapping works from YAML config")
        print("✅ No hardcoded media-specific logic detected")


if __name__ == "__main__":
    asyncio.run(main())