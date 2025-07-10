#!/usr/bin/env python3
"""
Test script for the generic media ingestion system.
Tests the ingestion manager with the example movie data.
"""

import asyncio
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion_manager import IngestionManager


async def test_movie_ingestion():
    """Test movie ingestion from the example data file."""
    print("🎬 Testing Movie Ingestion System")
    print("=" * 50)
    
    try:

        async with IngestionManager(media_type="movie") as manager:
            print(f"✅ Connected to MongoDB")
            print(f"📋 Loaded configuration: {manager.media_config.name}")
            print(f"🎯 Media type: {manager.media_type}")
            print(f"🏗️  Dynamic model: {manager.dynamic_model.__name__}")
            

            data_file = Path("test_movie_data.json")
            if not data_file.exists():
                print(f"❌ Sample data file not found: {data_file}")
                return
                
            print(f"\n📂 Loading data from: {data_file}")
            movies = await manager.load_media_from_json(data_file)
            print(f"✅ Loaded {len(movies)} movies")
            

            if movies:
                sample = movies[0]
                print(f"\n📋 Sample movie: {sample.Name}")
                print(f"   ID: {sample.Id}")
                print(f"   Type: {getattr(sample, 'Type', 'N/A')}")
                print(f"   Year: {getattr(sample, 'ProductionYear', 'N/A')}")
                

            if movies:
                print(f"\n🔄 Testing enrichment on: {movies[0].Name}")
                enriched = await manager.enrich_media_item(movies[0])
                print(f"✅ Enrichment completed")
                print(f"📊 Original fields: {len(movies[0].model_dump())}")
                print(f"📊 Enriched fields: {len(enriched)}")
                

                new_fields = set(enriched.keys()) - set(movies[0].model_dump().keys())
                if new_fields:
                    print(f"🆕 New fields: {', '.join(list(new_fields)[:5])}")
                

            print(f"\n💾 Testing ingestion (first movie only)")
            await manager.ingest_media(
                movies[:1], 
                batch_size=1, 
                skip_enrichment=False
            )
            print(f"✅ Ingestion completed")
            

            print(f"\n🔍 Verifying ingestion...")
            results = await manager.verify_ingestion()
            print(f"✅ Verification results:")
            print(f"   Media type: {results['media_type']}")
            print(f"   Total items: {results['total']}")
            print(f"   {results['media_type']} items: {results['media_type_total']}")
            print(f"   Enriched items: {results['enriched']}")
            
            if results['sample_fields']:
                print(f"   Sample fields: {', '.join(results['sample_fields'][:10])}")
                
        print(f"\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_config_loading():
    """Test loading different media type configurations."""
    print("\n📁 Testing Configuration Loading")
    print("=" * 40)
    
    config_dir = Path("config/media_types")
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return
        
    for config_file in config_dir.glob("*.yaml"):
        if config_file.name.startswith("movie_new_format"):
            continue
            
        media_type = config_file.stem
        print(f"\n🔧 Testing {media_type} configuration...")
        
        try:
            manager = IngestionManager(media_type=media_type)
            await manager.connect()
            
            print(f"   ✅ {manager.media_config.name}")
            print(f"   📋 Fields: {len(manager.media_config.fields)}")
            print(f"   🏗️  Model: {manager.dynamic_model.__name__}")
            
            await manager.disconnect()
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


async def main():
    """Run all tests."""
    print("🧪 Generic Media Ingestion System Test")
    print("=" * 60)
    

    await test_config_loading()
    

    await test_movie_ingestion()
    
    print(f"\n✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())