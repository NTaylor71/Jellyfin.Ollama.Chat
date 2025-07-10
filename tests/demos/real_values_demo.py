#!/usr/bin/env python3
"""
REAL VALUES DEMONSTRATION
Shows actual data values at every step using config/media_types/movie.yaml
"""

import asyncio
import json
import sys
from pathlib import Path
from pprint import pprint


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion_manager import IngestionManager


def show_values(title: str, data, highlight_fields=None):
    """Show actual data values clearly."""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            marker = "🔥" if highlight_fields and key in highlight_fields else "  "
            if isinstance(value, (list, dict)) and len(str(value)) > 80:
                print(f"{marker} {key}: {type(value).__name__} with {len(value)} items")
                if isinstance(value, list) and value:
                    print(f"     First item: {value[0]}")
                elif isinstance(value, dict) and value:
                    first_key = list(value.keys())[0]
                    print(f"     Sample: {first_key}: {value[first_key]}")
            else:
                print(f"{marker} {key}: {value}")
    else:
        print(data)


async def demonstrate_real_values():
    """Show real values flowing through the pipeline."""
    
    print("🎬 REAL VALUES PIPELINE DEMONSTRATION")
    print("Using config/media_types/movie.yaml for enrichment control")
    



    raw_movie = {
        "Name": "Blade Runner 2049",
        "OriginalTitle": "Blade Runner 2049",
        "Id": "blade-runner-2049", 
        "Overview": "Thirty years after the events of the first film, a new blade runner, LAPD Officer K, unearths a long-buried secret that has the potential to plunge what's left of society into chaos.",
        "Taglines": ["The key to the future is finally unearthed"],
        "Genres": ["Science Fiction", "Drama", "Thriller"],
        "ProductionYear": 2017,
        "CommunityRating": 8.0,
        "OfficialRating": "R",
        "Tags": ["cyberpunk", "dystopian", "artificial intelligence", "sequel"],
        "People": [
            {"Name": "Ryan Gosling", "Role": "K", "Type": "Actor"},
            {"Name": "Harrison Ford", "Role": "Rick Deckard", "Type": "Actor"},
            {"Name": "Ana de Armas", "Role": "Joi", "Type": "Actor"},
            {"Name": "Denis Villeneuve", "Role": "Director", "Type": "Director"},
            {"Name": "Hampton Fancher", "Role": "Writer", "Type": "Writer"}
        ],
        "Studios": [{"Name": "Warner Bros. Pictures"}, {"Name": "Sony Pictures"}],
        "ProductionLocations": ["United States", "Canada", "Hungary"]
    }
    
    show_values("RAW INPUT DATA", raw_movie)
    
    async with IngestionManager(media_type="movie") as manager:
        



        print(f"\n{'='*60}")
        print("📄 YAML CONFIGURATION DRIVING ENRICHMENT")
        print(f"{'='*60}")
        print(f"Config file: config/media_types/movie.yaml")
        print(f"Media type: {manager.media_config.media_type}")
        print(f"Collection: {manager.media_config.output.get('collection', 'default') if manager.media_config.output else 'default'}")
        

        print(f"\n🏷️  FIELD WEIGHTS (from YAML):")
        if manager.media_config.field_weights:
            for field, weight in manager.media_config.field_weights.items():
                print(f"   {field}: {weight}")
        

        print(f"\n🔧 ENRICHMENT FIELDS (from YAML):")
        enrichment_fields = []
        if manager.media_config.fields:
            for field_name, config in manager.media_config.fields.items():
                source = config.get('source_field', 'SYNTHETIC')
                enrichments = len(config.get('enrichments', []))
                enrichment_fields.append(field_name)
                print(f"   {field_name}: source='{source}', enrichments={enrichments}")
        



        validated_movie = manager.dynamic_model(**raw_movie)
        validated_data = validated_movie.model_dump()
        
        show_values("AFTER DYNAMIC VALIDATION", validated_data)
        



        before_computed = validated_data.copy()
        manager._add_computed_fields(validated_data)
        

        computed_fields = {k: v for k, v in validated_data.items() if k not in before_computed}
        if computed_fields:
            show_values("COMPUTED FIELDS ADDED", computed_fields, computed_fields.keys())
        
        show_values("AFTER COMPUTED FIELDS", validated_data, computed_fields.keys())
        



        print(f"\n{'='*60}")
        print("🔄 FIELD-BY-FIELD ENRICHMENT (YAML-driven)")
        print(f"{'='*60}")
        
        enriched_data = validated_data.copy()
        

        for field_name in enrichment_fields:
            field_config = manager.media_config.fields[field_name]
            
            print(f"\n🎯 PROCESSING: {field_name}")
            print(f"   Source field: {field_config.get('source_field', 'SYNTHETIC')}")
            print(f"   Field weight: {field_config.get('field_weight', 'default')}")
            print(f"   Type: {field_config.get('type', 'unknown')}")
            

            source_field = field_config.get('source_field')
            if source_field and source_field in validated_data:
                original_value = validated_data[source_field]
                print(f"   Original value: {original_value}")
            else:
                print(f"   Original value: None (synthetic field)")
            

            try:
                enriched_value = await manager._process_field_enrichments(
                    field_name, field_config, validated_data
                )
                enriched_data[field_name] = enriched_value
                print(f"   ✅ Enriched value: {enriched_value}")
                
            except Exception as e:
                print(f"   ❌ Enrichment failed: {e}")
        



        show_values("FULLY ENRICHED DATA", enriched_data, enrichment_fields)
        



        print(f"\n{'='*60}")
        print("📊 BEFORE vs AFTER VALUE COMPARISON")
        print(f"{'='*60}")
        
        all_fields = set(validated_data.keys()) | set(enriched_data.keys())
        for field in sorted(all_fields):
            before = validated_data.get(field, "❌ Not present")
            after = enriched_data.get(field, "❌ Not present")
            
            if field in enrichment_fields:
                status = "🔥 ENRICHED"
            elif field in computed_fields:
                status = "🧮 COMPUTED"
            else:
                status = "📋 ORIGINAL"
                
            print(f"\n{status} {field}:")
            print(f"   Before: {before}")
            print(f"   After:  {after}")
            
            if before != after:
                print(f"   🔄 CHANGED!")
        



        print(f"\n{'='*60}")
        print("💾 MONGODB STORAGE WITH WEIGHTS")
        print(f"{'='*60}")
        

        storage_data = enriched_data.copy()
        storage_data["_ingested_at"] = "2025-07-08T02:00:00"
        storage_data["_media_type"] = "movie"
        storage_data["_enrichment_version"] = "2.0"
        storage_data["_field_weights"] = manager.media_config.field_weights
        

        print(f"Collection: {manager.media_config.output.get('collection', 'movies_enriched') if manager.media_config.output else 'movies_enriched'}")
        print(f"Total fields to store: {len(storage_data)}")
        print(f"Field weights applied: {len(manager.media_config.field_weights) if manager.media_config.field_weights else 0}")
        

        metadata_fields = {k: v for k, v in storage_data.items() if k.startswith("_")}
        show_values("METADATA BEING ADDED", metadata_fields)
        

        collection_name = manager.media_config.output.get("collection", "movies_enriched") if manager.media_config.output else "movies_enriched"
        collection = manager.db[collection_name]
        
        result = await collection.replace_one(
            {"Id": storage_data["Id"]},
            storage_data,
            upsert=True
        )
        
        print(f"\n✅ STORED in MongoDB")
        print(f"   Operation: {'INSERT' if result.upserted_id else 'UPDATE'}")
        print(f"   Document ID: {result.upserted_id or 'existing'}")
        



        retrieved = await collection.find_one({"Id": storage_data["Id"]})
        
        if retrieved:

            retrieved["_id"] = str(retrieved["_id"])
            
            print(f"\n{'='*60}")
            print("🔍 RETRIEVED FROM MONGODB")
            print(f"{'='*60}")
            

            key_fields = ["Name", "Overview", "Genres", "ProductionYear", "CommunityRating"]
            print(f"\n🔍 KEY FIELD VERIFICATION:")
            for field in key_fields:
                original = raw_movie.get(field, "N/A")
                retrieved_val = retrieved.get(field, "N/A")
                match = "✅" if original == retrieved_val else "❌"
                print(f"   {field}: {match} {retrieved_val}")
            

            enriched_fields_in_db = {k: v for k, v in retrieved.items() if k in enrichment_fields}
            show_values("ENRICHED FIELDS IN DATABASE", enriched_fields_in_db)
            

            if "_field_weights" in retrieved:
                show_values("FIELD WEIGHTS STORED", retrieved["_field_weights"])
        



        print(f"\n{'='*60}")
        print("🎉 PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        print(f"Input fields: {len(raw_movie)}")
        print(f"Validated fields: {len(validated_data)}")
        print(f"Computed fields added: {len(computed_fields)}")
        print(f"Enrichment fields processed: {len(enrichment_fields)}")
        print(f"Final stored fields: {len(storage_data)}")
        print(f"Successfully retrieved: {'✅' if retrieved else '❌'}")
        
        print(f"\n📋 YAML CONFIG CONTROLLED:")
        print(f"   ✅ Field weights: {len(manager.media_config.field_weights) if manager.media_config.field_weights else 0} fields")
        print(f"   ✅ Enrichment rules: {len(manager.media_config.fields) if manager.media_config.fields else 0} fields")
        print(f"   ✅ Validation constraints: {len(manager.media_config.validation.get('field_constraints', {})) if manager.media_config.validation else 0} fields")
        print(f"   ✅ Output collection: {manager.media_config.output.get('collection', 'default') if manager.media_config.output else 'default'}")


async def main():
    await demonstrate_real_values()


if __name__ == "__main__":
    asyncio.run(main())