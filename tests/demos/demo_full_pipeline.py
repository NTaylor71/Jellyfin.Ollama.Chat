#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Demonstration
Shows real values at every step: Input → Enrichment → MongoDB → Retrieval
"""

import asyncio
import json
import sys
from pathlib import Path
from pprint import pprint
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion_manager import IngestionManager


def print_section(title: str, data=None):
    """Pretty print a section with data."""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")
    if data is not None:
        if isinstance(data, (dict, list)):
            pprint(data, width=100, depth=3)
        else:
            print(data)


def show_field_comparison(original: dict, enriched: dict):
    """Show side-by-side field comparison."""
    print(f"\n📊 FIELD COMPARISON")
    print(f"{'Field':<25} {'Original':<30} {'Enriched':<30}")
    print(f"{'-'*25} {'-'*30} {'-'*30}")
    
    all_fields = set(original.keys()) | set(enriched.keys())
    for field in sorted(all_fields):
        orig_val = str(original.get(field, "N/A"))[:28]
        enriched_val = str(enriched.get(field, "N/A"))[:28]
        

        marker = "🆕" if field not in original else "  "
        print(f"{marker} {field:<23} {orig_val:<30} {enriched_val:<30}")


async def demonstrate_full_pipeline():
    """Run complete pipeline demonstration."""
    print("🎬 COMPLETE PIPELINE DEMONSTRATION")
    print("Raw Input → Dynamic Model → Enrichment → MongoDB → Retrieval")
    



    raw_input = {
        "Name": "The Matrix",
        "OriginalTitle": "The Matrix", 
        "Id": "demo-matrix-001",
        "Overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "Taglines": ["Welcome to the Real World"],
        "Genres": ["Action", "Science Fiction"],
        "ProductionYear": 1999,
        "CommunityRating": 8.7,
        "OfficialRating": "R",
        "Tags": ["cyberpunk", "philosophy", "martial arts", "virtual reality"],
        "People": [
            {"Name": "Keanu Reeves", "Role": "Neo", "Type": "Actor"},
            {"Name": "Laurence Fishburne", "Role": "Morpheus", "Type": "Actor"},
            {"Name": "Carrie-Anne Moss", "Role": "Trinity", "Type": "Actor"},
            {"Name": "Lana Wachowski", "Role": "Director", "Type": "Director"},
            {"Name": "Lilly Wachowski", "Role": "Director", "Type": "Director"}
        ],
        "Studios": [{"Name": "Warner Bros."}],
        "ProductionLocations": ["United States"]
    }
    
    print_section("STEP 1: RAW INPUT DATA", raw_input)
    



    async with IngestionManager(media_type="movie") as manager:
        
        print_section("STEP 2: CONFIGURATION LOADED")
        config_info = {
            "media_type": manager.media_type,
            "config_name": manager.media_config.name,
            "config_description": manager.media_config.description,
            "total_fields_defined": len(manager.media_config.fields),
            "field_list": list(manager.media_config.fields.keys()),
            "dynamic_model": manager.dynamic_model.__name__,
            "validation_rules": manager.media_config.validation.get("field_constraints", {}) if manager.media_config.validation else {},
            "output_collection": manager.media_config.output.get("collection", "default") if manager.media_config.output else "default"
        }
        print_section("Configuration Details", config_info)
        



        print_section("STEP 3: DYNAMIC MODEL VALIDATION")
        
        try:
            media_item = manager.dynamic_model(**raw_input)
            print("✅ Validation successful!")
            print(f"Model type: {type(media_item)}")
            print(f"Model fields: {len(media_item.model_dump())}")
            
            validated_data = media_item.model_dump()
            print_section("Validated Data", validated_data)
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return
            



        print_section("STEP 4: COMPUTED FIELDS ADDITION")
        

        before_computed = validated_data.copy()
        

        manager._add_computed_fields(validated_data)
        
        print("Computed fields added:")
        for field, value in validated_data.items():
            if field not in before_computed:
                print(f"  🆕 {field}: {value}")
        
        print_section("Data After Computed Fields", validated_data)
        



        print_section("STEP 5: FIELD-BY-FIELD ENRICHMENT PROCESS")
        
        enriched_data = validated_data.copy()
        enrichment_log = []
        
        if manager.media_config.fields:
            for field_name, field_config in manager.media_config.fields.items():
                print(f"\n🔄 Processing field: {field_name}")
                
                field_info = {
                    "field_name": field_name,
                    "source_field": field_config.get("source_field"),
                    "field_type": field_config.get("type"),
                    "field_weight": field_config.get("field_weight"),
                    "has_enrichments": "enrichments" in field_config,
                    "enrichment_count": len(field_config.get("enrichments", []))
                }
                
                print(f"  📋 Config: {field_info}")
                
                if "enrichments" in field_config:
                    try:
                        enriched_value = await manager._process_field_enrichments(
                            field_name, field_config, validated_data
                        )
                        enriched_data[field_name] = enriched_value
                        
                        enrichment_log.append({
                            "field": field_name,
                            "status": "success",
                            "original_value": validated_data.get(field_config.get("source_field")),
                            "enriched_value": enriched_value,
                            "enrichment_type": "synthetic" if field_config.get("source_field") is None else "enhanced"
                        })
                        
                        print(f"  ✅ Enriched: {str(enriched_value)[:100]}")
                        
                    except Exception as e:
                        enrichment_log.append({
                            "field": field_name,
                            "status": "error", 
                            "error": str(e)
                        })
                        print(f"  ❌ Error: {e}")
                else:
                    print(f"  ℹ️  No enrichments configured")
        
        print_section("ENRICHMENT LOG", enrichment_log)
        print_section("FULLY ENRICHED DATA", enriched_data)
        



        show_field_comparison(validated_data, enriched_data)
        



        print_section("STEP 7: MONGODB STORAGE")
        

        storage_data = enriched_data.copy()
        storage_data["_ingested_at"] = datetime.now().isoformat()
        storage_data["_media_type"] = manager.media_type
        storage_data["_enrichment_version"] = "2.0"
        storage_data["_demo_run"] = True
        
        collection_name = manager.media_config.output.get("collection", f"{manager.media_type}_enriched") if manager.media_config.output else f"{manager.media_type}_enriched"
        collection = manager.db[collection_name]
        
        print(f"📊 Storage Details:")
        print(f"  Collection: {collection_name}")
        print(f"  Database: {manager.db.name}")
        print(f"  Document size: {len(json.dumps(storage_data, default=str))} bytes")
        print(f"  Total fields: {len(storage_data)}")
        

        result = await collection.replace_one(
            {"Id": storage_data["Id"]},
            storage_data,
            upsert=True
        )
        
        storage_result = {
            "operation": "upsert",
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
            "acknowledged": result.acknowledged
        }
        
        print_section("Storage Result", storage_result)
        



        print_section("STEP 8: MONGODB RETRIEVAL & VERIFICATION")
        

        retrieved_doc = await collection.find_one({"Id": storage_data["Id"]})
        
        if retrieved_doc:

            if "_id" in retrieved_doc:
                retrieved_doc["_id"] = str(retrieved_doc["_id"])
                
            print("✅ Document successfully retrieved from MongoDB")
            print(f"📊 Retrieved document size: {len(json.dumps(retrieved_doc, default=str))} bytes")
            print(f"📊 Retrieved fields: {len(retrieved_doc)}")
            

            metadata_fields = {k: v for k, v in retrieved_doc.items() if k.startswith("_")}
            print_section("Metadata Fields", metadata_fields)
            

            print(f"\n🔍 ORIGINAL vs RETRIEVED COMPARISON")
            print(f"{'Field':<25} {'Match':<8} {'Original':<30} {'Retrieved':<30}")
            print(f"{'-'*25} {'-'*8} {'-'*30} {'-'*30}")
            
            for field in ["Name", "ProductionYear", "Genres", "Overview"]:
                orig_val = str(raw_input.get(field, "N/A"))[:28]
                retr_val = str(retrieved_doc.get(field, "N/A"))[:28]
                match = "✅" if raw_input.get(field) == retrieved_doc.get(field) else "❌"
                print(f"{field:<25} {match:<8} {orig_val:<30} {retr_val:<30}")
                

            synthetic_fields = {}
            for field_name, field_config in manager.media_config.fields.items():
                if field_config.get("source_field") is None and field_name in retrieved_doc:
                    synthetic_fields[field_name] = retrieved_doc[field_name]
                    
            if synthetic_fields:
                print_section("SYNTHETIC FIELDS CREATED", synthetic_fields)
            
        else:
            print("❌ Failed to retrieve document from MongoDB")
            



        print_section("STEP 9: COLLECTION STATISTICS")
        
        stats = await manager.verify_ingestion()
        print_section("Final Statistics", stats)
        

        all_docs = []
        async for doc in collection.find({"_demo_run": True}).limit(5):
            doc["_id"] = str(doc["_id"])
            all_docs.append({
                "Name": doc.get("Name"),
                "Id": doc.get("Id"), 
                "_ingested_at": doc.get("_ingested_at"),
                "field_count": len(doc)
            })
            
        print_section("DEMO DOCUMENTS IN COLLECTION", all_docs)
        



        print_section("STEP 10: ENRICHMENT ANALYSIS")
        
        analysis = {
            "original_field_count": len(raw_input),
            "final_field_count": len(retrieved_doc) if retrieved_doc else 0,
            "fields_added": len(retrieved_doc) - len(raw_input) if retrieved_doc else 0,
            "enrichment_success_rate": len([e for e in enrichment_log if e["status"] == "success"]) / len(enrichment_log) if enrichment_log else 0,
            "synthetic_fields_created": len(synthetic_fields),
            "data_size_increase": f"{len(json.dumps(retrieved_doc, default=str)) / len(json.dumps(raw_input)) * 100:.1f}%" if retrieved_doc else "N/A"
        }
        
        print_section("ENRICHMENT ANALYSIS", analysis)
        
    print_section("🎉 PIPELINE DEMONSTRATION COMPLETE!")
    print("The movie data has been successfully processed through the entire pipeline:")
    print("✅ Raw Input → Dynamic Validation → Enrichment → MongoDB Storage → Retrieval")


async def main():
    """Run the complete demonstration."""
    await demonstrate_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())