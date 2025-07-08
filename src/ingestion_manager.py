"""
Universal Media Ingestion Manager
Handles ingestion of any media type from multiple data sources (JSON files, APIs, databases) into MongoDB.
Uses YAML-based configuration for enrichment processing - completely data-driven and media-agnostic.
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
import yaml
from pydantic import BaseModel, Field, ValidationError, create_model

from src.shared.config import get_settings
import logging


class MediaData(BaseModel):
    """Generic media data model - dynamically created based on YAML config."""
    Name: str
    Id: str
    Type: Optional[str] = None  # Movie, Series, Episode, etc.
    
    class Config:
        extra = "allow"  # Allow any additional fields from the data source


class MediaTypeConfig(BaseModel):
    """Configuration for a media type loaded from YAML."""
    media_type: str
    name: str
    description: str
    field_weights: Optional[Dict[str, float]] = Field(default_factory=dict)
    fields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    weighting: Optional[Dict[str, Any]] = Field(default_factory=dict)
    execution: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority_order: Optional[List[str]] = Field(default_factory=list)
    output: Optional[Dict[str, Any]] = Field(default_factory=dict)
    validation: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IngestionManager:
    """Generic media ingestion manager with YAML-driven configuration."""
    
    def __init__(self, media_type: str = "movie"):
        self.logger = logging.getLogger(f"{__name__}.{media_type}")
        self.settings = get_settings()
        self.mongo_client = None
        self.db = None
        self.media_type = media_type
        self.media_config: Optional[MediaTypeConfig] = None
        self.dynamic_model: Optional[Type[BaseModel]] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self):
        """Connect to MongoDB and load configuration."""
        # Connect to MongoDB
        self.mongo_client = AsyncIOMotorClient(self.settings.mongodb_url)
        self.db = self.mongo_client[self.settings.MONGODB_DATABASE]
        
        # Test connection
        await self.db.command('ping')
        self.logger.info("Connected to MongoDB")
        
        # Load media type configuration
        config_path = Path(f"config/media_types/{self.media_type}.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Media type configuration not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            self.media_config = MediaTypeConfig(**config_data)
            
        # Create dynamic model based on validation rules
        self._create_dynamic_model()
            
        self.logger.info(f"Loaded {self.media_type} configuration: {self.media_config.name}")
        
    def _create_dynamic_model(self):
        """Create a dynamic Pydantic model based on YAML validation rules."""
        from pydantic import Field, validator
        from typing import Annotated
        
        field_definitions = {"Name": (str, ...), "Id": (str, ...)}  # Required base fields
        
        if self.media_config.validation and "field_constraints" in self.media_config.validation:
            constraints = self.media_config.validation["field_constraints"]
            
            for field_name, rules in constraints.items():
                field_type = rules.get("type", "str")
                is_required = field_name in self.media_config.validation.get("required_fields", [])
                
                # Map YAML types to Python types with constraints
                if field_type == "integer":
                    min_val = rules.get("min")
                    max_val = rules.get("max")
                    if min_val is not None and max_val is not None:
                        field_def = Annotated[int, Field(ge=min_val, le=max_val)]
                    elif min_val is not None:
                        field_def = Annotated[int, Field(ge=min_val)]
                    elif max_val is not None:
                        field_def = Annotated[int, Field(le=max_val)]
                    else:
                        field_def = int
                        
                elif field_type == "float":
                    min_val = rules.get("min")
                    max_val = rules.get("max")
                    if min_val is not None and max_val is not None:
                        field_def = Annotated[float, Field(ge=min_val, le=max_val)]
                    elif min_val is not None:
                        field_def = Annotated[float, Field(ge=min_val)]
                    elif max_val is not None:
                        field_def = Annotated[float, Field(le=max_val)]
                    else:
                        field_def = float
                        
                elif field_type == "list":
                    max_items = rules.get("max_items")
                    allowed_values = rules.get("allowed_values")
                    
                    if allowed_values:
                        # Use Literal for allowed values validation
                        from typing import Literal
                        literal_type = Literal[tuple(allowed_values)]
                        if max_items:
                            field_def = Annotated[List[literal_type], Field(max_length=max_items)]
                        else:
                            field_def = List[literal_type]
                    else:
                        if max_items:
                            field_def = Annotated[List[str], Field(max_length=max_items)]
                        else:
                            field_def = List[str]
                            
                elif field_type == "string":
                    field_def = str
                elif field_type == "dict":
                    field_def = Dict[str, Any]
                elif field_type == "boolean":
                    field_def = bool
                else:
                    field_def = str
                
                if is_required:
                    field_definitions[field_name] = (field_def, ...)
                else:
                    field_definitions[field_name] = (Optional[field_def], None)
        
        # Create the dynamic model
        self.dynamic_model = create_model(
            f"{self.media_type.capitalize()}Data",
            __base__=MediaData,
            **field_definitions
        )
        
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.mongo_client:
            self.mongo_client.close()
            self.logger.info("Disconnected from MongoDB")
            
    async def load_media_from_json(self, file_path: Union[str, Path]) -> List[MediaData]:
        """Load media data from JSON file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
            
        # Handle both .json and .py files
        if file_path.suffix == '.py':
            # For .py files, treat them as JSON data (common pattern in this project)
            with open(file_path, 'r') as f:
                content = f.read()
                # Remove the [ and ] and treat as JSON
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    data = json.loads(content)
                else:
                    # Try to execute as Python module
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("media_data", file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for the data - check for media type specific variable names first
                    media_type_var = f"{self.media_type}s"  # e.g., 'movies', 'books', 'series'
                    if hasattr(module, media_type_var):
                        data = getattr(module, media_type_var)
                    elif hasattr(module, 'data'):
                        data = module.data
                    else:
                        # Get the first list variable
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, list) and not attr_name.startswith('_'):
                                data = attr
                                break
                        else:
                            raise ValueError(f"No {self.media_type} data found in {file_path}")
        else:
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
            
        # Validate and convert to MediaData objects using dynamic model
        media_items = []
        for item in data:
            try:
                media_item = self.dynamic_model(**item)
                media_items.append(media_item)
            except ValidationError as e:
                self.logger.error(f"Invalid {self.media_type} data: {e}")
                continue
                
        self.logger.info(f"Loaded {len(media_items)} {self.media_type} items from {file_path}")
        return media_items
        
    async def load_media_from_api(self, 
                                limit: Optional[int] = None,
                                item_names: Optional[List[str]] = None) -> List[MediaData]:
        """Load media data from external API (Jellyfin/Emby/Plex) based on configured media type."""
        if not self.settings.JELLYFIN_API_KEY:
            raise ValueError("External API key not configured")
            
        # Get API type from YAML config  
        if self.media_config.execution and "api_type" in self.media_config.execution:
            api_type = self.media_config.execution["api_type"]
        elif self.media_config.execution and "jellyfin_type" in self.media_config.execution:
            # Backward compatibility
            api_type = self.media_config.execution["jellyfin_type"]
        else:
            # Use generic capitalization if not specified in YAML
            api_type = self.media_type.capitalize()
        media_items = []
        
        # Get all possible fields from the YAML config
        all_fields = set(self.media_config.fields.keys()) if self.media_config.fields else set()
        # Add validation required fields
        if self.media_config.validation and "field_constraints" in self.media_config.validation:
            all_fields.update(self.media_config.validation["field_constraints"].keys())
        
        # Standard Jellyfin fields to include
        standard_fields = ["Overview", "Taglines", "Genres", "People", "Studios", "Tags", "ProductionLocations"]
        field_list = ",".join(standard_fields + list(all_fields))
        
        async with httpx.AsyncClient() as client:
            headers = {"X-Emby-Token": self.settings.JELLYFIN_API_KEY}
            
            if item_names:
                # Load specific items by name
                for name in item_names:
                    params = {
                        "searchTerm": name,
                        "includeItemTypes": api_type,
                        "recursive": True,
                        "fields": field_list
                    }
                    
                    response = await client.get(
                        f"{self.settings.JELLYFIN_URL}/Items",
                        headers=headers,
                        params=params
                    )
                    
                    if response.status_code == 200:
                        items = response.json().get("Items", [])
                        for item in items:
                            try:
                                media_item = self.dynamic_model(**item)
                                media_items.append(media_item)
                            except ValidationError as e:
                                self.logger.error(f"Invalid {self.media_type} data for {name}: {e}")
                                
            else:
                # Load all items with optional limit
                params = {
                    "includeItemTypes": api_type,
                    "recursive": True,
                    "fields": field_list,
                    "sortBy": "DateCreated",
                    "sortOrder": "Descending"
                }
                
                if limit:
                    params["limit"] = limit
                    
                response = await client.get(
                    f"{self.settings.JELLYFIN_URL}/Items",
                    headers=headers,
                    params=params
                )
                
                if response.status_code == 200:
                    items = response.json().get("Items", [])
                    for item in items:
                        try:
                            media_item = self.dynamic_model(**item)
                            media_items.append(media_item)
                        except ValidationError as e:
                            self.logger.error(f"Invalid {self.media_type} data: {e}")
                            
        self.logger.info(f"Loaded {len(media_items)} {self.media_type} items from external API")
        return media_items
        
    async def enrich_media_item(self, media_item: MediaData) -> Dict[str, Any]:
        """Enrich a single media item using the configured pipeline."""
        # Convert to dict for processing
        media_dict = media_item.model_dump()
        
        # Add computed fields that might be referenced in YAML
        self._add_computed_fields(media_dict)
        
        # For now, we'll use a simple enrichment approach
        # This will be replaced with proper router service calls
        enriched_data = media_dict.copy()
        
        # Process each field configured for enrichment
        if self.media_config.fields:
            for field_name, field_config in self.media_config.fields.items():
                if "enrichments" in field_config:
                    try:
                        enriched_value = await self._process_field_enrichments(
                            field_name, field_config, media_dict
                        )
                        enriched_data[field_name] = enriched_value
                    except Exception as e:
                        self.logger.error(f"Error enriching field {field_name} for {media_item.Name}: {e}")
                        
        return enriched_data
        
    def _add_computed_fields(self, media_dict: Dict[str, Any]):
        """Add computed fields based on YAML configuration."""
        if not self.media_config.fields:
            return
            
        # Process computed fields defined in YAML config
        for field_name, field_config in self.media_config.fields.items():
            computed_rule = field_config.get("computed_from")
            if computed_rule and isinstance(computed_rule, dict):
                try:
                    # Get source array
                    source_field = computed_rule.get("source")
                    if not source_field or source_field not in media_dict:
                        continue
                        
                    source_data = media_dict[source_field]
                    if not isinstance(source_data, list):
                        continue
                    
                    # Apply filters
                    filtered_items = source_data
                    filters = computed_rule.get("filters", {})
                    for filter_key, filter_value in filters.items():
                        filtered_items = [item for item in filtered_items 
                                        if isinstance(item, dict) and item.get(filter_key) == filter_value]
                    
                    if not filtered_items:
                        continue
                    
                    # Extract value
                    extract_field = computed_rule.get("extract", "Name")
                    multiple = computed_rule.get("multiple", False)
                    
                    if multiple:
                        # Return list of extracted values
                        values = [item.get(extract_field) for item in filtered_items if item.get(extract_field)]
                        if values:
                            media_dict[field_name] = values
                    else:
                        # Return first extracted value
                        first_value = filtered_items[0].get(extract_field)
                        if first_value:
                            media_dict[field_name] = first_value
                            
                except Exception as e:
                    self.logger.warning(f"Failed to compute field {field_name}: {e}")
                    
    async def _process_field_enrichments(self, field_name: str, field_config: Dict[str, Any], 
                                       media_dict: Dict[str, Any]) -> Any:
        """Process enrichments for a specific field."""
        # For now, return the original value or a placeholder
        # This is where we'll eventually call the plugin system
        
        source_field = field_config.get("source_field")
        if source_field and source_field in media_dict:
            return media_dict[source_field]
        elif source_field is None:
            # Synthetic field - would be generated by plugins
            return f"[Synthetic {field_name} - would be enriched by plugins]"
        else:
            return None
            
    async def store_media_item(self, media_data: Dict[str, Any]):
        """Store enriched media data in MongoDB."""
        collection_name = self.media_config.output.get("collection", f"{self.media_type}_enriched") if self.media_config.output else f"{self.media_type}_enriched"
        collection = self.db[collection_name]
        
        # Add metadata
        media_data["_ingested_at"] = datetime.utcnow()
        media_data["_media_type"] = self.media_type
        media_data["_enrichment_version"] = "2.0"
        
        # Upsert based on ID
        result = await collection.replace_one(
            {"Id": media_data["Id"]},
            media_data,
            upsert=True
        )
        
        if result.upserted_id:
            self.logger.info(f"Inserted {self.media_type}: {media_data['Name']}")
        else:
            self.logger.info(f"Updated {self.media_type}: {media_data['Name']}")
            
    async def ingest_media(self, 
                         media_items: List[MediaData],
                         batch_size: int = 5,
                         skip_enrichment: bool = False):
        """Ingest media items with optional enrichment."""
        total = len(media_items)
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = media_items[i:i + batch_size]
            
            # Process each item in the batch
            tasks = []
            for item in batch:
                if skip_enrichment:
                    # Store without enrichment
                    tasks.append(self.store_media_item(item.model_dump()))
                else:
                    # Enrich and store
                    async def process_item(media_item):
                        enriched = await self.enrich_media_item(media_item)
                        await self.store_media_item(enriched)
                        
                    tasks.append(process_item(item))
                    
            # Execute batch
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
            
    async def verify_ingestion(self):
        """Verify ingestion results."""
        collection_name = self.media_config.output.get("collection", f"{self.media_type}_enriched") if self.media_config.output else f"{self.media_type}_enriched"
        collection = self.db[collection_name]
        
        # Get counts
        total_count = await collection.count_documents({})
        enriched_count = await collection.count_documents({"_enrichment_version": "2.0"})
        media_type_count = await collection.count_documents({"_media_type": self.media_type})
        
        # Get sample
        sample = await collection.find_one({"_enrichment_version": "2.0", "_media_type": self.media_type})
        
        self.logger.info(f"Verification Results for {self.media_type}:")
        self.logger.info(f"  Total items: {total_count}")
        self.logger.info(f"  {self.media_type} items: {media_type_count}")
        self.logger.info(f"  Enriched items: {enriched_count}")
        
        if sample:
            self.logger.info(f"  Sample enriched fields: {list(sample.keys())}")
            
        return {
            "media_type": self.media_type,
            "total": total_count,
            "media_type_total": media_type_count,
            "enriched": enriched_count,
            "sample_fields": list(sample.keys()) if sample else []
        }


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest media items into MongoDB with enrichment")
    
    # Media type
    parser.add_argument("--media-type", type=str, default="movie", 
                       help="Media type to ingest (movie, series, music, etc.)")
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--json", type=str, help="Path to JSON or Python file with media data")
    source_group.add_argument("--api", action="store_true", help="Load from external API (Jellyfin/Emby/Plex)")
    
    # API options
    parser.add_argument("--limit", type=int, help="Limit number of items to load")
    parser.add_argument("--items", nargs="+", help="Specific item names to load")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip enrichment pipeline")
    parser.add_argument("--verify", action="store_true", help="Verify ingestion after completion")
    
    args = parser.parse_args()
    
    # Run ingestion
    async with IngestionManager(media_type=args.media_type) as manager:
        # Load media items
        if args.json:
            media_items = await manager.load_media_from_json(args.json)
        else:
            media_items = await manager.load_media_from_api(
                limit=args.limit,
                item_names=args.items
            )
            
        if not media_items:
            print(f"No {args.media_type} items found to ingest")
            return
            
        print(f"Found {len(media_items)} {args.media_type} items to ingest")
        
        # Ingest media items
        await manager.ingest_media(
            media_items,
            batch_size=args.batch_size,
            skip_enrichment=args.skip_enrichment
        )
        
        # Verify if requested
        if args.verify:
            results = await manager.verify_ingestion()
            print(f"\nVerification Results:")
            print(f"  Media type: {results['media_type']}")
            print(f"  Total items: {results['total']}")
            print(f"  {results['media_type']} items: {results['media_type_total']}")
            print(f"  Enriched items: {results['enriched']}")
            if results['sample_fields']:
                print(f"  Sample fields: {', '.join(results['sample_fields'][:10])}...")


if __name__ == "__main__":
    asyncio.run(main())