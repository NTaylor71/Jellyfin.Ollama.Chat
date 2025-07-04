"""
MongoDB Plugin Manager
Handles plugin metadata storage, versioning, and status tracking in MongoDB.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field

from ..data.mongo_client import get_mongo_client
from ..data.models import PyObjectId
from .base import PluginMetadata, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class PluginStatus(str, Enum):
    """Plugin status in the registry."""
    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class PluginVersion(BaseModel):
    """Plugin version information."""
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')
    release_date: datetime = Field(default_factory=datetime.utcnow)
    changelog: str = Field(default="", description="Version changelog")
    is_stable: bool = Field(default=True, description="Is this a stable release")
    min_system_version: Optional[str] = Field(None, description="Minimum required system version")


class PluginDependency(BaseModel):
    """Plugin dependency information."""
    name: str = Field(..., description="Dependency plugin name")
    version: str = Field(..., description="Required version (semver)")
    optional: bool = Field(default=False, description="Is this dependency optional")


class PluginDocument(BaseModel):
    """MongoDB document model for plugin metadata."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    # Core metadata
    name: str = Field(..., min_length=1, max_length=100, description="Plugin name")
    display_name: str = Field(..., min_length=1, max_length=100, description="Human-readable name")
    description: str = Field(..., min_length=1, max_length=1000, description="Plugin description")
    author: str = Field(..., min_length=1, max_length=100, description="Plugin author")
    plugin_type: PluginType = Field(..., description="Plugin type")
    
    # Version management
    current_version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$', description="Current version")
    versions: List[PluginVersion] = Field(default_factory=list, description="Version history")
    
    # Status and lifecycle
    status: PluginStatus = Field(default=PluginStatus.DRAFT, description="Plugin status")
    is_enabled: bool = Field(default=True, description="Is plugin enabled")
    execution_priority: ExecutionPriority = Field(default=ExecutionPriority.NORMAL)
    
    # Dependencies and requirements
    dependencies: List[PluginDependency] = Field(default_factory=list, description="Plugin dependencies")
    system_requirements: Dict[str, Any] = Field(default_factory=dict, description="System requirements")
    
    # File information
    file_path: Optional[str] = Field(None, description="Plugin file path")
    file_hash: Optional[str] = Field(None, description="Plugin file hash (SHA-256)")
    file_size: Optional[int] = Field(None, description="Plugin file size in bytes")
    
    # Configuration
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="Plugin configuration schema")
    default_config: Dict[str, Any] = Field(default_factory=dict, description="Default configuration")
    
    # Usage and performance
    installation_count: int = Field(default=0, description="Number of installations")
    last_used: Optional[datetime] = Field(None, description="Last execution time")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Plugin tags")
    category: Optional[str] = Field(None, description="Plugin category")
    license: Optional[str] = Field(None, description="Plugin license")
    homepage: Optional[str] = Field(None, description="Plugin homepage URL")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = Field(None, description="When plugin was published")
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {PyObjectId: str}
    }


class PluginDeployment(BaseModel):
    """Plugin deployment record."""
    
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    plugin_name: str = Field(..., description="Plugin name")
    version: str = Field(..., description="Deployed version")
    environment: str = Field(..., description="Deployment environment (dev/staging/prod)")
    
    # Deployment details
    deployed_by: str = Field(..., description="User who deployed")
    deployment_time: datetime = Field(default_factory=datetime.utcnow)
    rollback_version: Optional[str] = Field(None, description="Previous version for rollback")
    
    # Status
    status: str = Field(default="deployed", description="Deployment status")
    health_check_passed: bool = Field(default=False, description="Health check status")
    error_message: Optional[str] = Field(None, description="Error message if deployment failed")
    
    # Configuration
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Environment-specific config")
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }


class MongoPluginManager:
    """MongoDB-based plugin management system."""
    
    def __init__(self):
        self.mongo_client = get_mongo_client()
        self.plugins_collection = "plugins"
        self.deployments_collection = "plugin_deployments"
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the plugin manager."""
        if self._initialized:
            return
        
        try:
            # Ensure indexes exist
            await self._create_indexes()
            self._initialized = True
            logger.info("MongoDB plugin manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB plugin manager: {e}")
            raise
    
    async def _create_indexes(self) -> None:
        """Create necessary MongoDB indexes."""
        plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
        deployments_coll = self.mongo_client.get_collection(self.deployments_collection)
        
        # Plugin collection indexes
        await plugins_coll.create_index("name", unique=True)
        await plugins_coll.create_index("plugin_type")
        await plugins_coll.create_index("status")
        await plugins_coll.create_index("author")
        await plugins_coll.create_index("tags")
        await plugins_coll.create_index([("name", 1), ("current_version", 1)])
        
        # Deployment collection indexes
        await deployments_coll.create_index([("plugin_name", 1), ("environment", 1)])
        await deployments_coll.create_index("deployment_time")
        await deployments_coll.create_index("status")
        
        logger.info("Plugin manager indexes created successfully")
    
    async def register_plugin(self, plugin_metadata: PluginMetadata, 
                            file_path: str, file_hash: str, 
                            file_size: int) -> Optional[str]:
        """Register a new plugin or update existing one."""
        try:
            await self.initialize()
            
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            # Check if plugin already exists
            existing = await plugins_coll.find_one({"name": plugin_metadata.name})
            
            if existing:
                # Update existing plugin
                return await self._update_plugin(existing, plugin_metadata, file_path, file_hash, file_size)
            else:
                # Create new plugin
                return await self._create_plugin(plugin_metadata, file_path, file_hash, file_size)
                
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_metadata.name}: {e}")
            return None
    
    async def _create_plugin(self, plugin_metadata: PluginMetadata, 
                           file_path: str, file_hash: str, file_size: int) -> str:
        """Create a new plugin document."""
        plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
        
        # Create version record
        version_record = PluginVersion(
            version=plugin_metadata.version,
            changelog="Initial version",
            is_stable=True
        )
        
        # Create plugin document
        plugin_doc = PluginDocument(
            name=plugin_metadata.name,
            display_name=plugin_metadata.name,
            description=plugin_metadata.description,
            author=plugin_metadata.author,
            plugin_type=plugin_metadata.plugin_type,
            current_version=plugin_metadata.version,
            versions=[version_record],
            status=PluginStatus.DRAFT,
            is_enabled=plugin_metadata.is_enabled,
            execution_priority=plugin_metadata.execution_priority,
            dependencies=[
                PluginDependency(name=dep, version="*", optional=False) 
                for dep in plugin_metadata.dependencies
            ],
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            tags=plugin_metadata.tags
        )
        
        result = await plugins_coll.insert_one(plugin_doc.model_dump(by_alias=True))
        plugin_id = str(result.inserted_id)
        
        logger.info(f"Created new plugin: {plugin_metadata.name} (ID: {plugin_id})")
        return plugin_id
    
    async def _update_plugin(self, existing_doc: Dict[str, Any], 
                           plugin_metadata: PluginMetadata, 
                           file_path: str, file_hash: str, file_size: int) -> str:
        """Update an existing plugin document."""
        plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
        
        # Create new version record if version changed
        versions = existing_doc.get("versions", [])
        if plugin_metadata.version != existing_doc.get("current_version"):
            new_version = PluginVersion(
                version=plugin_metadata.version,
                changelog="Updated version",
                is_stable=True
            )
            versions.append(new_version.model_dump())
        
        # Update document
        update_data = {
            "description": plugin_metadata.description,
            "author": plugin_metadata.author,
            "plugin_type": plugin_metadata.plugin_type,
            "current_version": plugin_metadata.version,
            "versions": versions,
            "is_enabled": plugin_metadata.is_enabled,
            "execution_priority": plugin_metadata.execution_priority,
            "dependencies": [
                PluginDependency(name=dep, version="*", optional=False).model_dump() 
                for dep in plugin_metadata.dependencies
            ],
            "file_path": file_path,
            "file_hash": file_hash,
            "file_size": file_size,
            "tags": plugin_metadata.tags,
            "updated_at": datetime.utcnow()
        }
        
        await plugins_coll.update_one(
            {"_id": existing_doc["_id"]},
            {"$set": update_data}
        )
        
        plugin_id = str(existing_doc["_id"])
        logger.info(f"Updated plugin: {plugin_metadata.name} (ID: {plugin_id})")
        return plugin_id
    
    async def get_plugin(self, plugin_name: str) -> Optional[PluginDocument]:
        """Get plugin by name."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            doc = await plugins_coll.find_one({"name": plugin_name})
            if doc:
                return PluginDocument(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get plugin {plugin_name}: {e}")
            return None
    
    async def list_plugins(self, plugin_type: Optional[PluginType] = None,
                         status: Optional[PluginStatus] = None,
                         enabled_only: bool = False) -> List[PluginDocument]:
        """List plugins with optional filtering."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            # Build filter
            filter_query = {}
            if plugin_type:
                filter_query["plugin_type"] = plugin_type
            if status:
                filter_query["status"] = status
            if enabled_only:
                filter_query["is_enabled"] = True
            
            # Execute query
            cursor = plugins_coll.find(filter_query).sort("name", 1)
            docs = await cursor.to_list(length=None)
            
            return [PluginDocument(**doc) for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to list plugins: {e}")
            return []
    
    async def publish_plugin(self, plugin_name: str, version: Optional[str] = None) -> bool:
        """Publish a plugin to production."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            # Find plugin
            plugin = await plugins_coll.find_one({"name": plugin_name})
            if not plugin:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Update status
            update_data = {
                "status": PluginStatus.PUBLISHED,
                "published_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Update to specific version if provided
            if version:
                versions = plugin.get("versions", [])
                if not any(v.get("version") == version for v in versions):
                    logger.error(f"Version {version} not found for plugin {plugin_name}")
                    return False
                update_data["current_version"] = version
            
            await plugins_coll.update_one(
                {"name": plugin_name},
                {"$set": update_data}
            )
            
            logger.info(f"Published plugin: {plugin_name} version {version or 'current'}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish plugin {plugin_name}: {e}")
            return False
    
    async def deprecate_plugin(self, plugin_name: str, reason: str = "") -> bool:
        """Deprecate a plugin."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            update_data = {
                "status": PluginStatus.DEPRECATED,
                "updated_at": datetime.utcnow()
            }
            
            if reason:
                update_data["deprecation_reason"] = reason
            
            result = await plugins_coll.update_one(
                {"name": plugin_name},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            logger.info(f"Deprecated plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deprecate plugin {plugin_name}: {e}")
            return False
    
    async def record_deployment(self, plugin_name: str, version: str, 
                              environment: str, deployed_by: str,
                              config_overrides: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Record a plugin deployment."""
        try:
            await self.initialize()
            deployments_coll = self.mongo_client.get_collection(self.deployments_collection)
            
            # Get previous deployment for rollback info
            previous = await deployments_coll.find_one(
                {"plugin_name": plugin_name, "environment": environment},
                sort=[("deployment_time", -1)]
            )
            
            deployment = PluginDeployment(
                plugin_name=plugin_name,
                version=version,
                environment=environment,
                deployed_by=deployed_by,
                rollback_version=previous.get("version") if previous else None,
                config_overrides=config_overrides or {}
            )
            
            result = await deployments_coll.insert_one(deployment.model_dump(by_alias=True))
            deployment_id = str(result.inserted_id)
            
            logger.info(f"Recorded deployment: {plugin_name} v{version} to {environment}")
            return deployment_id
            
        except Exception as e:
            logger.error(f"Failed to record deployment: {e}")
            return None
    
    async def get_deployment_history(self, plugin_name: str, 
                                   environment: Optional[str] = None) -> List[PluginDeployment]:
        """Get deployment history for a plugin."""
        try:
            await self.initialize()
            deployments_coll = self.mongo_client.get_collection(self.deployments_collection)
            
            filter_query = {"plugin_name": plugin_name}
            if environment:
                filter_query["environment"] = environment
            
            cursor = deployments_coll.find(filter_query).sort("deployment_time", -1)
            docs = await cursor.to_list(length=50)  # Limit to last 50 deployments
            
            return [PluginDeployment(**doc) for doc in docs]
            
        except Exception as e:
            logger.error(f"Failed to get deployment history: {e}")
            return []
    
    async def update_plugin_metrics(self, plugin_name: str, 
                                  metrics: Dict[str, Any]) -> bool:
        """Update plugin performance metrics."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            update_data = {
                "performance_metrics": metrics,
                "last_used": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await plugins_coll.update_one(
                {"name": plugin_name},
                {"$set": update_data}
            )
            
            return result.matched_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update plugin metrics: {e}")
            return False
    
    async def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin system statistics."""
        try:
            await self.initialize()
            plugins_coll = self.mongo_client.get_collection(self.plugins_collection)
            
            # Aggregate statistics
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_plugins": {"$sum": 1},
                        "published_plugins": {
                            "$sum": {"$cond": [{"$eq": ["$status", "published"]}, 1, 0]}
                        },
                        "enabled_plugins": {
                            "$sum": {"$cond": ["$is_enabled", 1, 0]}
                        },
                        "plugin_types": {"$addToSet": "$plugin_type"}
                    }
                }
            ]
            
            result = await plugins_coll.aggregate(pipeline).to_list(length=1)
            stats = result[0] if result else {}
            
            # Get type distribution
            type_pipeline = [
                {"$group": {"_id": "$plugin_type", "count": {"$sum": 1}}}
            ]
            type_result = await plugins_coll.aggregate(type_pipeline).to_list(length=None)
            
            stats["plugin_type_distribution"] = {
                item["_id"]: item["count"] for item in type_result
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get plugin statistics: {e}")
            return {}


# Global plugin manager instance
_plugin_manager = None


async def get_plugin_manager() -> MongoPluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = MongoPluginManager()
        await _plugin_manager.initialize()
    return _plugin_manager