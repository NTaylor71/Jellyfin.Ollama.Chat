"""
Task Types - Definitions for all supported task types in the queue system.

Provides standardized task type definitions for the worker's
direct plugin execution system.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class TaskType(str, Enum):
    """Supported task types for the queue system."""
    
    # Plugin execution tasks
    PLUGIN_EXECUTION = "plugin_execution"
    
    # Concept expansion tasks
    CONCEPT_EXPANSION = "concept_expansion"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    QUESTION_EXPANSION = "question_expansion"
    
    # Service management tasks
    SERVICE_HEALTH = "service_health"
    SERVICE_DISCOVERY = "service_discovery"
    
    # Data processing tasks
    DATA_INGESTION = "data_ingestion"
    DATA_ENHANCEMENT = "data_enhancement"
    
    # System tasks
    HEALTH_CHECK = "health_check"
    METRICS_COLLECTION = "metrics_collection"
    CACHE_MAINTENANCE = "cache_maintenance"


@dataclass
class TaskDefinition:
    """Definition of a task type for direct plugin execution."""
    
    task_type: TaskType
    plugin_name: str
    description: str
    required_fields: list
    optional_fields: list = None
    execution_timeout: float = 30.0
    priority: str = "normal"
    requires_service: bool = False
    service_type: Optional[str] = None
    
    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = []


# Task type definitions for direct plugin execution
TASK_DEFINITIONS: Dict[TaskType, TaskDefinition] = {
    
    TaskType.PLUGIN_EXECUTION: TaskDefinition(
        task_type=TaskType.PLUGIN_EXECUTION,
        plugin_name="dynamic",  # Determined from task data
        description="Execute a specific plugin by name",
        required_fields=["plugin_name", "data"],
        optional_fields=["context", "timeout"],
        execution_timeout=60.0
    ),
    
    TaskType.CONCEPT_EXPANSION: TaskDefinition(
        task_type=TaskType.CONCEPT_EXPANSION,
        plugin_name="ConceptExpansionPlugin",
        description="Expand concepts using multiple providers",
        required_fields=["concept"],
        optional_fields=["media_context", "max_concepts", "field_name"],
        execution_timeout=30.0,
        requires_service=True,
        service_type="split_nlp"
    ),
    
    TaskType.TEMPORAL_ANALYSIS: TaskDefinition(
        task_type=TaskType.TEMPORAL_ANALYSIS,
        plugin_name="TemporalAnalysisPlugin",
        description="Analyze temporal expressions in text",
        required_fields=["text"],
        optional_fields=["media_context", "document_date"],
        execution_timeout=20.0,
        requires_service=True,
        service_type="split_nlp"
    ),
    
    TaskType.QUESTION_EXPANSION: TaskDefinition(
        task_type=TaskType.QUESTION_EXPANSION,
        plugin_name="QuestionExpansionPlugin",
        description="Expand user questions using LLM",
        required_fields=["question"],
        optional_fields=["context", "max_expansions"],
        execution_timeout=45.0,
        requires_service=True,
        service_type="llm"
    ),
    
    TaskType.SERVICE_HEALTH: TaskDefinition(
        task_type=TaskType.SERVICE_HEALTH,
        plugin_name="ServiceRegistryPlugin",
        description="Check health of registered services",
        required_fields=[],
        optional_fields=["service_name", "service_type"],
        execution_timeout=10.0
    ),
    
    TaskType.SERVICE_DISCOVERY: TaskDefinition(
        task_type=TaskType.SERVICE_DISCOVERY,
        plugin_name="ServiceRegistryPlugin",
        description="Discover available services",
        required_fields=[],
        optional_fields=["service_type", "capability"],
        execution_timeout=5.0
    ),
    
    TaskType.DATA_INGESTION: TaskDefinition(
        task_type=TaskType.DATA_INGESTION,
        plugin_name="DataIngestionPlugin",
        description="Ingest and process media data",
        required_fields=["data_source", "media_data"],
        optional_fields=["processing_options"],
        execution_timeout=120.0,
        requires_service=True,
        service_type="split_nlp"
    ),
    
    TaskType.DATA_ENHANCEMENT: TaskDefinition(
        task_type=TaskType.DATA_ENHANCEMENT,
        plugin_name="DataEnhancementPlugin", 
        description="Enhance media data with AI insights",
        required_fields=["media_data"],
        optional_fields=["enhancement_types", "confidence_threshold"],
        execution_timeout=90.0,
        requires_service=True,
        service_type="split_nlp"
    ),
    
    TaskType.HEALTH_CHECK: TaskDefinition(
        task_type=TaskType.HEALTH_CHECK,
        plugin_name="HealthMonitorPlugin",
        description="Perform system health checks",
        required_fields=[],
        optional_fields=["component", "detailed"],
        execution_timeout=15.0
    ),
    
    TaskType.METRICS_COLLECTION: TaskDefinition(
        task_type=TaskType.METRICS_COLLECTION,
        plugin_name="MetricsPlugin",
        description="Collect system metrics",
        required_fields=[],
        optional_fields=["metric_types", "time_range"],
        execution_timeout=10.0
    ),
    
    TaskType.CACHE_MAINTENANCE: TaskDefinition(
        task_type=TaskType.CACHE_MAINTENANCE,
        plugin_name="CacheManagerPlugin",
        description="Perform cache maintenance operations",
        required_fields=["operation"],
        optional_fields=["cache_type", "parameters"],
        execution_timeout=30.0
    )
}


def get_task_definition(task_type: str) -> Optional[TaskDefinition]:
    """Get task definition for a task type."""
    try:
        task_enum = TaskType(task_type)
        return TASK_DEFINITIONS.get(task_enum)
    except ValueError:
        return None


def validate_task_data(task_type: str, task_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate task data against task definition.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    definition = get_task_definition(task_type)
    if not definition:
        return False, f"Unknown task type: {task_type}"
    
    # Check required fields
    missing_fields = []
    for field in definition.required_fields:
        if field not in task_data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None


def get_plugin_for_task(task_type: str, task_data: Dict[str, Any]) -> Optional[str]:
    """Get the plugin name that should handle a task type."""
    definition = get_task_definition(task_type)
    if not definition:
        return None
    
    # Handle dynamic plugin selection
    if definition.plugin_name == "dynamic":
        return task_data.get("plugin_name")
    
    return definition.plugin_name


def get_service_requirements(task_type: str) -> Dict[str, Any]:
    """Get service requirements for a task type."""
    definition = get_task_definition(task_type)
    if not definition:
        return {}
    
    return {
        "requires_service": definition.requires_service,
        "service_type": definition.service_type,
        "execution_timeout": definition.execution_timeout,
        "priority": definition.priority
    }


def list_task_types() -> Dict[str, Dict[str, Any]]:
    """List all available task types with their definitions."""
    return {
        task_type.value: {
            "plugin_name": definition.plugin_name,
            "description": definition.description,
            "required_fields": definition.required_fields,
            "optional_fields": definition.optional_fields,
            "execution_timeout": definition.execution_timeout,
            "requires_service": definition.requires_service,
            "service_type": definition.service_type
        }
        for task_type, definition in TASK_DEFINITIONS.items()
    }