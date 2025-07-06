"""
Smart configuration management with automatic environment detection.
Supports localhost, Docker, and production environments seamlessly.
"""

import os
from functools import lru_cache
from typing import List, Optional, Union
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with automatic environment detection.
    
    Supports three environments:
    - localhost: Local development with services on localhost
    - docker: Docker Compose environment with container networking
    - production: Production environment with external services
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ==========================================================================
    # ENVIRONMENT DETECTION
    # ==========================================================================
    
    ENV: str = Field(default="localhost", description="Environment: localhost, docker, production")
    
    @validator("ENV")
    def validate_env(cls, v):
        valid_envs = ["localhost", "docker", "production"]
        if v not in valid_envs:
            raise ValueError(f"ENV must be one of {valid_envs}, got: {v}")
        return v
    
    @property
    def is_localhost(self) -> bool:
        """Check if running in localhost development mode."""
        return self.ENV == "localhost"
    
    @property
    def is_docker(self) -> bool:
        """Check if running in Docker environment."""
        return self.ENV == "docker"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENV == "production"
    
    # ==========================================================================
    # REDIS CONFIGURATION
    # ==========================================================================
    
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_DB: int = Field(default=0)
    REDIS_QUEUE: str = Field(default="chat:queue")
    REDIS_RESULT_QUEUE: str = Field(default="chat:results")
    REDIS_DEAD_LETTER_QUEUE: str = Field(default="chat:failed")
    
    # Docker overrides
    DOCKER_REDIS_HOST: str = Field(default="redis")
    DOCKER_REDIS_PORT: int = Field(default=6379)
    
    @property
    def redis_host(self) -> str:
        """Get Redis host based on environment."""
        if self.is_docker:
            return self.DOCKER_REDIS_HOST
        return self.REDIS_HOST
    
    @property
    def redis_port(self) -> int:
        """Get Redis port based on environment."""
        if self.is_docker:
            return self.DOCKER_REDIS_PORT
        return self.REDIS_PORT
    
    @property
    def redis_url(self) -> str:
        """Get complete Redis URL."""
        password_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.REDIS_DB}"
    
    # ==========================================================================
    # OLLAMA CONFIGURATION
    # ==========================================================================
    
    # Chat service
    OLLAMA_CHAT_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = Field(default="llama3.2:3b")
    OLLAMA_CHAT_TIMEOUT: int = Field(default=300)

    DOCKER_CHAT_BASE_URL: str = Field(default="http://localhost:11434")
    DOCKER_CHAT_MODEL: str = Field(default="llama3.2:3b")
    DOCKER_CHAT_TIMEOUT: int = Field(default=300)
    
    # Embedding service (use same as chat for localhost)
    OLLAMA_EMBED_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_EMBED_MODEL: str = Field(default="nomic-embed-text")
    OLLAMA_EMBED_TIMEOUT: int = Field(default=60)


    @property
    def ollama_chat_url(self) -> str:
        """Get Ollama chat URL based on environment."""
        if self.is_docker:
            return self.DOCKER_OLLAMA_CHAT_URL
        return self.OLLAMA_CHAT_BASE_URL

    @property
    def ollama_chat_model(self) -> str:
        """Get Ollama chat model based on environment."""
        if self.is_docker:
            return self.DOCKER_OLLAMA_CHAT_URL
        return self.OLLAMA_CHAT_MODEL

    @property
    def ollama_embed_url(self) -> str:
        """Get Ollama embedding URL based on environment."""
        if self.is_docker:
            return self.DOCKER_OLLAMA_EMBED_URL
        return self.OLLAMA_EMBED_BASE_URL
    
    # ==========================================================================
    # VECTOR DATABASE (FAISS) CONFIGURATION
    # ==========================================================================
    
    VECTORDB_URL: str = Field(default="http://localhost:6333")
    VECTORDB_TIMEOUT: int = Field(default=30)
    FAISS_INDEX_PATH: str = Field(default="./data/faiss_index")
    FAISS_BACKUP_ENABLED: bool = Field(default=True)
    FAISS_BACKUP_INTERVAL: int = Field(default=3600)  # seconds
    
    # Docker override
    DOCKER_VECTORDB_URL: str = Field(default="http://faiss-service:6333")
    
    @property
    def vectordb_url(self) -> str:
        """Get FAISS service URL based on environment."""
        if self.is_docker:
            return self.DOCKER_VECTORDB_URL
        return self.VECTORDB_URL
    
    # ==========================================================================
    # API CONFIGURATION
    # ==========================================================================
    
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_URL: str = Field(default="http://localhost:8000")
    API_WORKERS: int = Field(default=1)
    API_RELOAD: bool = Field(default=True)
    
    # Docker override
    DOCKER_API_URL: str = Field(default="http://api:8000")
    
    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        if self.is_docker:
            return self.DOCKER_API_URL
        return self.API_URL
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    RATE_LIMIT_BURST: int = Field(default=10)
    
    # ==========================================================================
    # MONGODB CONFIGURATION
    # ==========================================================================
    
    MONGODB_HOST: str = Field(default="localhost")
    MONGODB_PORT: int = Field(default=27017)
    MONGODB_DATABASE: str = Field(default="jellyfin_rag")
    MONGODB_USERNAME: Optional[str] = Field(default=None)
    MONGODB_PASSWORD: Optional[str] = Field(default=None)
    MONGODB_AUTH_SOURCE: str = Field(default="admin")
    
    # Docker overrides
    DOCKER_MONGODB_HOST: str = Field(default="mongodb")
    DOCKER_MONGODB_PORT: int = Field(default=27017)
    
    @property
    def mongodb_host(self) -> str:
        """Get MongoDB host based on environment."""
        if self.is_docker:
            return self.DOCKER_MONGODB_HOST
        return self.MONGODB_HOST
    
    @property
    def mongodb_port(self) -> int:
        """Get MongoDB port based on environment."""
        if self.is_docker:
            return self.DOCKER_MONGODB_PORT
        return self.MONGODB_PORT
    
    @property
    def mongodb_url(self) -> str:
        """Get complete MongoDB URL."""
        auth_part = ""
        if self.MONGODB_USERNAME and self.MONGODB_PASSWORD:
            auth_part = f"{self.MONGODB_USERNAME}:{self.MONGODB_PASSWORD}@"
        
        return f"mongodb://{auth_part}{self.mongodb_host}:{self.mongodb_port}/{self.MONGODB_DATABASE}"
    
    # ==========================================================================
    # EXTERNAL SERVICES
    # ==========================================================================
    
    # Jellyfin
    JELLYFIN_URL: str = Field(default="http://localhost:8096")
    JELLYFIN_API_KEY: str = Field(default="")
    JELLYFIN_USER_ID: Optional[str] = Field(default=None)
    
    @validator("JELLYFIN_API_KEY")
    def validate_jellyfin_key(cls, v):
        if not v and os.getenv("ENV", "localhost") == "production":
            raise ValueError("JELLYFIN_API_KEY is required in production")
        return v
    
    # ==========================================================================
    # LOGGING & DEBUGGING
    # ==========================================================================
    
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="structured")  # structured, json, text
    ENABLE_DEBUG_LOGS: bool = Field(default=False)
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    # ==========================================================================
    # DEVELOPMENT FEATURES
    # ==========================================================================
    
    ENABLE_CORS: bool = Field(default=True)
    ENABLE_API_DOCS: bool = Field(default=True)
    ENABLE_METRICS: bool = Field(default=True)
    ENABLE_TRACING: bool = Field(default=False)
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    CORS_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    CORS_HEADERS: List[str] = Field(default=["*"])
    
    # ==========================================================================
    # MICROSERVICES CONFIGURATION  
    # ==========================================================================
    
    # NLP Provider Service
    NLP_SERVICE_URL: str = Field(default="http://localhost:8001")
    NLP_SERVICE_PORT: int = Field(default=8001)
    
    # LLM Provider Service
    LLM_SERVICE_URL: str = Field(default="http://localhost:8002")
    LLM_SERVICE_PORT: int = Field(default=8002)
    
    # Plugin Router Service
    ROUTER_SERVICE_URL: str = Field(default="http://localhost:8003")
    ROUTER_SERVICE_PORT: int = Field(default=8003)
    
    # Docker overrides
    DOCKER_NLP_SERVICE_URL: str = Field(default="http://nlp-service:8001")
    DOCKER_LLM_SERVICE_URL: str = Field(default="http://llm-service:8002")
    DOCKER_ROUTER_SERVICE_URL: str = Field(default="http://router-service:8003")
    
    @property
    def nlp_service_url(self) -> str:
        """Get NLP service URL based on environment."""
        if self.is_docker:
            return self.DOCKER_NLP_SERVICE_URL
        return self.NLP_SERVICE_URL
    
    @property
    def llm_service_url(self) -> str:
        """Get LLM service URL based on environment."""
        if self.is_docker:
            return self.DOCKER_LLM_SERVICE_URL
        return self.LLM_SERVICE_URL
    
    @property
    def router_service_url(self) -> str:
        """Get Router service URL based on environment."""
        if self.is_docker:
            return self.DOCKER_ROUTER_SERVICE_URL
        return self.ROUTER_SERVICE_URL
    
    # ==========================================================================
    # PLUGIN SYSTEM
    # ==========================================================================
    
    PLUGIN_DIRECTORY: str = Field(default="./src/plugins")
    PLUGIN_HOT_RELOAD: bool = Field(default=True)
    PLUGIN_RELOAD_INTERVAL: int = Field(default=5)  # seconds
    
    @property
    def plugin_path(self) -> Path:
        """Get plugin directory as Path object."""
        return Path(self.PLUGIN_DIRECTORY)
    
    # ==========================================================================
    # JAVA CONFIGURATION (for HeidelTime, SUTime, etc.)
    # ==========================================================================
    
    JAVA_HOME: Optional[str] = Field(default=None)
    
    @property
    def java_home(self) -> str:
        """Get JAVA_HOME based on environment."""
        if self.JAVA_HOME:
            return self.JAVA_HOME
        
        # Environment-specific defaults
        if self.is_docker:
            # Docker typically has Java at this path
            return "/usr/lib/jvm/java-11-openjdk-amd64"
        else:
            # Localhost - detect Java installation
            import subprocess
            try:
                # Try to find Java home using readlink
                java_path = subprocess.check_output(['readlink', '-f', '/usr/bin/java'], text=True).strip()
                # Remove /jre/bin/java or /bin/java suffix to get JAVA_HOME
                if '/jre/bin/java' in java_path:
                    return java_path.replace('/jre/bin/java', '')
                elif '/bin/java' in java_path:
                    return java_path.replace('/bin/java', '')
                return java_path
            except:
                # Fallback to common locations
                return "/usr/lib/jvm/java-8-openjdk-amd64"
    
    # ==========================================================================
    # MONITORING & OBSERVABILITY
    # ==========================================================================
    
    # Prometheus
    PROMETHEUS_ENABLED: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=9090)
    
    # Jaeger tracing
    JAEGER_ENDPOINT: str = Field(default="http://localhost:14268/api/traces")
    JAEGER_SERVICE_NAME: str = Field(default="rag-system")
    
    # Health checks
    HEALTH_CHECK_INTERVAL: int = Field(default=30)
    HEALTH_CHECK_TIMEOUT: int = Field(default=10)
    
    # ==========================================================================
    # PERFORMANCE & CACHING
    # ==========================================================================
    
    # Response caching
    ENABLE_RESPONSE_CACHE: bool = Field(default=True)
    CACHE_TTL: int = Field(default=3600)  # seconds
    
    # Semantic caching
    SEMANTIC_CACHE_ENABLED: bool = Field(default=True)
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.85)
    
    # Worker configuration
    WORKER_CONCURRENCY: int = Field(default=4)
    WORKER_MAX_RETRIES: int = Field(default=3)
    WORKER_RETRY_DELAY: int = Field(default=5)  # seconds
    
    # FAISS performance
    FAISS_NPROBE: int = Field(default=32)
    FAISS_USE_GPU: bool = Field(default=False)
    FAISS_BATCH_SIZE: int = Field(default=100)
    
    # ==========================================================================
    # SECURITY
    # ==========================================================================
    
    # API security - Enhanced for production
    API_KEY_ENABLED: bool = Field(default=False)
    API_KEY: Optional[str] = Field(default=None)
    JWT_SECRET_KEY: Optional[str] = Field(default=None)
    JWT_ALGORITHM: str = Field(default="RS256")  # Changed to RS256 for better security
    JWT_EXPIRE_MINUTES: int = Field(default=60)
    
    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret(cls, v, values):
        if values.get("API_KEY_ENABLED") and not v:
            raise ValueError("JWT_SECRET_KEY is required when API_KEY_ENABLED=True")
        return v
    
    # ==========================================================================
    # DATABASE (Optional - for job persistence)
    # ==========================================================================
    
    POSTGRES_ENABLED: bool = Field(default=False)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="rag_system")
    POSTGRES_USER: str = Field(default="rag_user")
    POSTGRES_PASSWORD: str = Field(default="rag_password")
    
    @property
    def database_url(self) -> Optional[str]:
        """Get database URL if enabled."""
        if not self.POSTGRES_ENABLED:
            return None
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # ==========================================================================
    # COMPUTED PROPERTIES
    # ==========================================================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in any development mode."""
        return self.ENV in ["localhost", "docker"]
    
    @property
    def debug_enabled(self) -> bool:
        """Check if debug features should be enabled."""
        return self.is_development or self.ENABLE_DEBUG_LOGS
    
    @property
    def data_directory(self) -> Path:
        """Get data directory path."""
        return Path("./data")
    
    @property
    def logs_directory(self) -> Path:
        """Get logs directory path."""
        return Path("./logs")
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.data_directory,
            self.logs_directory,
            Path(self.FAISS_INDEX_PATH).parent,
            self.plugin_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # VALIDATION METHODS
    # ==========================================================================
    
    def validate_ollama_connection(self) -> bool:
        """Validate Ollama services are accessible."""
        import httpx
        
        try:
            # Test chat service
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.ollama_chat_url}/api/tags")
                if response.status_code != 200:
                    return False
                
                # Test embed service
                response = client.get(f"{self.ollama_embed_url}/api/tags")
                return response.status_code == 200
                
        except Exception:
            return False
    
    def validate_redis_connection(self) -> bool:
        """Validate Redis connection."""
        try:
            import redis
            r = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.REDIS_PASSWORD,
                db=self.REDIS_DB,
                socket_timeout=5
            )
            r.ping()
            return True
        except Exception:
            return False
    
    def validate_faiss_service(self) -> bool:
        """Validate FAISS service connection."""
        import httpx
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.vectordb_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    def get_health_status(self) -> dict:
        """Get health status of all services."""
        return {
            "ollama_chat": self.validate_ollama_connection(),
            "redis": self.validate_redis_connection(),
            "faiss": self.validate_faiss_service(),
            "environment": self.ENV,
            "debug_mode": self.debug_enabled
        }
    
    # ==========================================================================
    # ENVIRONMENT-SPECIFIC METHODS
    # ==========================================================================
    
    def get_service_urls(self) -> dict:
        """Get all service URLs for current environment."""
        return {
            "api": self.api_url,
            "vectordb": self.vectordb_url,
            "ollama_chat": self.ollama_chat_url,
            "ollama_embed": self.ollama_embed_url,
            "redis": self.redis_url,
            "prometheus": f"http://localhost:{self.PROMETHEUS_PORT}" if self.PROMETHEUS_ENABLED else None
        }
    
    def get_docker_networks(self) -> List[str]:
        """Get required Docker networks."""
        if not self.is_docker:
            return []
        
        return ["rag-network", "monitoring-network"]
    
    def get_required_volumes(self) -> List[str]:
        """Get required Docker volumes."""
        return [
            "faiss-data",
            "redis-data", 
            "logs-data"
        ]
    
    # ==========================================================================
    # CONFIGURATION EXPORT
    # ==========================================================================
    
    def to_env_dict(self) -> dict:
        """Export configuration as environment variables dictionary."""
        env_dict = {}
        
        for field_name, field_info in self.__fields__.items():
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, list):
                    env_dict[field_name] = ",".join(map(str, value))
                elif isinstance(value, bool):
                    env_dict[field_name] = "true" if value else "false"
                else:
                    env_dict[field_name] = str(value)
        
        return env_dict
    
    def export_docker_env(self, output_path: str = ".env.docker") -> None:
        """Export Docker-compatible environment file."""
        env_dict = self.to_env_dict()
        env_dict["ENV"] = "docker"  # Force Docker environment
        
        with open(output_path, "w") as f:
            f.write("# Auto-generated Docker environment file\n")
            f.write("# Do not edit manually\n\n")
            
            for key, value in sorted(env_dict.items()):
                f.write(f"{key}={value}\n")
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return f"Settings(ENV={self.ENV}, redis={self.redis_host}:{self.redis_port})"


# =============================================================================
# CONFIGURATION FACTORY
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


def get_settings_for_environment(env: str) -> Settings:
    """
    Get settings for a specific environment.
    
    Args:
        env: Environment name (localhost, docker, production)
        
    Returns:
        Settings instance configured for the specified environment
    """
    # Temporarily override ENV
    original_env = os.environ.get("ENV")
    os.environ["ENV"] = env
    
    try:
        # Clear cache to force reload
        get_settings.cache_clear()
        return get_settings()
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["ENV"] = original_env
        else:
            os.environ.pop("ENV", None)
        get_settings.cache_clear()


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration() -> tuple[bool, List[str]]:
    """
    Validate current configuration.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    try:
        settings = get_settings()
        
        # Check required directories
        if not settings.data_directory.exists():
            errors.append(f"Data directory does not exist: {settings.data_directory}")
        
        # Check Jellyfin configuration in production
        if settings.is_production and not settings.JELLYFIN_API_KEY:
            errors.append("JELLYFIN_API_KEY is required in production")
        
        # Check JWT secret in production with API key enabled
        if settings.is_production and settings.API_KEY_ENABLED and not settings.JWT_SECRET_KEY:
            errors.append("JWT_SECRET_KEY is required when API_KEY_ENABLED=True in production")
        
        # Validate service connections (optional - may not be available during build)
        health_status = settings.get_health_status()
        if settings.is_production:
            for service, status in health_status.items():
                if not status and service != "debug_mode":
                    errors.append(f"Service {service} is not accessible")
        
    except Exception as e:
        errors.append(f"Configuration validation failed: {str(e)}")
    
    return len(errors) == 0, errors


# =============================================================================
# DEVELOPMENT UTILITIES
# =============================================================================

def print_configuration_summary():
    """Print a human-readable configuration summary."""
    settings = get_settings()
    
    print(f"""
🔧 RAG System Configuration Summary

Environment: {settings.ENV}
Debug Mode: {settings.debug_enabled}

📡 Service URLs:
""")
    
    for service, url in settings.get_service_urls().items():
        if url:
            status = "🟢" if service in ["api", "vectordb"] else "🔵"
            print(f"   {status} {service.upper()}: {url}")
    
    print(f"""
📊 Performance Settings:
   • Worker Concurrency: {settings.WORKER_CONCURRENCY}
   • FAISS nprobe: {settings.FAISS_NPROBE}
   • Cache TTL: {settings.CACHE_TTL}s
   • Rate Limit: {settings.RATE_LIMIT_PER_MINUTE}/min

🔌 Plugin System:
   • Directory: {settings.PLUGIN_DIRECTORY}
   • Hot Reload: {settings.PLUGIN_HOT_RELOAD}
   • Reload Interval: {settings.PLUGIN_RELOAD_INTERVAL}s

📈 Monitoring:
   • Metrics: {settings.ENABLE_METRICS}
   • Tracing: {settings.ENABLE_TRACING}
   • Prometheus: {settings.PROMETHEUS_ENABLED}
""")


if __name__ == "__main__":
    # Development utility - print configuration when run directly
    print_configuration_summary()
    
    # Validate configuration
    is_valid, errors = validate_configuration()
    if not is_valid:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("\n✅ Configuration is valid")
