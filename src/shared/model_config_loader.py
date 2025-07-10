"""
Model Configuration Loader - Loads model definitions from YAML files.

This module provides configuration-driven model management, allowing users to
specify which models to use for each service via YAML configuration files.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from src.shared.model_manager import ModelInfo, ModelStatus

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    required: bool
    size_mb: int
    description: str
    storage_path: str
    service_type: str
    use_cases: List[str] = None
    language: Optional[str] = None


class ModelConfigLoader:
    """Loads model configurations from YAML files."""
    
    def __init__(self, config_dir: str = "/app/config/models"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, ModelConfig]] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all model configuration files."""
        if not self.config_dir.exists():
            logger.warning(f"Model config directory {self.config_dir} does not exist")
            return
            
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                self._load_config_file(config_file)
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _load_config_file(self, config_file: Path):
        """Load a single YAML configuration file."""
        logger.info(f"Loading model config: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        

        service_type = config_file.stem.replace("_models", "")
        
        models_config = {}
        

        if service_type == "ollama":
            models_config = self._parse_ollama_config(config_data)
        elif service_type == "nltk":
            models_config = self._parse_nltk_config(config_data)
        elif service_type == "gensim":
            models_config = self._parse_gensim_config(config_data)
        elif service_type == "spacy":
            models_config = self._parse_spacy_config(config_data)
        else:
            logger.warning(f"Unknown service type: {service_type}")
            return
        
        self.configs[service_type] = models_config
        logger.info(f"Loaded {len(models_config)} models for {service_type}")
    
    def _parse_ollama_config(self, config_data: dict) -> Dict[str, ModelConfig]:
        """Parse simplified Ollama configuration."""
        models = {}
        
        if "ingestion_model" in config_data:
            models["ingestion_model"] = ModelConfig(
                name=config_data["ingestion_model"],
                required=True,
                size_mb=2000,
                description="Main ingestion model",
                storage_path="external",
                service_type="ollama",
                use_cases=["ingestion", "concept_expansion"]
            )
        
        if "embedding_model" in config_data:
            models["embedding_model"] = ModelConfig(
                name=config_data["embedding_model"],
                required=False,
                size_mb=500,
                description="Text embedding model",
                storage_path="external",
                service_type="ollama",
                use_cases=["embeddings", "similarity"]
            )
        

        for key, value in config_data.items():
            if key not in ["ingestion_model", "embedding_model"] and not key.startswith("#"):
                models[key] = ModelConfig(
                    name=value,
                    required=False,
                    size_mb=1000,
                    description=f"Custom model: {key}",
                    storage_path="external",
                    service_type="ollama",
                    use_cases=["custom"]
                )
        
        return models
    
    def _parse_nltk_config(self, config_data: dict) -> Dict[str, ModelConfig]:
        """Parse simplified NLTK configuration."""
        models = {}
        

        model_defaults = {
            "punkt": {"size_mb": 20, "description": "Punkt sentence tokenizer", "storage_path": "nltk_data/tokenizers/punkt"},
            "stopwords": {"size_mb": 8, "description": "Stopwords corpus", "storage_path": "nltk_data/corpora/stopwords"},
            "wordnet": {"size_mb": 80, "description": "WordNet lexical database", "storage_path": "nltk_data/corpora/wordnet"},
            "vader_lexicon": {"size_mb": 2, "description": "VADER sentiment lexicon", "storage_path": "nltk_data/vader_lexicon"}
        }
        
        for model_name, enabled in config_data.items():
            if enabled and model_name in model_defaults:
                defaults = model_defaults[model_name]
                models[model_name] = ModelConfig(
                    name=model_name,
                    required=True,
                    size_mb=defaults["size_mb"],
                    description=defaults["description"],
                    storage_path=defaults["storage_path"],
                    service_type="nltk"
                )
        
        return models
    
    def _parse_gensim_config(self, config_data: dict) -> Dict[str, ModelConfig]:
        """Parse simplified Gensim configuration."""
        models = {}
        

        model_defaults = {
            "word2vec-google-news-300": {"size_mb": 1700, "description": "Google News Word2Vec model"},
            "glove-wiki-gigaword-300": {"size_mb": 1000, "description": "GloVe Wikipedia + Gigaword model"},
            "fasttext-wiki-news-subwords-300": {"size_mb": 1000, "description": "FastText model with subwords"}
        }
        
        if "word_model" in config_data:
            model_name = config_data["word_model"]
            defaults = model_defaults.get(model_name, {"size_mb": 1000, "description": f"Custom model: {model_name}"})
            
            models["word_model"] = ModelConfig(
                name=model_name,
                required=True,
                size_mb=defaults["size_mb"],
                description=defaults["description"],
                storage_path=f"gensim_data/{model_name}",
                service_type="gensim"
            )
        
        return models
    
    def _parse_spacy_config(self, config_data: dict) -> Dict[str, ModelConfig]:
        """Parse simplified SpaCy configuration."""
        models = {}
        

        model_defaults = {
            "en_core_web_sm": {"size_mb": 15, "description": "English core model (small)", "language": "en"},
            "en_core_web_md": {"size_mb": 50, "description": "English core model (medium)", "language": "en"},
            "en_core_web_lg": {"size_mb": 750, "description": "English core model (large)", "language": "en"},
            "es_core_news_sm": {"size_mb": 15, "description": "Spanish core model (small)", "language": "es"},
            "fr_core_news_sm": {"size_mb": 15, "description": "French core model (small)", "language": "fr"},
            "de_core_news_sm": {"size_mb": 15, "description": "German core model (small)", "language": "de"}
        }
        
        for config_key, model_name in config_data.items():
            if config_key.endswith("_model") and model_name in model_defaults:
                defaults = model_defaults[model_name]
                models[config_key] = ModelConfig(
                    name=model_name,
                    required=True,
                    size_mb=defaults["size_mb"],
                    description=defaults["description"],
                    storage_path=f"spacy_data/{model_name}",
                    service_type="spacy",
                    language=defaults["language"]
                )
        
        return models
    
    def get_models_for_service(self, service_type: str) -> Dict[str, ModelConfig]:
        """Get all models for a specific service type."""
        return self.configs.get(service_type, {})
    
    def get_required_models_for_service(self, service_type: str) -> Dict[str, ModelConfig]:
        """Get only required models for a specific service type."""
        all_models = self.get_models_for_service(service_type)
        return {k: v for k, v in all_models.items() if v.required}
    
    def convert_to_model_info(self, service_type: str, models_base_path: str) -> Dict[str, ModelInfo]:
        """Convert ModelConfig objects to ModelInfo objects for compatibility."""
        models_config = self.get_models_for_service(service_type)
        model_info_dict = {}
        
        for model_id, config in models_config.items():
            
            if config.storage_path and not config.storage_path.startswith("/"):

                storage_path = str(Path(models_base_path) / config.storage_path)
            else:
                storage_path = config.storage_path or "external"
            
            model_info = ModelInfo(
                name=config.name,
                package=config.service_type,
                storage_path=storage_path,
                size_mb=config.size_mb,
                required=config.required,
                status=ModelStatus.UNKNOWN
            )
            
            model_info_dict[model_id] = model_info
        
        return model_info_dict
    
    def get_model_by_use_case(self, service_type: str, use_case: str) -> Optional[ModelConfig]:
        """Get a model that supports a specific use case."""
        models = self.get_models_for_service(service_type)
        
        for model_config in models.values():
            if use_case in (model_config.use_cases or []):
                return model_config
        
        return None
    
    def reload_configs(self):
        """Reload all configuration files."""
        self.configs.clear()
        self._load_all_configs()
    
    def get_service_types(self) -> List[str]:
        """Get all available service types."""
        return list(self.configs.keys())
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of all loaded configurations."""
        summary = {}
        
        for service_type, models in self.configs.items():
            required_count = sum(1 for m in models.values() if m.required)
            total_size = sum(m.size_mb for m in models.values() if m.required)
            
            summary[service_type] = {
                "total_models": len(models),
                "required_models": required_count,
                "optional_models": len(models) - required_count,
                "total_size_mb": total_size,
                "models": list(models.keys())
            }
        
        return summary



_model_config_loader = None


def get_model_config_loader() -> ModelConfigLoader:
    """Get the global ModelConfigLoader instance."""
    global _model_config_loader
    if _model_config_loader is None:
        _model_config_loader = ModelConfigLoader()
    return _model_config_loader


if __name__ == "__main__":

    loader = ModelConfigLoader()
    
    print("=== Model Configuration Summary ===")
    summary = loader.get_config_summary()
    for service_type, info in summary.items():
        print(f"\n{service_type.upper()}:")
        print(f"  Total models: {info['total_models']}")
        print(f"  Required models: {info['required_models']}")
        print(f"  Total size: {info['total_size_mb']} MB")
        print(f"  Models: {', '.join(info['models'])}")
    

    print("\n=== Use Case Examples ===")
    ingestion_model = loader.get_model_by_use_case("ollama", "ingestion")
    if ingestion_model:
        print(f"Ingestion model: {ingestion_model.name}")
    
    embedding_model = loader.get_model_by_use_case("ollama", "embeddings")
    if embedding_model:
        print(f"Embedding model: {embedding_model.name}")