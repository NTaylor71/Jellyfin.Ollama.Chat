"""
Unified Model Manager for all NLP models and data.

Ensures all required models are downloaded and available before services start.
Designed for Docker entrypoint usage with persistent volume mounting.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import httpx
from src.shared.config import get_settings


try:
    from src.shared.model_config_loader import get_model_config_loader
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)



class ModelStatus(Enum):
    """Model availability status."""
    AVAILABLE = "available"
    MISSING = "missing"
    DOWNLOADING = "downloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    package: str
    storage_path: str
    size_mb: int
    required: bool = True
    status: ModelStatus = ModelStatus.MISSING
    error_message: Optional[str] = None


class ModelManager:
    """
    Unified manager for all NLP models and data.
    
    Handles NLTK, Gensim, SpaCy model downloads with Docker volume persistence.
    Ollama models are checked for availability but managed externally.
    """
    
    def __init__(self, models_base_path: str = "/app/models"):

        self.settings = get_settings()

        self.models_base_path = Path(models_base_path)
        self.models_base_path.mkdir(parents=True, exist_ok=True)
        
        
        self.nltk_data_path = self.models_base_path / "nltk_data"
        self.gensim_data_path = self.models_base_path / "gensim_data"
        self.spacy_data_path = self.models_base_path / "spacy_data"
        

        for path in [self.nltk_data_path, self.gensim_data_path, self.spacy_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        
        os.environ['NLTK_DATA'] = str(self.nltk_data_path)
        os.environ['GENSIM_DATA_DIR'] = str(self.gensim_data_path)
        

        self.models = self._load_models_from_config()
        self.ollama_base_url = self.settings.OLLAMA_INGESTION_BASE_URL
    
    def _define_required_models(self) -> Dict[str, ModelInfo]:
        """Define all required models and their properties."""
        return {

            "nltk_punkt": ModelInfo(
                name="punkt",
                package="nltk",
                storage_path=str(self.nltk_data_path / "tokenizers" / "punkt"),
                size_mb=20,
                required=True
            ),
            "nltk_stopwords": ModelInfo(
                name="stopwords",
                package="nltk",
                storage_path=str(self.nltk_data_path / "corpora" / "stopwords"),
                size_mb=8,
                required=True
            ),
            "nltk_wordnet": ModelInfo(
                name="wordnet",
                package="nltk",
                storage_path=str(self.nltk_data_path / "corpora" / "wordnet"),
                size_mb=80,
                required=True
            ),
            

            "gensim_word2vec": ModelInfo(
                name="word2vec-google-news-300",
                package="gensim",
                storage_path=str(self.gensim_data_path / "word2vec-google-news-300"),
                size_mb=1700,
                required=True
            ),
            

            "spacy_en_core": ModelInfo(
                name="en_core_web_sm",
                package="spacy",
                storage_path=str(self.spacy_data_path / "en_core_web_sm"),
                size_mb=15,
                required=True
            ),
            



            "ollama_ingestion": ModelInfo(
                name="mistral:latest",  
                package="ollama",
                storage_path="external",
                size_mb=2000,
                required=True
            ),
            "ollama_embed": ModelInfo(
                name="nomic-embed-text",
                package="ollama",
                storage_path="external",
                size_mb=500,
                required=False
            )
        }
    
    def _load_models_from_config(self) -> Dict[str, ModelInfo]:
        """Load models from YAML configuration files."""
        if not CONFIG_LOADER_AVAILABLE:
            logger.warning("Config loader not available, falling back to hardcoded models")
            return self._define_required_models()
        
        try:
            config_loader = get_model_config_loader()
            all_models = {}
            

            for service_type in config_loader.get_service_types():
                service_models = config_loader.convert_to_model_info(service_type, str(self.models_base_path))
                all_models.update(service_models)
            
            logger.info(f"Loaded {len(all_models)} models from configuration")
            return all_models
            
        except Exception as e:
            logger.error(f"Failed to load models from config: {e}")
            logger.info("Falling back to hardcoded models")
            return self._define_required_models()
    
    async def check_all_models(self) -> Dict[str, ModelStatus]:
        """Check status of all models."""
        logger.info("Checking status of all required models...")
        
        status_results = {}
        
        for model_id, model_info in self.models.items():
            status = await self._check_model_status(model_info)
            model_info.status = status
            status_results[model_id] = status
            
            status_symbol = "✅" if status == ModelStatus.AVAILABLE else "❌"
            logger.info(f"{status_symbol} {model_info.package}/{model_info.name}: {status.value}")
        
        return status_results
    
    async def _check_model_status(self, model_info: ModelInfo) -> ModelStatus:
        """Check if a specific model is available."""
        try:
            if model_info.package == "ollama":
                return await self._check_ollama_model(model_info.name)
            elif model_info.package == "nltk":
                return self._check_nltk_model(model_info)
            elif model_info.package == "gensim":
                return self._check_gensim_model(model_info)
            elif model_info.package == "spacy":
                return self._check_spacy_model(model_info)
            else:
                logger.warning(f"Unknown package: {model_info.package}")
                return ModelStatus.ERROR
                
        except Exception as e:
            logger.error(f"Error checking {model_info.name}: {e}")
            model_info.error_message = str(e)
            return ModelStatus.ERROR
    
    def _check_nltk_model(self, model_info: ModelInfo) -> ModelStatus:
        """Check if NLTK model is available by actually trying to use it."""
        try:
            import nltk
            
            
            nltk.data.path.clear()
            nltk.data.path.append(str(self.nltk_data_path))
            

            if model_info.name == "punkt":
                nltk.data.find('tokenizers/punkt')
                logger.debug(f"✅ NLTK model '{model_info.name}' found at tokenizers/punkt")
            elif model_info.name == "stopwords":
                nltk.data.find('corpora/stopwords')
                logger.debug(f"✅ NLTK model '{model_info.name}' found at corpora/stopwords")
            elif model_info.name == "wordnet":
                nltk.data.find('corpora/wordnet')
                logger.debug(f"✅ NLTK model '{model_info.name}' found at corpora/wordnet")
            else:

                if Path(model_info.storage_path).exists():
                    logger.debug(f"✅ NLTK model '{model_info.name}' found at {model_info.storage_path}")
                    return ModelStatus.AVAILABLE
                logger.debug(f"❌ NLTK model '{model_info.name}' not found at {model_info.storage_path}")
                return ModelStatus.MISSING
                
            return ModelStatus.AVAILABLE
            
        except Exception as e:
            logger.debug(f"❌ NLTK model '{model_info.name}' not available: {e}")
            return ModelStatus.MISSING
    
    def _check_gensim_model(self, model_info: ModelInfo) -> ModelStatus:
        """Check if Gensim model is available."""
        if Path(model_info.storage_path).exists():
            return ModelStatus.AVAILABLE
        return ModelStatus.MISSING
    
    def _check_spacy_model(self, model_info: ModelInfo) -> ModelStatus:
        """Check if SpaCy model is available."""
        try:
            import spacy

            spacy.load(model_info.name)
            return ModelStatus.AVAILABLE
        except OSError:

            return ModelStatus.MISSING
        except Exception:
            return ModelStatus.ERROR
    
    async def _check_ollama_model(self, model_name: str) -> ModelStatus:
        """Check if Ollama model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")
                
                if response.status_code != 200:
                    logger.debug(f"Ollama service not reachable (HTTP {response.status_code}) - this is expected if Ollama is on host")
                    return ModelStatus.ERROR
                
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                
                if model_name in available_models:
                    logger.debug(f"✅ Ollama model '{model_name}' found and available")
                    return ModelStatus.AVAILABLE
                
                
                model_with_latest = f"{model_name}:latest"
                if model_with_latest in available_models:
                    logger.debug(f"✅ Ollama model '{model_name}' found as '{model_with_latest}'")
                    return ModelStatus.AVAILABLE
                
                
                for available in available_models:
                    if available.startswith(f"{model_name}:"):
                        logger.debug(f"✅ Ollama model '{model_name}' found as '{available}'")
                        return ModelStatus.AVAILABLE
                
                logger.info(f"Ollama model '{model_name}' not found. Available: {available_models}")
                return ModelStatus.MISSING
                    
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.debug(f"Ollama connection failed for '{model_name}' - this is normal if Ollama runs on host: {type(e).__name__}")
            return ModelStatus.ERROR
        except Exception as e:
            logger.warning(f"Unexpected error checking Ollama model '{model_name}': {e}")
            return ModelStatus.ERROR
    
    async def download_missing_models(self, force_download: bool = False) -> bool:
        """Download all missing required models."""
        logger.info("Starting model download process...")
        
        if force_download:
            logger.info("Force download enabled - will re-download all models")
        
        success = True
        
        for model_id, model_info in self.models.items():
            if not model_info.required:
                logger.info(f"Skipping optional model: {model_info.name}")
                continue
            
            if model_info.package == "ollama":

                status = await self._check_ollama_model(model_info.name)
                if status == ModelStatus.AVAILABLE:
                    logger.debug(f"✅ Ollama model '{model_info.name}' is available")
                elif status == ModelStatus.MISSING:
                    logger.info(f"⚠️ Ollama model '{model_info.name}' not found. Use: ollama pull {model_info.name}")
                else:
                    logger.debug(f"🔌 Cannot connect to Ollama to check '{model_info.name}' (normal if Ollama runs on host)")
                continue
            
            if force_download or model_info.status == ModelStatus.MISSING:
                logger.info(f"Downloading {model_info.package}/{model_info.name} ({model_info.size_mb}MB)...")
                
                download_success = await self._download_model(model_info)
                if not download_success:
                    logger.error(f"Failed to download {model_info.name}")
                    success = False
                else:
                    logger.info(f"✅ Successfully downloaded {model_info.name}")
                    model_info.status = ModelStatus.AVAILABLE
        
        return success
    
    async def _download_model(self, model_info: ModelInfo) -> bool:
        """Download a specific model."""
        try:
            model_info.status = ModelStatus.DOWNLOADING
            
            if model_info.package == "nltk":
                return await self._download_nltk_model(model_info)
            elif model_info.package == "gensim":
                return await self._download_gensim_model(model_info)
            elif model_info.package == "spacy":
                return await self._download_spacy_model(model_info)
            else:
                logger.error(f"Unknown package for download: {model_info.package}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {model_info.name}: {e}")
            model_info.error_message = str(e)
            model_info.status = ModelStatus.ERROR
            return False
    
    async def _download_nltk_model(self, model_info: ModelInfo) -> bool:
        """Download NLTK model."""
        try:
            import nltk
            
            
            nltk.data.path.clear()
            nltk.data.path.append(str(self.nltk_data_path))
            
            
            existing_status = self._check_nltk_model(model_info)
            if existing_status == ModelStatus.AVAILABLE:
                logger.info(f"📁 NLTK model '{model_info.name}' already exists and is accessible, skipping download")
                return True
            
            logger.info(f"📥 Downloading NLTK model '{model_info.name}' to {self.nltk_data_path}")

            success = nltk.download(model_info.name, download_dir=str(self.nltk_data_path), quiet=False)
            
            if success:

                if model_info.name == "wordnet":
                    wordnet_zip = self.nltk_data_path / "corpora" / "wordnet.zip"
                    wordnet_dir = self.nltk_data_path / "corpora" / "wordnet"
                    
                    if wordnet_zip.exists() and not wordnet_dir.exists():
                        import zipfile
                        logger.info(f"Extracting wordnet.zip to {wordnet_dir}")
                        with zipfile.ZipFile(wordnet_zip, 'r') as zip_ref:
                            zip_ref.extractall(self.nltk_data_path / "corpora")
                

                try:
                    if model_info.name == "punkt":
                        nltk.data.find('tokenizers/punkt')
                    elif model_info.name == "stopwords":
                        nltk.data.find('corpora/stopwords')
                    elif model_info.name == "wordnet":
                        nltk.data.find('corpora/wordnet')
                    logger.info(f"NLTK model {model_info.name} verified after download")
                    return True
                except Exception as verify_error:
                    logger.error(f"NLTK model {model_info.name} downloaded but not accessible: {verify_error}")
                    return False
            
            return success
            
        except Exception as e:
            logger.error(f"NLTK download failed for {model_info.name}: {e}")
            return False
    
    async def _download_gensim_model(self, model_info: ModelInfo) -> bool:
        """Download Gensim model."""
        try:
            import gensim.downloader as api
            


            logger.info(f"Downloading {model_info.name} via Gensim API...")
            

            model = api.load(model_info.name)
            

            logger.info(f"Gensim model {model_info.name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Gensim download failed for {model_info.name}: {e}")
            return False
    
    async def _download_spacy_model(self, model_info: ModelInfo) -> bool:
        """Download SpaCy model."""
        try:
            import subprocess
            import sys
            
            
            cmd = [sys.executable, "-m", "spacy", "download", model_info.name]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"SpaCy model {model_info.name} downloaded successfully")
                return True
            else:
                logger.error(f"SpaCy download failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"SpaCy download failed for {model_info.name}: {e}")
            return False
    
    async def ensure_all_models(self) -> bool:
        """Ensure all required models are available. Main entrypoint method."""
        logger.info("🚀 Starting Model Manager - ensuring all models are available")
        
        
        status_results = await self.check_all_models()
        

        missing_required = [
            model_id for model_id, status in status_results.items()
            if status != ModelStatus.AVAILABLE and self.models[model_id].required
        ]
        
        if not missing_required:
            logger.info("✅ All required models are available!")
            return True
        
        logger.info(f"📥 Found {len(missing_required)} missing required models")
        

        download_success = await self.download_missing_models()
        
        if download_success:
            logger.info("✅ All required models are now available!")
            return True
        else:
            logger.error("❌ Some model downloads failed")
            return False
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get summary of all models for reporting."""
        summary = {
            "total_models": len(self.models),
            "required_models": len([m for m in self.models.values() if m.required]),
            "available_models": len([m for m in self.models.values() if m.status == ModelStatus.AVAILABLE]),
            "total_size_mb": sum(m.size_mb for m in self.models.values() if m.status == ModelStatus.AVAILABLE),
            "models": {}
        }
        
        for model_id, model_info in self.models.items():
            summary["models"][model_id] = {
                "name": model_info.name,
                "package": model_info.package,
                "size_mb": model_info.size_mb,
                "required": model_info.required,
                "status": model_info.status.value,
                "error": model_info.error_message
            }
        
        return summary
    
    async def cleanup_models(self, cleanup_cache: bool = False, dry_run: bool = False) -> Tuple[int, float]:
        """Clean up unused models and cache files."""
        logger.info("Starting model cleanup process...")
        
        cleaned_files = 0
        space_freed = 0.0
        
        cleanup_paths = []
        

        for path in [self.nltk_data_path, self.gensim_data_path, self.spacy_data_path]:
            if path.exists():

                for pattern in ["*.tmp", "*.download", "*.bak", "*~"]:
                    cleanup_paths.extend(path.rglob(pattern))
        
        if cleanup_cache:

            for path in [self.models_base_path / ".cache", self.models_base_path / "tmp"]:
                if path.exists():
                    cleanup_paths.extend(path.rglob("*"))
        

        for file_path in cleanup_paths:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                space_freed += size_mb
                cleaned_files += 1
                
                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.debug(f"Removed: {file_path} ({size_mb:.1f} MB)")
                    except Exception as e:
                        logger.warning(f"Could not remove {file_path}: {e}")
        
        return cleaned_files, space_freed
    
    async def verify_models(self) -> Dict[str, Dict[str, any]]:
        """Verify model integrity and functionality."""
        logger.info("Starting model verification process...")
        
        results = {}
        
        for model_id, model_info in self.models.items():
            if model_info.status != ModelStatus.AVAILABLE:
                results[model_id] = {
                    "valid": False,
                    "message": f"Model not available (status: {model_info.status.value})"
                }
                continue
            
            try:
                if model_info.package == "nltk":
                    valid, message = await self._verify_nltk_model(model_info)
                elif model_info.package == "gensim":
                    valid, message = await self._verify_gensim_model(model_info)
                elif model_info.package == "spacy":
                    valid, message = await self._verify_spacy_model(model_info)
                elif model_info.package == "ollama":
                    valid, message = await self._verify_ollama_model(model_info)
                else:
                    valid, message = False, f"Unknown package: {model_info.package}"
                
                results[model_id] = {"valid": valid, "message": message}
                
            except Exception as e:
                results[model_id] = {
                    "valid": False,
                    "message": f"Verification error: {str(e)}"
                }
        
        return results
    
    async def _verify_nltk_model(self, model_info: ModelInfo) -> Tuple[bool, str]:
        """Verify NLTK model can be loaded and used."""
        try:
            import nltk
            nltk.data.path.clear()
            nltk.data.path.append(str(self.nltk_data_path))
            
            if model_info.name == "punkt":
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                test_result = tokenizer.tokenize("Hello world. This is a test.")
                return len(test_result) > 0, "Punkt tokenizer functional"
            elif model_info.name == "stopwords":
                stopwords = nltk.corpus.stopwords.words('english')
                return len(stopwords) > 100, f"Stopwords loaded ({len(stopwords)} words)"
            elif model_info.name == "wordnet":
                from nltk.corpus import wordnet as wn
                synsets = wn.synsets('dog')
                return len(synsets) > 0, f"WordNet functional ({len(synsets)} synsets for 'dog')"
            
            return True, "Model exists and basic checks passed"
            
        except Exception as e:
            return False, f"NLTK verification failed: {str(e)}"
    
    async def _verify_gensim_model(self, model_info: ModelInfo) -> Tuple[bool, str]:
        """Verify Gensim model can be loaded and used."""
        try:
            import gensim.downloader as api
            

            model = api.load(model_info.name)
            

            if hasattr(model, 'similarity'):
                sim = model.similarity('computer', 'technology')
                return True, f"Word2Vec model functional (computer-technology similarity: {sim:.3f})"
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            return False, f"Gensim verification failed: {str(e)}"
    
    async def _verify_spacy_model(self, model_info: ModelInfo) -> Tuple[bool, str]:
        """Verify SpaCy model can be loaded and used."""
        try:
            import spacy
            

            nlp = spacy.load(model_info.name)
            

            doc = nlp("Hello world! This is a test sentence.")
            tokens = [token.text for token in doc]
            
            return len(tokens) > 0, f"SpaCy model functional ({len(tokens)} tokens parsed)"
            
        except Exception as e:
            return False, f"SpaCy verification failed: {str(e)}"
    
    async def _verify_ollama_model(self, model_info: ModelInfo) -> Tuple[bool, str]:
        """Verify Ollama model is accessible and responsive."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:

                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": model_info.name,
                        "prompt": "Hello",
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return True, f"Ollama model responsive (generated {len(result.get('response', ''))} chars)"
                else:
                    return False, f"Ollama model not responsive (HTTP {response.status_code})"
                    
        except Exception as e:
            return False, f"Ollama verification failed: {str(e)}"
    
    async def update_all_models(self, dry_run: bool = False) -> Dict[str, Dict[str, any]]:
        """Update all models to latest versions."""
        logger.info("Starting model update process...")
        
        results = {}
        
        for model_id, model_info in self.models.items():
            if model_info.package == "ollama":

                results[model_id] = {
                    "updated": False,
                    "message": "Ollama models managed externally"
                }
                continue
            
            try:
                if dry_run:
                    results[model_id] = {
                        "updated": False,
                        "message": "Would re-download to get latest version"
                    }
                else:

                    success = await self._download_model(model_info)
                    results[model_id] = {
                        "updated": success,
                        "message": "Re-downloaded successfully" if success else "Update failed"
                    }
                    
            except Exception as e:
                results[model_id] = {
                    "updated": False,
                    "message": f"Update error: {str(e)}"
                }
        
        return results


async def main():
    """CLI entrypoint for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Model Manager")
    parser.add_argument("--check", action="store_true", help="Check model status only")
    parser.add_argument("--download", action="store_true", help="Download missing models")
    parser.add_argument("--force", action="store_true", help="Force re-download all models")
    parser.add_argument("--models-path", default="/app/models", help="Base path for model storage")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    
    manager = ModelManager(models_base_path=args.models_path)
    
    if args.check:

        await manager.check_all_models()
        summary = manager.get_model_summary()
        print(json.dumps(summary, indent=2))
        
    elif args.download or args.force:

        success = await manager.download_missing_models(force_download=args.force)
        sys.exit(0 if success else 1)
        
    else:

        success = await manager.ensure_all_models()
        

        summary = manager.get_model_summary()
        logger.info(f"Model Summary: {summary['available_models']}/{summary['required_models']} required models available")
        logger.info(f"Total model storage: {summary['total_size_mb']} MB")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())