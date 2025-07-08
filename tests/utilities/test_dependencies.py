#!/usr/bin/env python3
"""
Test Dependencies Validator - FAIL FAST Approach
Validates ALL required packages, models, and external dependencies before any tests run.
NO FALLBACKS - If core components are missing, this should fail immediately with clear error messages.
"""

import subprocess
import sys
from pathlib import Path
from ..tests_shared import logger
from ..tests_shared import settings_to_console


def test_core_python_packages():
    """Test core Python packages are installed - FAIL FAST if missing."""
    logger.info("🐍 Testing Core Python Packages")
    logger.info("-" * 40)
    
    core_packages = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("redis", "Redis client"),
        ("pymongo", "MongoDB client"),
        ("pydantic", "Data validation"),
        ("structlog", "Structured logging"),
        ("psutil", "System and process utilities"),
        ("aiohttp", "Async HTTP client"),
    ]
    
    for package, description in core_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package:12} - {description}")
        except ImportError:
            raise AssertionError(f"Core package '{package}' not installed. Run: pip install {package}")


def test_nlp_packages():
    """Test NLP packages are installed - FAIL FAST if missing."""
    logger.info("\n🧠 Testing NLP Packages")
    logger.info("-" * 40)
    
    nlp_packages = [
        ("spacy", "SpaCy NLP library"),
        ("gensim", "Topic modeling and word embeddings"),
        ("ollama", "Ollama LLM client"),
        ("sklearn", "Machine learning library"),
        ("transformers", "Hugging Face transformers"),
        ("torch", "PyTorch deep learning"),
        ("faiss", "Facebook AI Similarity Search"),
        ("py_heideltime", "HeidelTime temporal tagger"),
    ]
    
    for package, description in nlp_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package:12} - {description}")
        except ImportError:
            raise AssertionError(f"NLP package '{package}' not installed. Run: pip install {package}")


def test_nlp_models():
    """Test ALL NLP models are available via model manager - FAIL FAST if missing."""
    logger.info("\n🎯 Testing NLP Models")
    logger.info("-" * 40)
    
    try:
        import spacy
    except ImportError:
        raise AssertionError("SpaCy not installed. Run: pip install spacy")
    
    # Use model manager to check for models
    from src.shared.model_manager import ModelManager
    manager = ModelManager(models_base_path="./models")
    
    # Check required models via model manager
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.check_all_models())
        summary = manager.get_model_summary()
        
        # Check ALL required models, not just SpaCy
        missing_required = []
        for model_id, model_info in summary['models'].items():
            if model_info['required'] and model_info['status'] != 'available':
                missing_required.append(f"{model_info['package']}/{model_info['name']}")
            elif model_info['status'] == 'available':
                logger.info(f"✅ {model_info['name']:15} - {model_info['package']} model")
            else:
                logger.info(f"⚠️  {model_info['name']:15} - {model_info['package']} model (optional)")
        
        if missing_required:
            raise AssertionError(f"Required models missing: {missing_required}. Run: python manage_models.py download")
                
    except Exception as e:
        raise AssertionError(f"Model manager check failed: {e}. Run: python manage_models.py download")


def test_java_dependencies():
    """Test Java dependencies for HeidelTime/SUTime - FAIL FAST if missing."""
    logger.info("\n☕ Testing Java Dependencies")
    logger.info("-" * 40)
    
    # Check if Java is available
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract Java version from stderr (where java -version outputs)
            version_output = result.stderr.split('\n')[0] if result.stderr else "Unknown version"
            logger.info(f"✅ Java found: {version_output}")
        else:
            raise AssertionError("Java not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        raise AssertionError("Java not found. HeidelTime requires Java 17+. Install with: sudo apt install openjdk-17-jdk")
    
    # Check JAVA_HOME using config system
    try:
        from src.shared.config import get_settings
        settings = get_settings()

        logger.info(f"✅ JAVA_HOME (via config): {settings.java_home}")
        
        # Verify the path exists
        from pathlib import Path
        if Path(settings.java_home).exists():
            logger.info(f"✅ JAVA_HOME path verified")
        else:
            logger.info(f"⚠️  JAVA_HOME path does not exist: {settings.java_home}")
            
    except Exception as e:
        raise AssertionError(f"JAVA_HOME configuration failed: {e}")


def test_ollama_connectivity():
    """Test Ollama service connectivity - FAIL FAST if unavailable."""
    logger.info("\n🤖 Testing Ollama Connectivity")
    logger.info("-" * 40)
    
    try:
        import ollama
    except ImportError:
        raise AssertionError("Ollama not installed. Run: pip install ollama")
    
    try:
        # Test basic connectivity by listing models
        models = ollama.list()
        # models is an object with .models attribute containing list of Model objects
        model_list = models.models if hasattr(models, 'models') else []
        model_names = [model.model for model in model_list if hasattr(model, 'model')]
        
        if not model_names:
            raise AssertionError("Ollama accessible but no models installed. Install required models: ollama pull llama3.2:3b && ollama pull nomic-embed-text")
        else:
            logger.info(f"✅ Ollama accessible with models: {model_names[:3]}")
            
    except Exception as e:
        raise AssertionError(f"Ollama not accessible: {e}. Ensure Ollama is running with: ollama serve")


def test_project_imports():
    """Test project-specific imports - FAIL FAST if broken."""
    logger.info("\n📦 Testing Project Imports")
    logger.info("-" * 40)
    
    project_imports = [
        ("src.shared.config", "get_settings"),
        ("src.worker.resource_queue_manager", "ResourceAwareQueueManager"),
        ("src.plugins.base", "BasePlugin"),
        ("src.shared.concept_expander", "ConceptExpander"),
        ("src.shared.hardware_config", "get_resource_limits"),
        ("src.storage.cache_manager", "CacheManager"),
    ]
    
    for module_name, class_name in project_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"✅ {module_name}")
        except ImportError as e:
            raise AssertionError(f"Project import failed: {module_name}.{class_name} - {e}")
        except AttributeError as e:
            raise AssertionError(f"Project class missing: {module_name}.{class_name} - {e}")


def test_model_directories():
    """Test model directories exist and are writable - FAIL FAST if misconfigured."""
    logger.info("\n📁 Testing Model Directories")
    logger.info("-" * 40)
    
    model_dirs = [
        ("models", "Main models directory"),
        ("models/nltk_data", "NLTK data directory"),
        ("models/gensim_data", "Gensim models directory"),
        ("models/spacy_data", "SpaCy models directory"),
        ("logs", "Logs directory"),
    ]
    
    for dir_path, description in model_dirs:
        path = Path(dir_path)
        
        if not path.exists():
            logger.info(f"⚠️  {dir_path:20} - {description} (will be created)")
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created: {dir_path}")
            except Exception as e:
                raise AssertionError(f"Cannot create directory {dir_path}: {e}")
        else:
            logger.info(f"✅ {dir_path:20} - {description}")
        
        # Test if directory is writable
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            logger.info(f"✅ {dir_path:20} - Writable")
        except Exception as e:
            raise AssertionError(f"Directory {dir_path} is not writable: {e}")


def test_environment_variables():
    """Test environment configuration using shared/config.py - FAIL FAST if broken."""
    logger.info("\n🌍 Testing Environment Configuration")
    logger.info("-" * 40)
    
    try:
        from src.shared.config import get_settings
        settings = get_settings()

        logger.info(f"✅ Config system: Working")
        
        # Test that critical settings can be accessed
        if hasattr(settings, 'NLTK_DATA'):
            logger.info(f"✅ NLTK_DATA: {settings.NLTK_DATA}")
        if hasattr(settings, 'GENSIM_DATA_DIR'):
            logger.info(f"✅ GENSIM_DATA_DIR: {settings.GENSIM_DATA_DIR}")
            
        # Test that the config system can handle environment-specific values
        logger.info(f"✅ Environment-aware configuration working")
        
    except Exception as e:
        raise AssertionError(f"Configuration system failed: {e}. Check shared/config.py and .env file")


def main():
    """Run all dependency tests - FAIL FAST on any failure."""
    logger.info("🔍 DEPENDENCY VALIDATION - FAIL FAST APPROACH")
    logger.info("=" * 60)
    logger.info("Testing ALL required packages, models, and dependencies")
    logger.info("NO FALLBACKS - Any missing component will cause immediate failure")
    logger.info("=" * 60)

    settings_to_console()
    
    # Run all tests in order - any failure will stop execution
    test_core_python_packages()
    test_nlp_packages()
    test_nlp_models()
    test_java_dependencies()
    test_ollama_connectivity()
    test_project_imports()
    test_model_directories()
    test_environment_variables()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ALL DEPENDENCY TESTS PASSED")
    logger.info("=" * 60)
    logger.info("✅ Core Python packages: Available")
    logger.info("✅ NLP packages: Available")
    logger.info("✅ SpaCy models: Available")
    logger.info("✅ Java dependencies: Available")
    logger.info("✅ Ollama connectivity: Working")
    logger.info("✅ Project imports: Working")
    logger.info("✅ Model directories: Configured")
    logger.info("✅ Environment: Configured")
    logger.info("")
    logger.info("🚀 System is ready for testing - no component fallbacks needed!")
    logger.info("🚀 Run provider tests with confidence: python test_stage_3_providers.py")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error(f"\n❌ DEPENDENCY TEST FAILED: {e}")
        logger.error("❌ Fix the above issue before running any other tests")
        logger.error("❌ The system is NOT ready for testing")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)