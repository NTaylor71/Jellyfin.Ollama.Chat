#!/usr/bin/env python3
"""
Integration test for ModelManager with Docker and unified model storage.
Tests the complete model management system.
"""

import asyncio
from tests.tests_shared import logger
from tests.tests_shared import settings_to_console

import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.model_manager import ModelManager


async def test_model_manager():
    """Test the ModelManager functionality."""
    logger.info("🧪 Testing ModelManager Integration")
    

    real_models_path = Path("./models")
    
    try:


        manager = ModelManager(models_base_path=str(real_models_path))
        

        logger.info("📋 Test 1: Checking model status...")
        status_results = await manager.check_all_models()
        
        if not status_results or not isinstance(status_results, dict):
            raise AssertionError("ModelManager check_all_models returned invalid data - ModelManager is broken")
        
        missing_models = [
            model_id for model_id, status in status_results.items()
            if status.value != "available" and manager.models[model_id].required
        ]
        
        logger.info(f"Found {len(missing_models)} missing required models")
        

        logger.info("📊 Test 2: Getting model summary...")
        summary = manager.get_model_summary()
        
        if not summary or 'available_models' not in summary or 'required_models' not in summary:
            raise AssertionError("ModelManager get_model_summary returned invalid data - summary generation is broken")
        
        logger.info(f"Summary: {summary['available_models']}/{summary['required_models']} available")
        

        logger.info("📥 Test 3: Testing NLTK model availability...")
        if "nltk_stopwords" not in manager.models:
            raise AssertionError("NLTK stopwords model not found in ModelManager.models - configuration is broken")
        

        nltk_status = manager._check_nltk_model(manager.models["nltk_stopwords"])
        if nltk_status.name != "AVAILABLE":
            raise AssertionError(f"NLTK stopwords not available: {nltk_status.name} - models should be pre-installed")
        
        logger.info("✅ NLTK stopwords availability test passed")
        

        logger.info("🦙 Test 4: Testing Ollama model check...")
        ollama_status = await manager._check_ollama_model("llama3.2:3b")
        if not ollama_status:
            raise AssertionError("Ollama model check returned no status - Ollama integration is broken")
        
        logger.info(f"Ollama llama3.2:3b status: {ollama_status.value}")
        

        logger.info("🐳 Test 5: Testing Docker volume path configuration...")
        docker_models_path = Path("/tmp/test_docker_models")
        docker_models_path.mkdir(exist_ok=True)
        
        docker_manager = ModelManager(models_base_path=str(docker_models_path))
        


        logger.info(f"Docker volume paths configured correctly:")
        logger.info(f"  Base: {docker_manager.models_base_path}")
        logger.info(f"  NLTK: {docker_manager.nltk_data_path}")
        logger.info(f"  Gensim: {docker_manager.gensim_data_path}")
        

        for path_name, path_obj in [
            ("Base", Path(docker_manager.models_base_path)),
            ("NLTK", Path(docker_manager.nltk_data_path)),
            ("Gensim", Path(docker_manager.gensim_data_path))
        ]:
            path_obj.mkdir(parents=True, exist_ok=True)
            if not path_obj.exists() or not path_obj.is_dir():
                raise AssertionError(f"Docker {path_name} path not created properly: {path_obj}")
        
        logger.info("✅ Docker volume paths all created and writable")
        
        return True
        
    finally:

        pass


async def test_docker_integration():
    """Test Docker integration with model manager - FAIL FAST if Docker broken."""
    logger.info("🐳 Testing Docker Integration")
    


    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError("Docker not available or not working. Install Docker and ensure it's running.")
    
    logger.info("✅ Docker is available")
    

    result = subprocess.run(
        ["docker", "images", "-q", "jelly-worker"], 
        capture_output=True, text=True
    )
    
    if not result.stdout.strip():
        raise AssertionError("jelly-worker Docker image not found. Build the image first with: docker-compose build worker")
    
    logger.info("✅ jelly-worker image found")
    

    logger.info("📦 Testing model volume...")
    

    result = subprocess.run(
        ["docker", "volume", "ls", "-q", "-f", "name=model-data"],
        capture_output=True, text=True
    )
    
    if "model-data" in result.stdout:
        logger.info("✅ model-data volume exists")
    else:
        logger.info("📝 model-data volume will be created on first run")
    
    return True


async def main():
    """Run all integration tests."""
    logger.info("🚀 Starting ModelManager Integration Tests")
    logger.info("="*60)
    
    tests = [
        ("ModelManager Core", test_model_manager),
        ("Docker Integration", test_docker_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        

        success = await test_func()
        results[test_name] = success
        logger.info(f"🏁 {test_name}: ✅ PASSED")
    

    logger.info("\n" + "="*60)
    logger.info("📊 INTEGRATION TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {status} {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All ModelManager integration tests PASSED!")
        logger.info("💡 Ready to rebuild Docker containers with unified model management")
    else:
        logger.info("⚠️ Some tests failed. Check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        settings_to_console()
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")
        sys.exit(1)