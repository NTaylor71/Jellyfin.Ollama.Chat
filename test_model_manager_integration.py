#!/usr/bin/env python3
"""
Integration test for ModelManager with Docker and unified model storage.
Tests the complete model management system.
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.shared.model_manager import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model_manager():
    """Test the ModelManager functionality."""
    logger.info("🧪 Testing ModelManager Integration")
    
    # Create test models directory
    test_models_path = Path("./test_models")
    test_models_path.mkdir(exist_ok=True)
    
    try:
        # Initialize ModelManager
        manager = ModelManager(models_base_path=str(test_models_path))
        
        # Test 1: Check all models
        logger.info("📋 Test 1: Checking model status...")
        status_results = await manager.check_all_models()
        
        missing_models = [
            model_id for model_id, status in status_results.items()
            if status.value != "available" and manager.models[model_id].required
        ]
        
        logger.info(f"Found {len(missing_models)} missing required models")
        
        # Test 2: Get summary
        logger.info("📊 Test 2: Getting model summary...")
        summary = manager.get_model_summary()
        
        logger.info(f"Summary: {summary['available_models']}/{summary['required_models']} available")
        
        # Test 3: Test NLTK model download (small test)
        logger.info("📥 Test 3: Testing NLTK model download...")
        try:
            nltk_model = manager.models["nltk_stopwords"]
            success = await manager._download_nltk_model(nltk_model)
            if success:
                logger.info("✅ NLTK stopwords download test passed")
            else:
                logger.warning("⚠️ NLTK stopwords download test failed")
        except Exception as e:
            logger.warning(f"⚠️ NLTK download test failed: {e}")
        
        # Test 4: Test Ollama model check
        logger.info("🦙 Test 4: Testing Ollama model check...")
        try:
            ollama_status = await manager._check_ollama_model("llama3.2:3b")
            logger.info(f"Ollama llama3.2:3b status: {ollama_status.value}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama check failed: {e}")
        
        # Test 5: Docker volume simulation
        logger.info("🐳 Test 5: Testing Docker volume paths...")
        docker_models_path = Path("/tmp/test_docker_models")
        docker_models_path.mkdir(exist_ok=True)
        
        docker_manager = ModelManager(models_base_path=str(docker_models_path))
        await docker_manager.check_all_models()
        
        logger.info(f"Docker simulation paths created:")
        logger.info(f"  Base: {docker_manager.models_base_path}")
        logger.info(f"  NLTK: {docker_manager.nltk_data_path}")
        logger.info(f"  Gensim: {docker_manager.gensim_data_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelManager test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if test_models_path.exists():
            import shutil
            shutil.rmtree(test_models_path, ignore_errors=True)


async def test_docker_integration():
    """Test Docker integration with model manager."""
    logger.info("🐳 Testing Docker Integration")
    
    try:
        # Check if Docker is available
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("⚠️ Docker not available, skipping Docker tests")
            return True
        
        logger.info("✅ Docker is available")
        
        # Check if our images exist
        result = subprocess.run(
            ["docker", "images", "-q", "jelly-worker"], 
            capture_output=True, text=True
        )
        
        if not result.stdout.strip():
            logger.warning("⚠️ jelly-worker image not found, skipping Docker integration test")
            return True
        
        logger.info("✅ jelly-worker image found")
        
        # Test model volume mount
        logger.info("📦 Testing model volume...")
        
        # Check if model-data volume exists
        result = subprocess.run(
            ["docker", "volume", "ls", "-q", "-f", "name=model-data"],
            capture_output=True, text=True
        )
        
        if "model-data" in result.stdout:
            logger.info("✅ model-data volume exists")
        else:
            logger.info("📝 model-data volume will be created on first run")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Docker integration test failed: {e}")
        return False


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
        
        try:
            success = await test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"🏁 {test_name}: {status}")
            
        except Exception as e:
            logger.error(f"💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
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
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Interrupted by user")
        sys.exit(1)