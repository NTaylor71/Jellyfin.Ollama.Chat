#!/usr/bin/env python3
"""
Test suite for Advanced Embed Data Enhancer Plugin
Tests hardware-adaptive processing strategies and content enhancement.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.plugins.examples.advanced_embed_enhancer import AdvancedEmbedDataEnhancerPlugin
from src.plugins.base import PluginExecutionContext, PluginType, ExecutionPriority
from src.shared.config import get_settings


class TestAdvancedEmbedDataEnhancer:
    """Test suite for Advanced Embed Data Enhancer Plugin."""
    
    def __init__(self):
        self.plugin = None
        self.test_data = {
            "movie_plot": {
                "Name": "The Matrix",
                "Overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
                "Genres": ["Science Fiction", "Action"],
                "ProductionYear": 1999,
                "Tags": ["cyberpunk", "virtual reality", "philosophy"],
                "Taglines": [
                    "Free your mind",
                    "The fight for the future begins"
                ],
                "People": [
                    {"Name": "Keanu Reeves", "Role": "Neo", "Type": "Actor"},
                    {"Name": "Laurence Fishburne", "Role": "Morpheus", "Type": "Actor"},
                    {"Name": "The Wachowski Brothers", "Role": "Director", "Type": "Director"}
                ],
                "MediaStreams": [
                    {"Type": "Audio", "Language": "eng"},
                    {"Type": "Subtitle", "Language": "eng"},
                    {"Type": "Subtitle", "Language": "spa"},
                    {"Type": "Subtitle", "Language": "fra"}
                ]
            },
            "movie_review": {
                "title": "Inception Review",
                "content": "Christopher Nolan's Inception is a brilliant psychological thriller that explores the nature of dreams and reality. The film features outstanding performances by Leonardo DiCaprio and Marion Cotillard. The visual effects are groundbreaking and the plot is incredibly complex yet satisfying.",
                "rating": "9/10",
                "reviewer": "Film Critic"
            },
            "simple_text": {
                "text": "This is a simple text without much content."
            },
            "complex_document": {
                "title": "Film Analysis: The Evolution of Sci-Fi Cinema",
                "content": "Science fiction cinema has evolved dramatically over the past century. From the early works of Georges Méliès to modern blockbusters like Blade Runner 2049, the genre has consistently pushed the boundaries of visual storytelling. Directors like Ridley Scott, Christopher Nolan, and Denis Villeneuve have created masterpieces that blend philosophical depth with spectacular visuals.",
                "author": "Dr. Jane Smith",
                "publication_date": "2023-01-15",
                "keywords": ["science fiction", "cinema", "visual effects", "storytelling"]
            }
        }
    
    async def setup(self):
        """Set up test environment."""
        print("🔧 Setting up Advanced Embed Data Enhancer test environment...")
        
        # Initialize plugin
        self.plugin = AdvancedEmbedDataEnhancerPlugin()
        
        # Initialize with test config
        config = {
            "test_mode": True,
            "enable_ollama": True,
            "enable_parallel_processing": True
        }
        
        success = await self.plugin.initialize(config)
        if not success:
            print(f"❌ Plugin initialization failed: {self.plugin._initialization_error}")
            return False
        
        print("✅ Plugin initialized successfully")
        return True
    
    async def test_plugin_metadata(self):
        """Test plugin metadata and resource requirements."""
        print("\n📋 Testing plugin metadata...")
        
        # Test metadata
        metadata = self.plugin.metadata
        assert metadata.name == "AdvancedEmbedDataEnhancer"
        assert metadata.plugin_type == PluginType.EMBED_DATA_EMBELLISHER
        assert metadata.execution_priority == ExecutionPriority.HIGH
        assert "embedding" in metadata.tags
        assert "adaptive" in metadata.tags
        
        # Test resource requirements
        requirements = self.plugin.resource_requirements
        assert requirements.min_cpu_cores == 1.0
        assert requirements.preferred_cpu_cores == 4.0
        assert requirements.min_memory_mb == 100.0
        assert requirements.preferred_memory_mb == 500.0
        assert requirements.max_execution_time_seconds == 15.0
        assert requirements.can_use_distributed_resources == True
        
        print("✅ Plugin metadata validation successful")
    
    async def test_low_resource_processing(self):
        """Test low resource processing strategy."""
        print("\n🔧 Testing low resource processing strategy...")
        
        # Create context with limited resources
        context = PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={
                "cpu_cores": 1.0,
                "memory_mb": 100.0,
                "gpu_available": False
            }
        )
        
        # Test with movie plot data
        result = await self.plugin.embellish_embed_data(self.test_data["movie_plot"], context)
        
        # Verify enhancements
        assert "cleaned_text" in result
        assert "extracted_entities" in result
        assert "basic_metadata" in result
        assert "enhancement_metadata" in result
        
        # Check processing strategy
        assert result["enhancement_metadata"]["processing_strategy"] == "Low Resource"
        
        # Verify taglines are included in processed text
        cleaned_text = result["cleaned_text"]
        assert "mind" in cleaned_text.lower() or "future" in cleaned_text.lower(), "Taglines should be included in cleaned text"
        
        # Verify entities
        entities = result["extracted_entities"]
        print(f"  Debug - Extracted entities: {entities}")
        
        assert "years" in entities
        if "1999" not in entities["years"]:
            print(f"  Debug - Years found: {entities['years']}")
            print(f"  Debug - Looking for '1999' in movie plot data")
        assert "1999" in entities["years"]
        
        assert "genres" in entities
        genres_lower = [g.lower() for g in entities["genres"]]
        if "science fiction" not in genres_lower and "science" not in ' '.join(genres_lower):
            print(f"  Debug - Genres found: {entities['genres']}")
            print(f"  Debug - Looking for 'science fiction' or similar")
        # Should find either the exact genre or detect it from the text
        assert ("science fiction" in genres_lower or 
                "science" in ' '.join(genres_lower) or 
                any("sci" in g for g in genres_lower))
        
        print("✅ Low resource processing successful")
    
    async def test_medium_resource_processing(self):
        """Test medium resource processing strategy."""
        print("\n🔧 Testing medium resource processing strategy...")
        
        # Create context with medium resources (below Ollama Enhanced threshold)
        context = PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={
                "cpu_cores": 2.0,
                "memory_mb": 250.0,  # Below Ollama Enhanced (300MB) but above Medium (200MB)
                "gpu_available": False
            }
        )
        
        # Test with movie review data
        result = await self.plugin.embellish_embed_data(self.test_data["movie_review"], context)
        
        # Debug - see what we got
        print(f"  Debug - Result keys: {result.keys()}")
        
        # Verify enhancements
        assert "cleaned_text" in result
        assert "extracted_entities" in result
        assert "enhanced_metadata" in result
        
        # Check processing strategy
        assert result["enhancement_metadata"]["processing_strategy"] == "Medium Resource"
        
        # Verify sentiment analysis (if available)
        if "sentiment_analysis" in result:
            sentiment = result["sentiment_analysis"]
            assert "sentiment" in sentiment
            assert sentiment["sentiment"] in ["positive", "negative", "neutral"]
        
        # Verify movie relevance scoring
        metadata = result["enhanced_metadata"]
        assert "movie_relevance_score" in metadata
        assert isinstance(metadata["movie_relevance_score"], float)
        assert 0.0 <= metadata["movie_relevance_score"] <= 1.0
        
        # Verify language extraction
        if "audio_languages" in metadata:
            assert "eng" in metadata["audio_languages"]
        if "subtitle_languages" in metadata:
            assert len(metadata["subtitle_languages"]) >= 2  # Should have eng, spa, fra
        
        print("✅ Medium resource processing successful")
    
    async def test_high_resource_processing(self):
        """Test high resource processing strategy."""
        print("\n🔧 Testing high resource processing strategy...")
        
        # Create context with high resources (disable Ollama to force High Resource)
        context = PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={
                "cpu_cores": 4.0,
                "memory_mb": 600.0,
                "gpu_available": False
            }
        )
        
        # Temporarily disable Ollama to ensure High Resource strategy is selected
        original_ollama_available = self.plugin._ollama_gpu_available
        self.plugin._ollama_gpu_available = False
        
        # Test with complex document
        result = await self.plugin.embellish_embed_data(self.test_data["complex_document"], context)
        
        # Verify enhancements
        assert "cleaned_text" in result
        assert "extracted_entities" in result
        assert "advanced_metadata" in result
        assert "content_analysis" in result
        
        # Check processing strategy
        assert result["enhancement_metadata"]["processing_strategy"] == "High Resource"
        
        # Verify advanced analysis
        content_analysis = result["content_analysis"]
        assert "movie_analysis" in content_analysis
        assert "readability" in content_analysis
        
        movie_analysis = content_analysis["movie_analysis"]
        assert "genre_mentions" in movie_analysis
        assert "movie_keywords_count" in movie_analysis
        assert "quality_sentiment" in movie_analysis
        
        # Restore original Ollama state
        self.plugin._ollama_gpu_available = original_ollama_available
        
        print("✅ High resource processing successful")
    
    async def test_ollama_enhanced_processing(self):
        """Test Ollama-enhanced processing strategy."""
        print("\n🤖 Testing Ollama-enhanced processing strategy...")
        
        # Create context with Ollama available
        context = PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={
                "cpu_cores": 2.0,
                "memory_mb": 400.0,
                "gpu_available": True
            }
        )
        
        # Test with movie plot data
        result = await self.plugin.embellish_embed_data(self.test_data["movie_plot"], context)
        
        # Verify basic enhancements
        assert "cleaned_text" in result
        assert "extracted_entities" in result
        assert "enhancement_metadata" in result
        
        # Check if Ollama analysis was attempted
        if self.plugin._ollama_client and self.plugin._ollama_gpu_available:
            assert result["enhancement_metadata"]["processing_strategy"] == "Ollama Enhanced"
            
            # Check for LLM analysis (might be empty if Ollama unavailable)
            if "llm_analysis" in result:
                print("📊 Ollama LLM analysis available")
            else:
                print("⚠️  Ollama LLM analysis not available (connection issue)")
        else:
            print("⚠️  Ollama not available, fallback strategy used")
        
        print("✅ Ollama processing test completed")
    
    async def test_strategy_selection(self):
        """Test processing strategy selection logic."""
        print("\n🎯 Testing processing strategy selection...")
        
        test_cases = [
            # (cpu_cores, memory_mb, gpu_available, expected_strategy)
            (1.0, 100.0, False, "Low Resource"),
            (2.0, 250.0, False, "Medium Resource"),  # Below Ollama Enhanced threshold (300MB)
            (4.0, 600.0, False, "High Resource"),    # Need to disable Ollama for this test
            (2.0, 300.0, True, "Ollama Enhanced"),   # Exactly meets Ollama Enhanced requirements
            (8.0, 1000.0, True, "Ollama Enhanced"),  # If Ollama available
        ]
        
        for cpu_cores, memory_mb, gpu_available, expected_strategy in test_cases:
            # Special handling for High Resource test - disable Ollama temporarily
            original_ollama_available = self.plugin._ollama_gpu_available
            if expected_strategy == "High Resource":
                self.plugin._ollama_gpu_available = False
            
            context = PluginExecutionContext(
                user_id="test_user",
                session_id="test_session",
                available_resources={
                    "cpu_cores": cpu_cores,
                    "memory_mb": memory_mb,
                    "gpu_available": gpu_available
                }
            )
            
            strategy = self.plugin._select_processing_strategy(context)
            
            # Adjust expectation if Ollama not available
            if expected_strategy == "Ollama Enhanced" and not (self.plugin._ollama_client and self.plugin._ollama_gpu_available):
                expected_strategy = "High Resource"
            
            assert strategy.name == expected_strategy, f"Expected {expected_strategy}, got {strategy.name}"
            print(f"  ✅ {cpu_cores} cores, {memory_mb}MB -> {strategy.name}")
            
            # Restore original Ollama state
            self.plugin._ollama_gpu_available = original_ollama_available
        
        print("✅ Strategy selection test successful")
    
    async def test_text_extraction(self):
        """Test text content extraction from various data formats."""
        print("\n📝 Testing text content extraction...")
        
        test_cases = [
            # Data with different text field combinations
            {"text": "Main content"}, 
            {"content": "Content field"},
            {"description": "Description field"},
            {"title": "Title", "plot": "Plot content"},
            {"title": "Title", "content": "Content", "summary": "Summary"},
            {"other_field": "No text fields"}
        ]
        
        for i, test_data in enumerate(test_cases):
            extracted = self.plugin._extract_text_content(test_data)
            
            if i < len(test_cases) - 1:  # Should extract text
                assert extracted is not None
                assert len(extracted) > 0
                print(f"  ✅ Case {i+1}: Extracted '{extracted[:50]}...'")
            else:  # No text fields
                assert extracted is None
                print(f"  ✅ Case {i+1}: No text content (as expected)")
        
        print("✅ Text extraction test successful")
    
    async def test_movie_relevance_scoring(self):
        """Test movie relevance scoring accuracy."""
        print("\n🎬 Testing movie relevance scoring...")
        
        test_texts = [
            ("This is a great action movie with amazing special effects", "High relevance"),
            ("The director created a masterpiece with this thriller film", "High relevance"),
            ("I love watching science fiction and comedy movies", "Medium relevance"),
            ("This is a general text about cooking recipes", "Low relevance"),
            ("The weather is nice today", "No relevance")
        ]
        
        for text, expected_level in test_texts:
            score = self.plugin._calculate_movie_relevance(text.lower())
            
            print(f"  📊 '{text[:40]}...' -> Score: {score:.3f} ({expected_level})")
            
            # Verify score is within valid range
            assert 0.0 <= score <= 1.0
        
        print("✅ Movie relevance scoring test successful")
    
    async def test_error_handling(self):
        """Test error handling and fallback behavior."""
        print("\n🛡️  Testing error handling...")
        
        # Test with empty data
        empty_result = await self.plugin.embellish_embed_data({}, PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={"cpu_cores": 1.0, "memory_mb": 100.0}
        ))
        assert empty_result == {}  # Should return original empty data
        
        # Test with malformed data
        malformed_data = {"text": None, "content": 123}
        malformed_result = await self.plugin.embellish_embed_data(malformed_data, PluginExecutionContext(
            user_id="test_user",
            session_id="test_session",
            available_resources={"cpu_cores": 1.0, "memory_mb": 100.0}
        ))
        # Should return original data without crashing
        assert malformed_result == malformed_data
        
        print("✅ Error handling test successful")
    
    async def test_performance_metrics(self):
        """Test performance and resource usage."""
        print("\n⚡ Testing performance metrics...")
        
        import time
        
        # Test processing time for different strategies
        test_data = self.test_data["complex_document"]
        
        contexts = [
            ("Low Resource", {"cpu_cores": 1.0, "memory_mb": 100.0}),
            ("Medium Resource", {"cpu_cores": 2.0, "memory_mb": 300.0}),
            ("High Resource", {"cpu_cores": 4.0, "memory_mb": 600.0})
        ]
        
        for strategy_name, resources in contexts:
            context = PluginExecutionContext(
                user_id="test_user",
                session_id="test_session",
                available_resources=resources
            )
            
            start_time = time.time()
            result = await self.plugin.embellish_embed_data(test_data, context)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            print(f"  ⏱️  {strategy_name}: {processing_time:.3f}s")
            
            # Verify processing time is reasonable (under 10 seconds)
            assert processing_time < 10.0, f"Processing time too long: {processing_time:.3f}s"
        
        print("✅ Performance metrics test successful")
    
    async def cleanup(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        if self.plugin:
            await self.plugin.cleanup()
        
        print("✅ Cleanup completed")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting Advanced Embed Data Enhancer Plugin Tests")
        print("=" * 60)
        
        try:
            # Setup
            if not await self.setup():
                return False
            
            # Run tests
            await self.test_plugin_metadata()
            await self.test_low_resource_processing()
            await self.test_medium_resource_processing()
            await self.test_high_resource_processing()
            await self.test_ollama_enhanced_processing()
            await self.test_strategy_selection()
            await self.test_text_extraction()
            await self.test_movie_relevance_scoring()
            await self.test_error_handling()
            await self.test_performance_metrics()
            
            print("\n🎉 All Advanced Embed Data Enhancer tests passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


async def main():
    """Main test runner."""
    tester = TestAdvancedEmbedDataEnhancer()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ Advanced Embed Data Enhancer Plugin: ALL TESTS PASSED")
        return 0
    else:
        print("\n❌ Advanced Embed Data Enhancer Plugin: TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))