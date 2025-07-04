#!/usr/bin/env python3
"""
Test suite for ConceptNet rate limiting functionality.
Validates that we respect the API limits with 90% safety margin.
"""

import asyncio
import pytest
import time
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from plugins.linguistic.conceptnet import ConceptNetRateLimiter, ConceptNetExpansionPlugin


class TestConceptNetRateLimiter:
    """Test the ConceptNet rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with 90% safety margins."""
        limiter = ConceptNetRateLimiter()
        
        # Should be 90% of actual limits
        assert limiter.hourly_limit == 3240  # 90% of 3600
        assert limiter.minute_limit == 108   # 90% of 120
        
        # Should start with no requests
        status = limiter.get_status()
        assert status["hourly_usage"] == 0
        assert status["minute_usage"] == 0
        assert status["hourly_remaining"] == 3240
        assert status["minute_remaining"] == 108
    
    def test_endpoint_cost_calculation(self):
        """Test correct cost calculation for different endpoints."""
        limiter = ConceptNetRateLimiter()
        
        # Standard endpoints cost 1
        assert limiter._get_endpoint_cost("http://api.conceptnet.io/c/en/robot") == 1
        
        # Related endpoints cost 2
        assert limiter._get_endpoint_cost("http://api.conceptnet.io/related/c/en/robot") == 2
        assert limiter._get_endpoint_cost("http://api.conceptnet.io/relatedness?node1=/c/en/robot") == 2
        
        # Unknown endpoints default to 1
        assert limiter._get_endpoint_cost("http://api.conceptnet.io/unknown") == 1
    
    def test_request_tracking(self):
        """Test that requests are properly tracked."""
        limiter = ConceptNetRateLimiter()
        
        # Record some requests
        asyncio.run(limiter.record_request("http://api.conceptnet.io/c/en/robot"))  # Cost 1
        asyncio.run(limiter.record_request("http://api.conceptnet.io/related/c/en/robot"))  # Cost 2
        
        status = limiter.get_status()
        assert status["hourly_usage"] == 3  # 1 + 2
        assert status["minute_usage"] == 3  # 1 + 2
        assert status["hourly_remaining"] == 3237  # 3240 - 3
        assert status["minute_remaining"] == 105   # 108 - 3
    
    def test_rate_limit_checking(self):
        """Test rate limit checking before requests."""
        limiter = ConceptNetRateLimiter()
        
        # Should allow requests initially
        url = "http://api.conceptnet.io/c/en/robot"
        assert asyncio.run(limiter.can_make_request(url)) == True
        
        # Simulate filling up the minute limit
        current_time = time.time()
        for i in range(54):  # 54 requests * 2 cost = 108 (at limit)
            limiter.minute_requests.append((current_time, 2))
        
        # Should not allow more requests
        assert asyncio.run(limiter.can_make_request("http://api.conceptnet.io/related/robot")) == False
        
        # Should still allow cheaper requests if within limit
        limiter.minute_requests.clear()
        for i in range(107):  # 107 requests * 1 cost = 107 (1 remaining)
            limiter.minute_requests.append((current_time, 1))
        
        assert asyncio.run(limiter.can_make_request("http://api.conceptnet.io/c/en/robot")) == True
        assert asyncio.run(limiter.can_make_request("http://api.conceptnet.io/related/robot")) == False
    
    def test_old_request_cleanup(self):
        """Test that old requests are cleaned up properly."""
        limiter = ConceptNetRateLimiter()
        
        # Add some old requests
        old_time = time.time() - 3700  # Over 1 hour ago
        limiter.hourly_requests.append((old_time, 5))
        limiter.minute_requests.append((old_time, 5))
        
        # Add recent requests
        recent_time = time.time() - 30  # 30 seconds ago
        limiter.hourly_requests.append((recent_time, 3))
        limiter.minute_requests.append((recent_time, 3))
        
        # Get status (should trigger cleanup)
        status = limiter.get_status()
        
        # Should only count recent requests
        assert status["hourly_usage"] == 3
        assert status["minute_usage"] == 3
    
    def test_percentage_calculations(self):
        """Test percentage usage calculations."""
        limiter = ConceptNetRateLimiter()
        
        # Add some requests
        current_time = time.time()
        limiter.hourly_requests.append((current_time, 324))   # 10% of hourly limit
        limiter.minute_requests.append((current_time, 54))    # 50% of minute limit
        
        status = limiter.get_status()
        assert abs(status["hourly_percent"] - 10.0) < 0.1
        assert abs(status["minute_percent"] - 50.0) < 0.1


class TestConceptNetPluginRateLimiting:
    """Test ConceptNet plugin integration with rate limiting."""
    
    @pytest.fixture
    def plugin(self):
        """Create plugin instance and reset rate limiter."""
        plugin = ConceptNetExpansionPlugin()
        plugin.reset_rate_limiter()  # Start fresh
        return plugin
    
    def test_shared_rate_limiter(self):
        """Test that multiple plugin instances share the same rate limiter."""
        plugin1 = ConceptNetExpansionPlugin()
        plugin2 = ConceptNetExpansionPlugin()
        
        # Should be the same instance
        assert plugin1.rate_limiter is plugin2.rate_limiter
    
    @pytest.mark.asyncio
    async def test_rate_limit_status_in_results(self, plugin):
        """Test that rate limit status is included in analysis results."""
        text = "robot artificial intelligence"
        
        # Mock the ConceptNet API to avoid actual requests
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"edges": []})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await plugin.analyze(text)
            
            assert "rate_limit_status" in result
            status = result["rate_limit_status"]
            assert "minute_remaining" in status
            assert "hourly_remaining" in status
            assert "minute_percent_used" in status
            assert "hourly_percent_used" in status
    
    @pytest.mark.asyncio
    async def test_rate_limit_prevents_expansion(self, plugin):
        """Test that rate limiting prevents concept expansion when at limit."""
        text = "robot artificial intelligence machine learning"
        
        # Fill up the rate limiter
        current_time = time.time()
        for i in range(108):  # Fill minute limit
            plugin.rate_limiter.minute_requests.append((current_time, 1))
        
        result = await plugin.analyze(text)
        
        # Should still extract primary concepts
        assert "primary_concepts" in result
        assert len(result["primary_concepts"]) > 0
        
        # But should not have expanded concepts due to rate limit
        assert result["expansion_count"] == 0
    
    @pytest.mark.asyncio
    async def test_conservative_concept_limiting(self, plugin):
        """Test that plugin limits concepts based on remaining rate limit."""
        text = "robot artificial intelligence machine learning data science computer vision neural network"
        
        # Set up rate limiter with limited remaining requests
        plugin.rate_limiter.minute_requests.clear()
        current_time = time.time()
        for i in range(102):  # Leave only 6 requests remaining
            plugin.rate_limiter.minute_requests.append((current_time, 1))
        
        # Mock successful API responses
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"edges": []})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await plugin.analyze(text)
            
            # Should limit expansions based on remaining rate limit
            # 6 remaining / 2 (conservative estimate) = max 3 concepts
            assert result["expansion_count"] <= 3
    
    @pytest.mark.asyncio
    async def test_cache_bypasses_rate_limiting(self, plugin):
        """Test that cached results bypass rate limiting."""
        concept = "robot"
        
        # Pre-populate cache
        plugin.cache[f"concept_{concept}"] = {
            "data": {"related": ["android", "machine"], "edges": []},
            "timestamp": time.time()
        }
        
        # Fill up rate limiter
        current_time = time.time()
        for i in range(108):
            plugin.rate_limiter.minute_requests.append((current_time, 1))
        
        # Should still get cached result
        result = await plugin._expand_concept(concept)
        assert result is not None
        assert "android" in result["related"]
    
    def test_rate_limiter_status_method(self, plugin):
        """Test the convenience method for getting rate limit status."""
        status = plugin.get_rate_limit_status()
        
        assert "hourly_usage" in status
        assert "minute_usage" in status
        assert "hourly_limit" in status
        assert "minute_limit" in status
        assert status["hourly_limit"] == 3240
        assert status["minute_limit"] == 108


def run_rate_limiting_tests():
    """Run all rate limiting tests."""
    import subprocess
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            __file__, 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("=== CONCEPTNET RATE LIMITING TEST RESULTS ===")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    success = run_rate_limiting_tests()
    if success:
        print("\n✅ All ConceptNet rate limiting tests passed!")
        print("\n📊 Rate Limiting Features:")
        print("  ✅ 90% safety margin (3240/hour, 108/minute)")
        print("  ✅ Endpoint cost tracking (related/relatedness = 2 requests)")
        print("  ✅ Automatic request cleanup after time windows")
        print("  ✅ Conservative concept limiting based on remaining quota")
        print("  ✅ Cache bypasses rate limiting")
        print("  ✅ Shared rate limiter across plugin instances")
        print("  ✅ Rate limit status included in analysis results")
    else:
        print("\n❌ Some rate limiting tests failed. Check output above.")
    
    sys.exit(0 if success else 1)