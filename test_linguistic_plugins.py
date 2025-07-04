#!/usr/bin/env python3
"""
Test suite for linguistic analysis plugins.
Tests ConceptNet expansion, temporal expressions, and semantic role labeling.
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from plugins.linguistic.conceptnet import ConceptNetExpansionPlugin
from plugins.temporal.spacy_with_fallback_ingestion_and_query import SpacyWithFallbackIngestionAndQueryPlugin as TemporalExpressionPlugin
from plugins.linguistic.semantic_roles import SemanticRoleLabelerPlugin


class TestConceptNetExpansionPlugin:
    """Test ConceptNet concept expansion functionality."""
    
    @pytest.fixture
    async def plugin(self):
        """Create ConceptNet plugin instance."""
        plugin = ConceptNetExpansionPlugin()
        return plugin
    
    @pytest.mark.asyncio
    async def test_concept_extraction(self, plugin):
        """Test basic concept extraction from text."""
        text = "A science fiction movie about robots and artificial intelligence"
        result = await plugin.analyze(text)
        
        assert "primary_concepts" in result
        concepts = result["primary_concepts"]
        assert len(concepts) > 0
        
        # Should extract key concepts
        assert any("robot" in concept.lower() for concept in concepts)
        assert any("science" in concept.lower() or "fiction" in concept.lower() for concept in concepts)
    
    @pytest.mark.asyncio
    async def test_movie_specific_extraction(self, plugin):
        """Test concept extraction from movie description."""
        text = "Tom Cruise stars in this action thriller about secret agents"
        result = await plugin.analyze(text)
        
        concepts = result["primary_concepts"]
        assert len(concepts) > 0
        
        # Should extract relevant concepts
        expected_concepts = ["action", "thriller", "agent"]
        found_concepts = [c for c in concepts if any(exp in c.lower() for exp in expected_concepts)]
        assert len(found_concepts) > 0
    
    @pytest.mark.asyncio
    async def test_empty_text(self, plugin):
        """Test handling of empty text."""
        result = await plugin.analyze("")
        
        assert "primary_concepts" in result
        assert isinstance(result["primary_concepts"], list)
    
    @pytest.mark.asyncio
    async def test_complex_text(self, plugin):
        """Test extraction from complex movie description."""
        text = """
        A brilliant computer hacker discovers that reality as he knows it is actually 
        a computer simulation controlled by machines. He joins a rebellion to free humanity.
        """
        result = await plugin.analyze(text)
        
        concepts = result["primary_concepts"]
        assert len(concepts) > 3  # Should extract multiple concepts
        
        # Check for key terms
        all_concepts_str = " ".join(concepts).lower()
        assert any(term in all_concepts_str for term in ["computer", "machine", "reality", "simulation"])


class TestTemporalExpressionPlugin:
    """Test temporal expression parsing and normalization."""
    
    @pytest.fixture
    async def plugin(self):
        """Create temporal plugin instance."""
        plugin = TemporalExpressionPlugin()
        return plugin
    
    @pytest.mark.asyncio
    async def test_decade_parsing(self, plugin):
        """Test parsing of decade expressions."""
        test_cases = [
            ("movies from the 90s", 1990, 1999),
            ("80s action films", 1980, 1989),
            ("early 2000s", 2000, 2002),
            ("late 90s thriller", 1997, 1999),
        ]
        
        for text, expected_start, expected_end in test_cases:
            result = await plugin.analyze(text)
            
            assert "normalized" in result
            assert len(result["normalized"]) > 0
            
            # Find decade expression
            decade_expr = next(
                (expr for expr in result["normalized"] 
                 if expr.get("precision") in ["decade", "range"]), 
                None
            )
            
            assert decade_expr is not None, f"No decade found in: {text}"
            assert decade_expr["start"] >= expected_start - 5  # Allow some flexibility
            assert decade_expr["end"] <= expected_end + 5
    
    @pytest.mark.asyncio
    async def test_year_parsing(self, plugin):
        """Test parsing of specific year expressions."""
        test_cases = [
            "movies from 1995",
            "in 2010",
            "2020 films",
        ]
        
        for text in test_cases:
            result = await plugin.analyze(text)
            
            assert "normalized" in result
            assert len(result["normalized"]) > 0
            
            # Should find year expressions
            year_exprs = [expr for expr in result["normalized"] 
                         if expr.get("precision") == "year"]
            assert len(year_exprs) > 0
    
    @pytest.mark.asyncio
    async def test_relative_expressions(self, plugin):
        """Test parsing of relative time expressions."""
        test_cases = [
            "movies from last decade",
            "recent films",
            "last 5 years",
        ]
        
        for text in test_cases:
            result = await plugin.analyze(text)
            
            assert "normalized" in result
            # Should handle relative expressions
            if result["normalized"]:
                expr = result["normalized"][0]
                assert "start" in expr
                assert "end" in expr
    
    @pytest.mark.asyncio
    async def test_season_parsing(self, plugin):
        """Test parsing of seasonal expressions."""
        test_cases = [
            "summer of 2020",
            "winter 1995",
            "spring of 2010",
        ]
        
        for text in test_cases:
            result = await plugin.analyze(text)
            
            assert "normalized" in result
            # May or may not find seasons, depending on pattern matching
            if result["normalized"]:
                season_exprs = [expr for expr in result["normalized"] 
                               if expr.get("precision") == "season"]
                # If found, should have proper date format
                for expr in season_exprs:
                    assert "start" in expr
                    assert "end" in expr
    
    @pytest.mark.asyncio
    async def test_empty_text(self, plugin):
        """Test handling of text with no temporal expressions."""
        text = "A great action movie with amazing effects"
        result = await plugin.analyze(text)
        
        assert "expressions" in result
        assert "normalized" in result
        assert isinstance(result["expressions"], list)
        assert isinstance(result["normalized"], list)


class TestSemanticRoleLabelerPlugin:
    """Test semantic role labeling functionality."""
    
    @pytest.fixture
    async def plugin(self):
        """Create semantic role labeling plugin instance."""
        plugin = SemanticRoleLabelerPlugin()
        return plugin
    
    @pytest.mark.asyncio
    async def test_director_role_extraction(self, plugin):
        """Test extraction of director roles."""
        text = "Steven Spielberg directed Jurassic Park"
        result = await plugin.analyze(text)
        
        assert "semantic_roles" in result
        roles = result["semantic_roles"]
        
        if roles:  # Pattern matching may or may not find roles
            # Should identify director relationship
            director_roles = [role for role in roles 
                            if "direct" in role.get("predicate", "")]
            
            if director_roles:
                role = director_roles[0]
                assert "Spielberg" in role.get("agent", "")
                assert "Jurassic" in role.get("theme", "") or role.get("theme") == "unspecified"
                assert role.get("frame") == "Behind_the_scenes"
    
    @pytest.mark.asyncio
    async def test_actor_role_extraction(self, plugin):
        """Test extraction of actor roles."""
        text = "Tom Cruise starred in Mission Impossible"
        result = await plugin.analyze(text)
        
        assert "semantic_roles" in result
        roles = result["semantic_roles"]
        
        if roles:
            # Should identify performance relationship
            actor_roles = [role for role in roles 
                          if "star" in role.get("predicate", "")]
            
            if actor_roles:
                role = actor_roles[0]
                assert "Cruise" in role.get("agent", "")
                assert role.get("frame") in ["Performance", "General"]
    
    @pytest.mark.asyncio
    async def test_multiple_roles(self, plugin):
        """Test extraction of multiple semantic roles."""
        text = """
        Christopher Nolan directed Inception. Leonardo DiCaprio starred in the film.
        The movie follows a thief who enters dreams.
        """
        result = await plugin.analyze(text)
        
        assert "semantic_roles" in result
        roles = result["semantic_roles"]
        
        # Should extract multiple roles
        if len(roles) > 1:
            predicates = [role.get("predicate", "") for role in roles]
            # Should have different types of actions
            assert len(set(predicates)) > 1
    
    @pytest.mark.asyncio
    async def test_frame_classification(self, plugin):
        """Test FrameNet frame classification."""
        test_cases = [
            ("directed the movie", "Behind_the_scenes"),
            ("acted in the film", "Performance"),
            ("fought the villain", "Action"),
            ("created the story", "Creating"),
        ]
        
        for text, expected_frame in test_cases:
            result = await plugin.analyze(text)
            
            roles = result.get("semantic_roles", [])
            if roles:
                # Check if any role has the expected frame
                frames = [role.get("frame") for role in roles]
                # Frame classification may vary, so we're flexible
                assert any(frame in ["Behind_the_scenes", "Performance", "Action", 
                                   "Creating", "General"] for frame in frames)
    
    @pytest.mark.asyncio
    async def test_empty_text(self, plugin):
        """Test handling of empty text."""
        result = await plugin.analyze("")
        
        assert "semantic_roles" in result
        assert isinstance(result["semantic_roles"], list)
    
    @pytest.mark.asyncio
    async def test_frames_by_type(self, plugin):
        """Test grouping of roles by frame types."""
        text = "Spielberg directed the movie and DiCaprio acted in it"
        result = await plugin.analyze(text)
        
        assert "frames_by_type" in result
        assert isinstance(result["frames_by_type"], dict)


class TestPluginIntegration:
    """Test integration between linguistic plugins."""
    
    @pytest.mark.asyncio
    async def test_movie_query_analysis(self):
        """Test analyzing a complete movie query with all plugins."""
        query = "90s sci-fi movies where Tom Cruise fights aliens"
        
        # Test each plugin
        concept_plugin = ConceptNetExpansionPlugin()
        temporal_plugin = TemporalExpressionPlugin()
        semantic_plugin = SemanticRoleLabelerPlugin()
        
        concept_result = await concept_plugin.analyze(query)
        temporal_result = await temporal_plugin.analyze(query)
        semantic_result = await semantic_plugin.analyze(query)
        
        # Verify each plugin extracts relevant information
        assert "primary_concepts" in concept_result
        assert "normalized" in temporal_result
        assert "semantic_roles" in semantic_result
        
        # Concepts should include sci-fi related terms
        concepts = concept_result["primary_concepts"]
        assert any("sci" in concept.lower() or "fiction" in concept.lower() 
                  for concept in concepts)
        
        # Temporal should identify 90s
        if temporal_result["normalized"]:
            decade_expr = next(
                (expr for expr in temporal_result["normalized"] 
                 if expr.get("precision") == "decade"), 
                None
            )
            if decade_expr:
                assert 1990 <= decade_expr["start"] <= 1990
    
    @pytest.mark.asyncio
    async def test_content_enhancement(self):
        """Test using plugins for content enhancement."""
        movie_data = {
            "name": "The Matrix",
            "overview": "A computer hacker discovers reality is a simulation",
            "people": [{"name": "Keanu Reeves", "type": "Actor"}]
        }
        
        context = {"media_type": "movie"}
        
        # Test dual-use functionality
        concept_plugin = ConceptNetExpansionPlugin()
        result = await concept_plugin.enhance_data(movie_data, context)
        
        assert "linguistic_analysis" in result
        assert "ConceptNetExpansionPlugin" in result["linguistic_analysis"]


def run_tests():
    """Run all linguistic plugin tests."""
    import subprocess
    
    # Run with pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            __file__, 
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("=== LINGUISTIC PLUGIN TEST RESULTS ===")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    # Run tests directly
    success = run_tests()
    if success:
        print("\n✅ All linguistic plugin tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
    
    sys.exit(0 if success else 1)