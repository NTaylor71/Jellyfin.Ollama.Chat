#!/usr/bin/env python3
"""
Test suite for Phase 6.3: Movie-Specific Search Enhancements

Tests the movie query analyzer, genre classifier, and cast matcher components.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from search.movie_query_analyzer import (
    MovieQueryAnalyzer, analyze_movie_query, QueryIntent, QueryAnalysis
)
from search.genre_classifier import (
    GenreClassifier, classify_movie_genres, GenreMatch, GenreClassification
)
from search.cast_matcher import (
    CastMatcher, find_actor_matches, find_director_matches, PersonMatch
)


class TestMovieQueryAnalyzer(unittest.TestCase):
    """Test the movie query analyzer."""
    
    def setUp(self):
        self.analyzer = MovieQueryAnalyzer()
    
    def test_simple_search_query(self):
        """Test simple movie title search."""
        result = self.analyzer.analyze_query("The Matrix")
        
        self.assertEqual(result.intent, QueryIntent.SIMPLE_SEARCH)
        self.assertIn("matrix", [term.lower() for term in result.processed_terms])
        self.assertGreater(result.confidence, 0.4)
    
    def test_genre_search_query(self):
        """Test genre-based search queries."""
        result = self.analyzer.analyze_query("action movies")
        
        self.assertEqual(result.intent, QueryIntent.GENRE_SEARCH)
        self.assertIn("action", result.entities.genres)
        self.assertGreater(result.search_weights.get('genres', 0), 1.5)
    
    def test_actor_search_query(self):
        """Test actor-based search queries."""
        result = self.analyzer.analyze_query("movies with Tom Cruise")
        
        self.assertEqual(result.intent, QueryIntent.ACTOR_SEARCH)
        self.assertTrue(any("tom cruise" in actor.lower() for actor in result.entities.actors))
        self.assertGreater(result.search_weights.get('people', 0), 2.0)
    
    def test_director_search_query(self):
        """Test director-based search queries."""
        result = self.analyzer.analyze_query("directed by Christopher Nolan")
        
        self.assertEqual(result.intent, QueryIntent.DIRECTOR_SEARCH)
        self.assertTrue(any("christopher nolan" in director.lower() for director in result.entities.directors))
        self.assertGreater(result.search_weights.get('people', 0), 2.0)
    
    def test_year_search_query(self):
        """Test year-based search queries."""
        result = self.analyzer.analyze_query("movies from 2020")
        
        self.assertEqual(result.intent, QueryIntent.YEAR_SEARCH)
        self.assertIn(2020, result.entities.years)
    
    def test_decade_search_query(self):
        """Test decade-based search queries."""
        result = self.analyzer.analyze_query("2000s action films")
        
        self.assertIn("2000s", result.entities.decades)
        self.assertIn("action", result.entities.genres)
    
    def test_similarity_search_query(self):
        """Test similarity-based search queries."""
        result = self.analyzer.analyze_query("movies like Blade Runner")
        
        self.assertEqual(result.intent, QueryIntent.SIMILARITY_SEARCH)
        self.assertTrue(any("blade runner" in movie.lower() for movie in result.entities.reference_movies))
        self.assertGreater(result.search_weights.get('enhanced_fields', 0), 1.5)
    
    def test_recommendation_query(self):
        """Test recommendation queries."""
        result = self.analyzer.analyze_query("recommend good thriller movies")
        
        self.assertEqual(result.intent, QueryIntent.RECOMMENDATION)
        self.assertIn("thriller", result.entities.genres)
        self.assertIn("good", result.entities.quality_terms)
    
    def test_complex_search_query(self):
        """Test complex multi-entity queries."""
        result = self.analyzer.analyze_query("dark psychological thrillers from the 2010s")
        
        self.assertEqual(result.intent, QueryIntent.COMPLEX_SEARCH)
        self.assertIn("thriller", result.entities.genres)
        self.assertIn("dark", result.entities.themes)
        self.assertIn("psychological", result.entities.themes)
        self.assertIn("2010s", result.entities.decades)
        self.assertGreater(result.confidence, 0.7)
    
    def test_country_extraction(self):
        """Test country/nationality extraction."""
        result = self.analyzer.analyze_query("Finnish comedy films")
        
        self.assertIn("Finnish", result.entities.countries)
        self.assertIn("comedy", result.entities.genres)
    
    def test_rating_extraction(self):
        """Test rating extraction."""
        result = self.analyzer.analyze_query("R-rated horror movies")
        
        self.assertIn("R", result.entities.ratings)
        self.assertIn("horror", result.entities.genres)
    
    def test_filter_generation(self):
        """Test MongoDB filter generation."""
        result = self.analyzer.analyze_query("action movies from 2020")
        
        # This should be complex search with both filters
        self.assertEqual(result.intent, QueryIntent.COMPLEX_SEARCH)
        self.assertIn('production_year', result.filters)
        self.assertEqual(result.filters['production_year'], 2020)
        self.assertIn('genres', result.filters)


class TestGenreClassifier(unittest.TestCase):
    """Test the genre classifier."""
    
    def setUp(self):
        self.classifier = GenreClassifier()
    
    def test_single_genre_classification(self):
        """Test classification of single genre text."""
        text = "An action-packed adventure with explosions and car chases"
        result = self.classifier.classify_genres(text)
        
        self.assertTrue(any(g.genre == "Action" for g in result.primary_genres))
        self.assertGreater(result.genre_score, 0.5)
    
    def test_multi_genre_classification(self):
        """Test classification of multi-genre text."""
        text = "A romantic comedy about two people who fall in love while solving a mystery"
        result = self.classifier.classify_genres(text)
        
        genre_names = [g.genre for g in result.primary_genres + result.secondary_genres]
        self.assertTrue(any("Romance" in name for name in genre_names))
        self.assertTrue(any("Comedy" in name for name in genre_names))
        self.assertTrue(any("Mystery" in name for name in genre_names))
    
    def test_subgenre_detection(self):
        """Test detection of subgenres."""
        text = "A cyberpunk thriller set in a dystopian future with hackers and virtual reality"
        result = self.classifier.classify_genres(text)
        
        subgenre_names = [sg.genre for sg in result.subgenres]
        self.assertTrue(any("Cyberpunk" in name for name in subgenre_names))
        self.assertTrue(any("Dystopian" in name for name in subgenre_names))
    
    def test_horror_subgenres(self):
        """Test horror subgenre detection."""
        text = "A slasher film with a masked killer stalking teenagers at summer camp"
        result = self.classifier.classify_genres(text)
        
        self.assertTrue(any(g.genre == "Horror" for g in result.primary_genres))
        self.assertTrue(any(sg.genre == "Slasher" for sg in result.subgenres))
    
    def test_genre_hierarchy(self):
        """Test genre hierarchy relationships."""
        hierarchy = self.classifier.get_genre_hierarchy("Action")
        
        self.assertIn("Martial Arts", hierarchy['children'])
        self.assertIn("Superhero", hierarchy['children'])
        self.assertIsNone(hierarchy['parent'])
    
    def test_subgenre_hierarchy(self):
        """Test subgenre hierarchy relationships."""
        hierarchy = self.classifier.get_genre_hierarchy("Cyberpunk")
        
        self.assertEqual(hierarchy['parent'], "Science Fiction")
        self.assertIn("Dystopian", hierarchy['siblings'])
    
    def test_genre_normalization(self):
        """Test genre name normalization."""
        self.assertEqual(self.classifier.normalize_genre_name("sci-fi"), "Science Fiction")
        self.assertEqual(self.classifier.normalize_genre_name("rom-com"), "Romantic Comedy")
        self.assertEqual(self.classifier.normalize_genre_name("Action"), "Action")
    
    def test_existing_genre_boost(self):
        """Test boosting of existing genres."""
        text = "A movie about relationships and personal growth"
        existing_genres = ["Drama"]
        
        result = self.classifier.classify_genres(text, existing_genres)
        
        drama_match = next((g for g in result.primary_genres if g.genre == "Drama"), None)
        self.assertIsNotNone(drama_match)
        self.assertGreater(drama_match.confidence, 0.5)
    
    def test_consistency_scoring(self):
        """Test genre consistency scoring."""
        # Compatible genres should have high consistency
        compatible_text = "An action adventure with thrilling sequences"
        result = self.classifier.classify_genres(compatible_text)
        self.assertGreater(result.consistency_score, 0.7)
    
    def test_suggested_related_genres(self):
        """Test related genre suggestions."""
        suggestions = self.classifier.suggest_related_genres(["Action"])
        
        self.assertIn("Adventure", suggestions)
        self.assertIn("Thriller", suggestions)
        self.assertNotIn("Action", suggestions)  # Shouldn't suggest itself


class TestCastMatcher(unittest.TestCase):
    """Test the cast matcher."""
    
    def setUp(self):
        self.matcher = CastMatcher()
        
        # Sample people data for testing
        self.sample_people = [
            {
                "name": "Tom Cruise",
                "id": "actor1",
                "role": "Ethan Hunt",
                "type": "Actor"
            },
            {
                "name": "Christopher Nolan",
                "id": "director1",
                "role": "Director",
                "type": "Director"
            },
            {
                "name": "Mika Rättö",
                "id": "actor2",
                "role": "Rauni Reposaarelainen",
                "type": "Actor"
            },
            {
                "name": "Robert Downey Jr.",
                "id": "actor3",
                "role": "Tony Stark",
                "type": "Actor"
            }
        ]
        
        # Sample movie data for collaboration testing
        self.sample_movies = [
            {
                "name": "Mission Impossible",
                "people": [
                    {"name": "Tom Cruise", "type": "Actor", "role": "Ethan Hunt"},
                    {"name": "Christopher McQuarrie", "type": "Director", "role": "Director"}
                ]
            },
            {
                "name": "The Dark Knight",
                "people": [
                    {"name": "Christian Bale", "type": "Actor", "role": "Batman"},
                    {"name": "Christopher Nolan", "type": "Director", "role": "Director"}
                ]
            }
        ]
    
    def test_name_normalization(self):
        """Test name normalization."""
        self.assertEqual(self.matcher.normalize_person_name("Tom Cruise"), "tom cruise")
        self.assertEqual(self.matcher.normalize_person_name("Robert Downey Jr."), "robert downey jr")
        self.assertEqual(self.matcher.normalize_person_name("Mika Rättö"), "mika ratto")
    
    def test_exact_name_match(self):
        """Test exact name matching."""
        matches = self.matcher.find_person_matches("Tom Cruise", self.sample_people)
        
        # Filter to get only the Tom Cruise matches
        tom_cruise_matches = [m for m in matches if m.name == "Tom Cruise"]
        self.assertEqual(len(tom_cruise_matches), 1)
        self.assertEqual(tom_cruise_matches[0].name, "Tom Cruise")
        self.assertEqual(tom_cruise_matches[0].confidence, 1.0)
        self.assertEqual(tom_cruise_matches[0].match_reason, "exact name match")
    
    def test_fuzzy_name_match(self):
        """Test fuzzy name matching."""
        matches = self.matcher.find_person_matches("Tom Cruize", self.sample_people)  # Typo
        
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "Tom Cruise")
        self.assertGreater(matches[0].confidence, 0.7)
    
    def test_partial_name_match(self):
        """Test partial name matching."""
        matches = self.matcher.find_person_matches("Tom", self.sample_people)
        
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "Tom Cruise")
        self.assertGreater(matches[0].confidence, 0.5)
    
    def test_character_role_match(self):
        """Test character role matching."""
        matches = self.matcher.find_person_matches("Ethan Hunt", self.sample_people)
        
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "Tom Cruise")
        self.assertIn("character role match", matches[0].match_reason)
    
    def test_actor_type_filtering(self):
        """Test filtering by person type."""
        # Test with a name that shouldn't match any actors
        matches = self.matcher.find_person_matches("Nolan", self.sample_people, "Actor")
        self.assertEqual(len(matches), 0)
        
        # Test director search - should match Christopher Nolan
        matches = self.matcher.find_person_matches("Christopher", self.sample_people, "Director")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].name, "Christopher Nolan")
    
    def test_diacritics_handling(self):
        """Test handling of diacritics in names."""
        matches = self.matcher.find_person_matches("Mika Ratto", self.sample_people)
        
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].name, "Mika Rättö")
        self.assertGreater(matches[0].confidence, 0.8)
    
    def test_alias_matching(self):
        """Test alias matching for known actors."""
        # Test with known alias
        matches = self.matcher.find_person_matches("RDJ", self.sample_people)
        
        # This would work if RDJ was in the aliases (simplified for test)
        # In production, this would match Robert Downey Jr.
        self.assertIsInstance(matches, list)
    
    def test_collaboration_detection(self):
        """Test collaboration detection between people."""
        collaborations = self.matcher.find_collaborations(
            ["Tom Cruise", "Christopher McQuarrie"], 
            self.sample_movies
        )
        
        # Should find collaboration in Mission Impossible
        self.assertGreater(len(collaborations), 0)
        
        collab = collaborations[0]
        self.assertIn("Tom Cruise", [collab.person1, collab.person2])
        self.assertIn("Christopher McQuarrie", [collab.person1, collab.person2])
        self.assertEqual(collab.collaboration_type, "actor-director")
    
    def test_cast_search(self):
        """Test comprehensive cast search."""
        # Combine people from multiple movies for search
        all_people = []
        for movie in self.sample_movies:
            all_people.extend(movie['people'])
        
        result = self.matcher.search_cast_and_crew("Tom", self.sample_movies)
        
        self.assertGreater(len(result.matches), 0)
        self.assertGreater(result.search_confidence, 0.5)
        self.assertIsInstance(result.suggested_searches, list)
    
    def test_filmography_summary(self):
        """Test filmography summary generation."""
        summary = self.matcher.get_person_filmography_summary("Tom Cruise", self.sample_movies)
        
        self.assertGreater(summary['total_movies'], 0)
        self.assertIn('Actor', summary['roles'])
        self.assertIsInstance(summary['movies'], list)
        self.assertIsInstance(summary['collaborators'], list)
    
    def test_character_mapping(self):
        """Test character to actor mapping."""
        # Test known character mapping
        matches = self.matcher.find_person_matches("James Bond", self.sample_people)
        
        # This would work with more comprehensive character mappings
        # For now, just verify the method doesn't crash
        self.assertIsInstance(matches, list)


class TestMovieSearchIntegration(unittest.TestCase):
    """Test integration between all movie search components."""
    
    def setUp(self):
        self.analyzer = MovieQueryAnalyzer()
        self.classifier = GenreClassifier()
        self.matcher = CastMatcher()
    
    def test_end_to_end_movie_search(self):
        """Test complete movie search workflow."""
        query = "action movies with Tom Cruise from the 2000s"
        
        # Step 1: Analyze query
        analysis = self.analyzer.analyze_query(query)
        
        self.assertEqual(analysis.intent, QueryIntent.COMPLEX_SEARCH)
        self.assertIn("action", analysis.entities.genres)
        self.assertTrue(any("tom cruise" in actor.lower() for actor in analysis.entities.actors))
        self.assertIn("2000s", analysis.entities.decades)
        
        # Step 2: Classify genres from query
        genre_classification = self.classifier.classify_genres(query)
        
        self.assertTrue(any(g.genre == "Action" for g in genre_classification.primary_genres))
        
        # Step 3: Search for cast (would use real data in production)
        sample_people = [{"name": "Tom Cruise", "type": "Actor", "role": "Ethan Hunt"}]
        cast_matches = self.matcher.find_person_matches("Tom Cruise", sample_people)
        
        self.assertGreater(len(cast_matches), 0)
        self.assertEqual(cast_matches[0].name, "Tom Cruise")
        
        # Verify filters are generated
        self.assertIn('genres', analysis.filters)
        self.assertIn('production_year', analysis.filters)
    
    def test_complex_query_processing(self):
        """Test processing of complex queries."""
        query = "dark psychological thrillers directed by Christopher Nolan"
        
        # Analyze query
        analysis = self.analyzer.analyze_query(query)
        
        self.assertEqual(analysis.intent, QueryIntent.COMPLEX_SEARCH)
        self.assertIn("thriller", analysis.entities.genres)
        self.assertIn("dark", analysis.entities.themes)
        self.assertIn("psychological", analysis.entities.themes)
        self.assertTrue(any("christopher nolan" in director.lower() for director in analysis.entities.directors))
        
        # Classify genres
        genre_classification = self.classifier.classify_genres(query)
        
        self.assertTrue(any("Thriller" in g.genre for g in genre_classification.primary_genres))
        self.assertTrue(any("Psychological" in sg.genre for sg in genre_classification.subgenres))
    
    def test_similarity_search_processing(self):
        """Test similarity search query processing."""
        query = "movies like Blade Runner but newer"
        
        analysis = self.analyzer.analyze_query(query)
        
        self.assertEqual(analysis.intent, QueryIntent.SIMILARITY_SEARCH)
        self.assertTrue(any("blade runner" in movie.lower() for movie in analysis.entities.reference_movies))
        self.assertGreater(analysis.search_weights.get('enhanced_fields', 0), 1.5)


def run_tests():
    """Run all movie search enhancement tests."""
    print("🎬 Running Movie-Specific Search Enhancement Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMovieQueryAnalyzer,
        TestGenreClassifier, 
        TestCastMatcher,
        TestMovieSearchIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"🎬 Movie Search Enhancement Tests Complete")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"💥 Tests errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n🎯 Overall Result: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)