#!/usr/bin/env python3
"""
Debug script for movie search issues.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from search.movie_query_analyzer import MovieQueryAnalyzer
from search.genre_classifier import GenreClassifier

def debug_query_analyzer():
    """Debug the query analyzer issues."""
    print("=== DEBUGGING QUERY ANALYZER ===")
    
    analyzer = MovieQueryAnalyzer()
    
    # Test cases that are failing
    test_cases = [
        "movies with Tom Cruise",
        "directed by Christopher Nolan", 
        "dark psychological thrillers from the 2010s",
        "action movies from 2020",
        "movies from 2020",
        "action movies with Tom Cruise from the 2000s",
        "dark psychological thrillers directed by Christopher Nolan"
    ]
    
    for query in test_cases:
        print(f"\nQuery: '{query}'")
        result = analyzer.analyze_query(query)
        
        print(f"  Intent: {result.intent}")
        print(f"  Actors: {result.entities.actors}")
        print(f"  Directors: {result.entities.directors}")
        print(f"  Genres: {result.entities.genres}")
        print(f"  Themes: {result.entities.themes}")
        print(f"  Years: {result.entities.years}")
        print(f"  Decades: {result.entities.decades}")
        print(f"  Filters: {result.filters}")
        print(f"  Confidence: {result.confidence:.2f}")

def debug_genre_classifier():
    """Debug the genre classifier issues."""
    print("\n=== DEBUGGING GENRE CLASSIFIER ===")
    
    classifier = GenreClassifier()
    
    # Test cases that are failing
    test_cases = [
        "A romantic comedy about two people who fall in love while solving a mystery",
        "A cyberpunk thriller set in a dystopian future with hackers and virtual reality",
        "dark psychological thrillers directed by Christopher Nolan"
    ]
    
    for text in test_cases:
        print(f"\nText: '{text}'")
        result = classifier.classify_genres(text)
        
        print(f"  Primary genres: {[(g.genre, g.confidence) for g in result.primary_genres]}")
        print(f"  Secondary genres: {[(g.genre, g.confidence) for g in result.secondary_genres]}")
        print(f"  Subgenres: {[(sg.genre, sg.confidence) for sg in result.subgenres]}")
        print(f"  Genre score: {result.genre_score:.2f}")

if __name__ == "__main__":
    debug_query_analyzer()
    debug_genre_classifier()