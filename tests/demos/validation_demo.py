#!/usr/bin/env python3
"""
VALIDATION RULES DEMONSTRATION
Shows actual validation rules from YAML in action with real data
"""

import asyncio
import sys
from pathlib import Path
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion_manager import IngestionManager


def show_validation_test(title: str, data: dict, expected_result: str):
    """Show a validation test case."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"Expected: {expected_result}")
    print(f"{'='*60}")
    for key, value in data.items():
        print(f"   {key}: {value}")


async def test_validation_rules():
    """Test actual validation rules from the YAML config."""
    
    print("🔍 VALIDATION RULES DEMONSTRATION")
    print("Testing real validation constraints from config/media_types/movie.yaml")
    
    async with IngestionManager(media_type="movie") as manager:
        
        # Show the actual validation rules from YAML
        print(f"\n{'='*60}")
        print("📋 VALIDATION RULES FROM YAML CONFIG")
        print(f"{'='*60}")
        
        if manager.media_config.validation:
            print(f"Required fields: {manager.media_config.validation.get('required_fields', [])}")
            
            constraints = manager.media_config.validation.get('field_constraints', {})
            for field, rules in constraints.items():
                print(f"\n🎯 {field}:")
                for rule, value in rules.items():
                    if rule == 'allowed_values' and isinstance(value, list) and len(value) > 5:
                        print(f"   {rule}: [{', '.join(value[:5])}, ...] ({len(value)} total)")
                    else:
                        print(f"   {rule}: {value}")
        
        # ==========================================================================
        # TEST 1: Valid data that should pass all validation
        # ==========================================================================
        valid_movie = {
            "Name": "Valid Movie",
            "Id": "valid-movie-001",
            "ProductionYear": 2020,  # Valid: between 1888-2030
            "CommunityRating": 7.5,  # Valid: between 0.0-10.0
            "Genres": ["Action", "Drama"],  # Valid: from allowed list
            "Overview": "A perfectly valid movie description."
        }
        
        show_validation_test("TEST 1: VALID DATA", valid_movie, "✅ SHOULD PASS")
        
        try:
            validated = manager.dynamic_model(**valid_movie)
            print(f"✅ VALIDATION PASSED")
            print(f"   Created model: {type(validated).__name__}")
            print(f"   Fields validated: {len(validated.model_dump())}")
        except ValidationError as e:
            print(f"❌ UNEXPECTED FAILURE: {e}")
        
        # ==========================================================================
        # TEST 2: Invalid ProductionYear (too old)
        # ==========================================================================
        invalid_year_old = {
            "Name": "Ancient Movie",
            "Id": "ancient-movie-001", 
            "ProductionYear": 1800,  # Invalid: before 1888
            "CommunityRating": 7.5,
            "Genres": ["Drama"]
        }
        
        show_validation_test("TEST 2: INVALID YEAR (TOO OLD)", invalid_year_old, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**invalid_year_old)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 3: Invalid ProductionYear (future)
        # ==========================================================================
        invalid_year_future = {
            "Name": "Future Movie",
            "Id": "future-movie-001",
            "ProductionYear": 2050,  # Invalid: after 2030
            "CommunityRating": 8.0,
            "Genres": ["Science Fiction"]
        }
        
        show_validation_test("TEST 3: INVALID YEAR (TOO FUTURE)", invalid_year_future, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**invalid_year_future)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 4: Invalid CommunityRating (too low)
        # ==========================================================================
        invalid_rating_low = {
            "Name": "Terrible Movie",
            "Id": "terrible-movie-001",
            "ProductionYear": 2020,
            "CommunityRating": -1.0,  # Invalid: below 0.0
            "Genres": ["Horror"]
        }
        
        show_validation_test("TEST 4: INVALID RATING (TOO LOW)", invalid_rating_low, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**invalid_rating_low)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 5: Invalid CommunityRating (too high)
        # ==========================================================================
        invalid_rating_high = {
            "Name": "Perfect Movie",
            "Id": "perfect-movie-001",
            "ProductionYear": 2020,
            "CommunityRating": 15.0,  # Invalid: above 10.0
            "Genres": ["Fantasy"]
        }
        
        show_validation_test("TEST 5: INVALID RATING (TOO HIGH)", invalid_rating_high, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**invalid_rating_high)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 6: Invalid Genre (not in allowed list)
        # ==========================================================================
        invalid_genre = {
            "Name": "Unknown Genre Movie",
            "Id": "unknown-genre-001",
            "ProductionYear": 2020,
            "CommunityRating": 7.0,
            "Genres": ["Fake Genre", "Made Up Category"]  # Invalid: not in allowed list
        }
        
        show_validation_test("TEST 6: INVALID GENRES", invalid_genre, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**invalid_genre)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 7: Missing required fields
        # ==========================================================================
        missing_required = {
            "Overview": "A movie without required fields",
            "ProductionYear": 2020
            # Missing "Name" and "Id" which are required
        }
        
        show_validation_test("TEST 7: MISSING REQUIRED FIELDS", missing_required, "❌ SHOULD FAIL")
        
        try:
            validated = manager.dynamic_model(**missing_required)
            print(f"❌ UNEXPECTED SUCCESS: Validation should have failed!")
        except ValidationError as e:
            print(f"✅ VALIDATION CORRECTLY FAILED")
            for error in e.errors():
                print(f"   Error: {error['msg']}")
                print(f"   Field: {error['loc']}")
                print(f"   Input: {error['input']}")
        
        # ==========================================================================
        # TEST 8: Show the actual allowed genres from YAML
        # ==========================================================================
        print(f"\n{'='*60}")
        print("📋 ACTUAL ALLOWED GENRES FROM YAML")
        print(f"{'='*60}")
        
        constraints = manager.media_config.validation.get('field_constraints', {})
        if 'Genres' in constraints and 'allowed_values' in constraints['Genres']:
            allowed_genres = constraints['Genres']['allowed_values']
            print(f"Total allowed genres: {len(allowed_genres)}")
            print("Allowed genres:")
            for i, genre in enumerate(allowed_genres, 1):
                print(f"   {i:2d}. {genre}")
        
        # ==========================================================================
        # TEST 9: Valid genre combinations
        # ==========================================================================
        valid_genres_test = {
            "Name": "Multi-Genre Movie",
            "Id": "multi-genre-001",
            "ProductionYear": 1995,
            "CommunityRating": 8.5,
            "Genres": ["Action", "Adventure", "Comedy", "Drama"]  # All valid from allowed list
        }
        
        show_validation_test("TEST 9: VALID MULTIPLE GENRES", valid_genres_test, "✅ SHOULD PASS")
        
        try:
            validated = manager.dynamic_model(**valid_genres_test)
            print(f"✅ VALIDATION PASSED")
            print(f"   Genres accepted: {validated.Genres}")
        except ValidationError as e:
            print(f"❌ UNEXPECTED FAILURE: {e}")
        
        # ==========================================================================
        # SUMMARY
        # ==========================================================================
        print(f"\n{'='*60}")
        print("🎯 VALIDATION SUMMARY")
        print(f"{'='*60}")
        print("The YAML config enforces these real validation rules:")
        print("✅ ProductionYear: Must be between 1888-2030")
        print("✅ CommunityRating: Must be between 0.0-10.0") 
        print("✅ Genres: Must be from predefined list of 21 valid genres")
        print("✅ Required fields: Name and Id must be present")
        print("✅ Dynamic model creation: Based on YAML field constraints")


async def main():
    await test_validation_rules()


if __name__ == "__main__":
    asyncio.run(main())