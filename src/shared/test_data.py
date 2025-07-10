"""
Test data structure for the procedural intelligence pipeline.

Uses actual Jellyfin movie data samples to test Field-Class MediaEntity
and plugin processing. Preserves real complexity for accurate testing.
"""

from typing import Dict, Any, List
from src.shared.media_fields import MediaEntity, FieldType, AnalysisWeight



RAW_JELLYFIN_SAMURAI_RAUNI = {
    "Name": "Samurai Rauni",
    "OriginalTitle": "Samurai Rauni Reposaarelainen",
    "ServerId": "a06ac75a6c2e40aab501522265dcb3c4",
    "Id": "659073c3152c4d5c12b531b00be93071",
    "Etag": "648920dbfce15eaee9a1547d6e739b03",
    "DateCreated": "2024-11-27T20:26:41.6344125Z",
    "Type": "Movie",
    "Overview": "Villagers are afraid of Samurai Rauni Reposaarelainen, who keeps them on their toes every day. When someone places a bounty on Rauni's head, he goes after this mysterious person.",
    "Taglines": ["Who would move a mountain, if not the mountain itself."],
    "Genres": ["Comedy", "Drama"],
    "Tags": ["samurai", "dark comedy", "surrealism", "chambara"],
    "ProductionYear": 2016,
    "ProductionLocations": ["Finland"],
    "CommunityRating": 5.1,
    "OfficialRating": None,
    "People": [
        {
            "Name": "Mika Rättö",
            "Id": "6725d54f7eec7cb122038434c306d144",
            "Role": "Rauni Reposaarelainen",
            "Type": "Actor",
            "PrimaryImageTag": "ef051c4545daa922f476ef1bd4ff6fbd"
        },
        {
            "Name": "Mika Rättö", 
            "Id": "6725d54f7eec7cb122038434c306d144",
            "Role": "Director",
            "Type": "Director",
            "PrimaryImageTag": "ef051c4545daa922f476ef1bd4ff6fbd"
        },
        {
            "Name": "Mika Rättö",
            "Id": "6725d54f7eec7cb122038434c306d144", 
            "Role": "Writer",
            "Type": "Writer",
            "PrimaryImageTag": "ef051c4545daa922f476ef1bd4ff6fbd"
        }
    ],
    "Studios": [
        {
            "Name": "Moderni Kanuuna",
            "Id": "61fd16fd8ddfcdaa0ac229fd2506ba15"
        }
    ],
    "GenreItems": [
        {
            "Name": "Comedy",
            "Id": "08d31605d366d63a7a924f944b4417f1"
        },
        {
            "Name": "Drama", 
            "Id": "090eac6e9de4fe1fbc194e5b96691277"
        }
    ],
    "ExternalUrls": [
        {
            "Name": "IMDb",
            "Url": "https://www.imdb.com/title/tt6043942"
        },
        {
            "Name": "TheMovieDb", 
            "Url": "https://www.themoviedb.org/movie/414547"
        }
    ],
    "ProviderIds": {
        "Tmdb": "414547",
        "Imdb": "tt6043942"
    }
}

RAW_JELLYFIN_REMOTE_CONTROL = {
    "Name": "Remote Control",
    "OriginalTitle": "Remote Control", 
    "ServerId": "a06ac75a6c2e40aab501522265dcb3c4",
    "Id": "a2225801be04a2a3f9d0ca17e7c5eb78",
    "Etag": "6c02f720300c164b3984a1cc4b06d075",
    "DateCreated": "2024-11-26T22:45:11.774859Z",
    "Type": "Movie",
    "Overview": "A video store clerk stumbles onto an alien plot to take over earth by brainwashing people with a bad '50s science fiction movie. He and his friends race to stop the aliens before the tapes can be distributed world-wide.",
    "Taglines": ["Your future is in their hands."],
    "Genres": ["Horror", "Science Fiction", "Comedy"],
    "Tags": [],
    "ProductionYear": 1988,
    "ProductionLocations": ["United States of America"],
    "CommunityRating": 5.5,
    "OfficialRating": "R",
    "People": [
        {
            "Name": "Kevin Dillon",
            "Id": "8b07bd1ac76bb2aae1ee62a3e484e741",
            "Role": "Cosmo", 
            "Type": "Actor",
            "PrimaryImageTag": "5bc05ff55c76265d32ec993514d2bcc7"
        },
        {
            "Name": "Jennifer Tilly",
            "Id": "c215aacce7ed8345a838266726fc66ac",
            "Role": "Allegra James",
            "Type": "Actor", 
            "PrimaryImageTag": "eaadeef63a794dd75f9f4cc8e1e589c7"
        },
        {
            "Name": "Jeff Lieberman",
            "Id": "a303eb8e3a8eb4138ae94241950957bc", 
            "Role": "Director",
            "Type": "Director",
            "PrimaryImageTag": "27434653e4798a3a4cba3a08281f54b3"
        },
        {
            "Name": "Jeff Lieberman",
            "Id": "a303eb8e3a8eb4138ae94241950957bc",
            "Role": "Writer", 
            "Type": "Writer",
            "PrimaryImageTag": "27434653e4798a3a4cba3a08281f54b3"
        }
    ],
    "Studios": [
        {
            "Name": "The Vista Organization",
            "Id": "a8d96699f7b0bb868032dd20a916e47c"
        }
    ],
    "GenreItems": [
        {
            "Name": "Horror",
            "Id": "8b9bd9a3eddad02f2b759b4938fdd0b8"
        },
        {
            "Name": "Science Fiction",
            "Id": "0430d514c35e1f158fa93f1eeb8361ea"
        },
        {
            "Name": "Comedy", 
            "Id": "08d31605d366d63a7a924f944b4417f1"
        }
    ],
    "ExternalUrls": [
        {
            "Name": "IMDb",
            "Url": "https://www.imdb.com/title/tt0093843"
        },
        {
            "Name": "TheMovieDb",
            "Url": "https://www.themoviedb.org/movie/60410"
        }
    ],
    "ProviderIds": {
        "Tmdb": "60410", 
        "Imdb": "tt0093843"
    }
}


def get_test_media_entities() -> List[MediaEntity]:
    """
    Create MediaEntity instances from test data using Field-Class architecture.
    
    Returns both simple and complex examples for testing:
    - Samurai Rauni: Complex tags, surreal content, Finnish production
    - Remote Control: Classic sci-fi horror comedy, no custom tags
    """
    samurai_rauni = MediaEntity.from_raw_data(RAW_JELLYFIN_SAMURAI_RAUNI, "Movie")
    remote_control = MediaEntity.from_raw_data(RAW_JELLYFIN_REMOTE_CONTROL, "Movie") 
    
    return [samurai_rauni, remote_control]


def get_intelligence_ready_samples() -> Dict[str, Dict[str, str]]:
    """
    Extract text samples suitable for NLP analysis and concept expansion.
    
    Used for testing Stage 3 concept expansion and Stage 4 content analysis.
    
    Returns:
        Dictionary mapping movie names to their analyzable text fields
    """
    entities = get_test_media_entities()
    
    samples = {}
    for entity in entities:
        samples[entity.entity_name] = entity.get_text_fields()
    
    return samples


def get_concept_expansion_test_cases() -> List[Dict[str, Any]]:
    """
    Generate test cases for concept expansion (Stage 3).
    
    Returns test cases with input terms extracted from actual movie data
    that should trigger intelligent concept expansion via PLUGINS.
    
    NOTE: Term extraction will be handled by Stage 3 plugins, not text_utils.
    """
    return [
        {
            "input_term": "samurai",
            "media_context": "movie", 
            "source_movie": "Samurai Rauni",
            "expected_concepts": ["warrior", "sword", "honor", "combat", "japanese"],
            "source_field": "tags",
            "note": "Will be processed by ConceptNet/LLM plugins in Stage 3"
        },
        {
            "input_term": "dark comedy",
            "media_context": "movie",
            "source_movie": "Samurai Rauni", 
            "expected_concepts": ["black humor", "satire", "irony", "absurd"],
            "source_field": "tags",
            "note": "Complex term requiring LLM understanding"
        },
        {
            "input_term": "surrealism",
            "media_context": "movie",
            "source_movie": "Samurai Rauni",
            "expected_concepts": ["bizarre", "dreamlike", "absurd", "unconventional"],
            "source_field": "tags",
            "note": "Art movement concept requiring sophisticated expansion"
        },
        {
            "input_term": "science fiction",
            "media_context": "movie",
            "source_movie": "Remote Control",
            "expected_concepts": ["sci-fi", "aliens", "technology", "future", "space"],
            "source_field": "genres",
            "note": "Genre with established concept relationships"
        },
        {
            "input_term": "horror",
            "media_context": "movie", 
            "source_movie": "Remote Control",
            "expected_concepts": ["scary", "fear", "thriller", "suspense", "terror"],
            "source_field": "genres",
            "note": "Emotional/psychological concept for expansion"
        }
    ]


def get_content_analysis_test_cases() -> List[Dict[str, Any]]:
    """
    Generate test cases for content analysis (Stage 4).
    
    Returns cases where the system should learn patterns from actual movie content.
    """
    return [
        {
            "movie_name": "Samurai Rauni",
            "analysis_type": "genre_content_mapping",
            "input_data": {
                "genres": ["Comedy", "Drama"],
                "overview": "Villagers are afraid of Samurai Rauni Reposaarelainen, who keeps them on their toes every day. When someone places a bounty on Rauni's head, he goes after this mysterious person.",
                "tags": ["samurai", "dark comedy", "surrealism", "chambara"]
            },
            "expected_insights": [
                "Comedy films can contain fear and tension themes", 
                "Drama can include action elements (bounty hunting)",
                "Tags provide more specific genre context than broad categories"
            ]
        },
        {
            "movie_name": "Remote Control",
            "analysis_type": "multi_genre_analysis",
            "input_data": {
                "genres": ["Horror", "Science Fiction", "Comedy"],
                "overview": "A video store clerk stumbles onto an alien plot to take over earth by brainwashing people with a bad '50s science fiction movie. He and his friends race to stop the aliens before the tapes can be distributed world-wide.",
                "taglines": ["Your future is in their hands."]
            },
            "expected_insights": [
                "Horror + Sci-Fi + Comedy = B-movie parody genre",
                "Video store setting indicates 1980s nostalgia themes",
                "Alien invasion plots are common sci-fi horror elements"
            ]
        }
    ]


def get_cache_key_test_cases() -> List[Dict[str, Any]]:
    """
    Generate test cases for cache key generation (Stage 2).
    
    Tests ASCII normalization and consistent key formatting.
    """
    return [
        {
            "input": {"cache_type": "conceptnet", "input_term": "action", "media_context": "movie"},
            "expected_key": "conceptnet:action:movie"
        },
        {
            "input": {"cache_type": "llm", "input_term": "psychological thriller", "media_context": "movie"},
            "expected_key": "llm:psychological thriller:movie"
        },
        {
            "input": {"cache_type": "conceptnet", "input_term": "Café", "media_context": "movie"}, 
            "expected_key": "conceptnet:cafe:movie"
        },
        {
            "input": {"cache_type": "gensim", "input_term": "Sci-Fi", "media_context": "movie"},
            "expected_key": "gensim:sci-fi:movie"
        }
    ]




def validate_media_entity_completeness(entity: MediaEntity) -> List[str]:
    """
    Validate that a MediaEntity contains expected fields for intelligence processing.
    
    Uses Field-Class architecture to dynamically check for required field types.
    """
    issues = []
    
    if not entity.entity_name:
        issues.append("Missing entity name")
    
    if not entity.entity_id:
        issues.append("Missing entity ID")
    
    
    text_fields = entity.get_text_fields()
    if not text_fields:
        issues.append("No text fields available for NLP analysis")
    
    
    field_summary = entity.get_field_summary()
    if field_summary['text_fields'] == 0:
        issues.append("No TEXT_CONTENT fields detected")
    
    if field_summary['concept_expandable'] == 0:
        issues.append("No concept-expandable fields detected")
    
    if len(field_summary['unknown_fields']) > 3:
        issues.append(f"Many unknown fields: {field_summary['unknown_fields']}")
    
    return issues


def simulate_plugin_processing(entity: MediaEntity) -> Dict[str, Any]:
    """
    Simulate how plugins would process a MediaEntity using Field-Class architecture.
    
    Shows the intelligence capabilities of the new field system.
    """
    text_fields = entity.get_text_fields()
    weighted_fields = entity.get_weighted_text_fields()
    expandable_fields = entity.get_concept_expandable_fields()
    field_summary = entity.get_field_summary()
    cache_context = entity.get_cache_context()
    
    return {
        "entity_id": entity.entity_id,
        "entity_name": entity.entity_name,
        "media_type": entity.media_type,
        "text_fields": text_fields,
        "weighted_fields": {k: f"{v[1]:.1f}" for k, v in weighted_fields.items()},
        "expandable_fields": list(expandable_fields.keys()),
        "cache_context": cache_context,
        "field_summary": field_summary,
        "total_text_length": sum(len(text) for text in text_fields.values()),
        "processing_notes": entity.processing_notes
    }


if __name__ == "__main__":

    print("Testing Field-Class MediaEntity architecture...")
    
    entities = get_test_media_entities()
    for entity in entities:
        print(f"\n🎬 Movie: {entity.entity_name}")
        

        issues = validate_media_entity_completeness(entity)
        if issues:
            print(f"  ⚠️  Issues: {issues}")
        else:
            print("  ✅ All required field types present")
        

        processing_info = simulate_plugin_processing(entity)
        print(f"  📝 Text fields: {list(processing_info['text_fields'].keys())}")
        print(f"  ⚖️  Weighted fields: {processing_info['weighted_fields']}")
        print(f"  🔍 Expandable fields: {processing_info['expandable_fields']}")
        print(f"  📊 Field summary: {processing_info['field_summary']}")
        print(f"  📏 Total text: {processing_info['total_text_length']} chars")
        
        if processing_info['processing_notes']:
            print(f"  📋 Notes: {processing_info['processing_notes']}")
    
    print("\n🧪 Testing concept expansion cases...")
    expansion_cases = get_concept_expansion_test_cases()
    for case in expansion_cases:
        print(f"  {case['input_term']} -> {case['expected_concepts'][:3]}...")
    
    print("\n🚀 Field-Class architecture ready for pipeline testing!")