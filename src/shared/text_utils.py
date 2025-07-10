"""
Text processing utilities for the procedural intelligence pipeline.
Shared functions for consistent text normalization and processing.
"""

import unicodedata
import re
from typing import Optional, List, Dict, Any


def to_ascii(text: str) -> str:
    """
    Convert text to ASCII using Unicode normalization.
    
    As specified in Stage 3.3: unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")
    
    This is the standard normalization used throughout the system for:
    - Cache key generation (Stage 2)
    - Concept expansion processing (Stage 3)
    - Content analysis (Stage 4)
    - Query processing (Stage 6)
    
    Args:
        text: Input text to normalize
        
    Returns:
        ASCII-normalized text with non-ASCII characters removed
        
    Examples:
        >>> to_ascii("Café")
        'Cafe'
        >>> to_ascii("Naïve résumé")
        'Naive resume'
        >>> to_ascii("中文")
        ''
    """
    if not isinstance(text, str):
        text = str(text)
    

    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    
    return ascii_text


def clean_for_cache_key(text: str) -> str:
    """
    Clean text for use in cache keys.
    
    Applies ASCII normalization plus additional cleaning for consistent cache keys.
    Used by CacheKey.generate_key() and throughout the caching system.
    
    Preserves underscores in field names (like "release_date") but cleans other text.
    
    Args:
        text: Input text to clean
        
    Returns:
        Clean text suitable for cache keys (lowercase, alphanumeric + underscores)
        
    Examples:
        >>> clean_for_cache_key("Action Movie!")
        'action_movie'
        >>> clean_for_cache_key("Sci-Fi & Fantasy")
        'sci_fi_fantasy'
        >>> clean_for_cache_key("release_date")
        'release_date'
    """

    ascii_text = to_ascii(text.lower())
    
    
    cleaned = re.sub(r'[^a-z0-9\s\-_]', '', ascii_text)
    

    cleaned = re.sub(r'[\s\-]+', '_', cleaned)
    

    cleaned = re.sub(r'_{2,}', '_', cleaned).strip('_')
    
    return cleaned


def normalize_text_for_analysis(text: str) -> str:
    """
    Normalize text for NLP analysis while preserving readability.
    
    Less aggressive than cache key cleaning - keeps punctuation and structure
    that might be important for concept understanding.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text suitable for NLP processing
        
    Examples:
        >>> normalize_text_for_analysis("A naïve approach to AI.")
        'A naive approach to AI.'
        >>> normalize_text_for_analysis("Café & Restaurant")
        'Cafe & Restaurant'
    """

    ascii_text = to_ascii(text)
    

    cleaned = re.sub(r'\s+', ' ', ascii_text).strip()
    
    return cleaned


def extract_key_concepts(text: str, max_concepts: int = 10) -> List[str]:
    """
    Extract key concepts from text using NLTK stopwords (proper NLP approach).
    
    Uses the NLTK stopwords corpus to filter out common words instead of
    hard-coded lists. This is the proper, non-brittle way to do it.
    
    Args:
        text: Input text to extract concepts from
        max_concepts: Maximum number of concepts to return
        
    Returns:
        List of extracted concepts (normalized strings)
    """
    if not text or not isinstance(text, str):
        return []
    
    try:
        import nltk
        from nltk.corpus import stopwords
        
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available - falling back to simple word splitting")

            words = re.split(r'[,\.\!\?\;\:\s\-\(\)]+', normalize_text_for_analysis(text).lower())
            return [w for w in words if len(w) >= 3 and not w.isdigit()][:max_concepts]
            
    except ImportError:

        words = re.split(r'[,\.\!\?\;\:\s\-\(\)]+', normalize_text_for_analysis(text).lower())
        return [w for w in words if len(w) >= 3 and not w.isdigit()][:max_concepts]
    

    normalized = normalize_text_for_analysis(text)
    

    words = re.split(r'[,\.\!\?\;\:\s\-\(\)]+', normalized.lower())
    

    concepts = []
    for word in words:
        if (len(word) >= 3 and 
            word not in stop_words and 
            not word.isdigit()):
            concepts.append(word)
    
    
    unique_concepts = []
    seen = set()
    for concept in concepts:
        if concept not in seen:
            unique_concepts.append(concept)
            seen.add(concept)
    
    return unique_concepts[:max_concepts]






def safe_string_conversion(value: Any) -> str:
    """
    Safely convert any value to a string with ASCII normalization.
    
    Used throughout the system when processing unknown field types from media data.
    Handles None values, lists, dictionaries, and other complex types.
    
    Args:
        value: Any value to convert to string
        
    Returns:
        ASCII-normalized string representation
        
    Examples:
        >>> safe_string_conversion(None)
        ''
        >>> safe_string_conversion(['action', 'comedy'])
        'action comedy'
        >>> safe_string_conversion({'name': 'John', 'role': 'Actor'})
        'John Actor'
    """
    if value is None:
        return ''
    
    if isinstance(value, str):
        return to_ascii(value)
    
    if isinstance(value, (list, tuple)):

        str_items = [safe_string_conversion(item) for item in value if item is not None]
        return ' '.join(str_items)
    
    if isinstance(value, dict):

        str_values = [safe_string_conversion(v) for v in value.values() if v is not None]
        return ' '.join(str_values)
    
     and normalize
    return to_ascii(str(value))


def validate_ascii_text(text: str) -> bool:
    """
    Validate that text contains only ASCII characters.
    
    Used for testing and validation throughout the system.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text is pure ASCII, False otherwise
    """
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False



_CACHE_KEY_PATTERN = re.compile(r'[^a-z0-9\s\-]')
_WHITESPACE_PATTERN = re.compile(r'\s+')
_UNDERSCORE_PATTERN = re.compile(r'_+')


def fast_cache_key_clean(text: str) -> str:
    """
    Optimized version of clean_for_cache_key using pre-compiled patterns.
    
    Used in high-frequency operations like cache key generation.
    """
    ascii_text = to_ascii(text.lower())
    cleaned = _CACHE_KEY_PATTERN.sub('', ascii_text)
    cleaned = _WHITESPACE_PATTERN.sub('_', cleaned)
    cleaned = _UNDERSCORE_PATTERN.sub('_', cleaned).strip('_')
    return cleaned


if __name__ == "__main__":

    test_cases = [
        "Café & Restaurant",
        "Naïve résumé", 
        "Sci-Fi Action",
        "dark comedy, surrealism",
        "中文测试",
        "Action Movie!!!",
        None,
        ['action', 'comedy'],
        {'name': 'John', 'role': 'Actor'}
    ]
    
    print("Testing text utilities...")
    
    for test_input in test_cases:
        print(f"\nInput: {test_input}")
        
        if isinstance(test_input, str):
            print(f"  to_ascii: '{to_ascii(test_input)}'")
            print(f"  cache_key: '{clean_for_cache_key(test_input)}'")
            print(f"  normalized: '{normalize_text_for_analysis(test_input)}'")
        
        print(f"  safe_string: '{safe_string_conversion(test_input)}'")
    
    print("\n✅ Text utilities test complete!")