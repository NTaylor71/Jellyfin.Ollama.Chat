"""
Text processing pipeline for movie search.
Handles tokenization, normalization, and NLP preprocessing for movie data.
"""

import re
import string
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""
    original: str
    normalized: str
    tokens: List[str]
    lemmatized: List[str]
    keywords: List[str]
    language: str = "en"
    
    
class TextProcessor:
    """Text processing pipeline for movie search."""
    
    def __init__(self, language: str = "en"):
        """Initialize text processor.
        
        Args:
            language: Language code for processing (default: "en")
        """
        self.language = language
        self.lemmatizer = None
        self.stop_words = set()
        self._initialized = False
        
        # Movie-specific terms that shouldn't be removed
        self.movie_terms = {
            "2d", "3d", "4k", "hd", "imax", "dvd", "blu-ray", "bluray",
            "director", "actor", "actress", "cast", "crew", "studio",
            "sequel", "prequel", "trilogy", "series", "episode", "season",
            "drama", "comedy", "action", "thriller", "horror", "sci-fi",
            "documentary", "animation", "musical", "western", "crime"
        }
        
        # Character mappings for normalization
        self.char_mappings = {
            # Common diacritics
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'å': 'a', 'ã': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
            'ñ': 'n', 'ç': 'c',
            'ß': 'ss',
            # Punctuation normalization
            ''': "'", ''': "'", '"': '"', '"': '"',
            '–': '-', '—': '-', '…': '...'
        }
        
        self._ensure_nltk_data()
        
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is available."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available - using basic text processing")
            return
            
        required_datasets = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'
        ]
        
        missing_datasets = []
        for dataset in required_datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{dataset}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{dataset}')
                    except LookupError:
                        missing_datasets.append(dataset)
        
        if missing_datasets:
            logger.info(f"Downloading missing NLTK datasets: {missing_datasets}")
            for dataset in missing_datasets:
                try:
                    nltk.download(dataset, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download {dataset}: {e}")
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components."""
        if not NLTK_AVAILABLE or self._initialized:
            return
            
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words(self.language))
            
            # Remove movie-specific terms from stop words
            self.stop_words = self.stop_words - self.movie_terms
            
            self._initialized = True
            logger.debug("NLTK components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
            self._initialized = False
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Apply character mappings
        normalized = text
        for char, replacement in self.char_mappings.items():
            normalized = normalized.replace(char, replacement)
        
        # Unicode normalization (NFD -> NFC)
        normalized = unicodedata.normalize('NFC', normalized)
        
        # Basic cleanup
        normalized = re.sub(r'\s+', ' ', normalized)  # Multiple spaces to single
        normalized = normalized.strip()
        
        return normalized
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        if NLTK_AVAILABLE:
            self._initialize_nltk()
            try:
                tokens = word_tokenize(text.lower())
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}")
                tokens = self._basic_tokenize(text)
        else:
            tokens = self._basic_tokenize(text)
        
        # Filter out punctuation and single characters
        tokens = [
            token for token in tokens 
            if token not in string.punctuation and len(token) > 1
        ]
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization fallback."""
        # Simple word splitting with punctuation handling
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        tokens = text.split()
        return [token for token in tokens if token and len(token) > 1]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their root forms.
        
        Args:
            tokens: List of tokens to lemmatize
            
        Returns:
            List of lemmatized tokens
        """
        if not tokens:
            return []
            
        if not NLTK_AVAILABLE or not self._initialized:
            self._initialize_nltk()
            
        if not self.lemmatizer:
            return tokens  # Return original tokens if lemmatizer not available
            
        try:
            # Get POS tags for better lemmatization
            pos_tags = pos_tag(tokens)
            lemmatized = []
            
            for token, pos in pos_tags:
                # Convert POS tag to WordNet format
                wordnet_pos = self._get_wordnet_pos(pos)
                if wordnet_pos:
                    lemma = self.lemmatizer.lemmatize(token, wordnet_pos)
                else:
                    lemma = self.lemmatizer.lemmatize(token)
                lemmatized.append(lemma)
            
            return lemmatized
            
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return tokens
    
    def _get_wordnet_pos(self, pos_tag: str) -> Optional[str]:
        """Convert POS tag to WordNet POS."""
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        return None
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered tokens without stop words
        """
        if not tokens:
            return []
            
        if not NLTK_AVAILABLE or not self._initialized:
            self._initialize_nltk()
            
        if not self.stop_words:
            # Basic English stop words if NLTK not available
            basic_stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he',
                'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
            }
            return [token for token in tokens if token.lower() not in basic_stop_words]
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def extract_keywords(self, tokens: List[str], min_length: int = 2) -> List[str]:
        """Extract meaningful keywords from tokens.
        
        Args:
            tokens: List of tokens
            min_length: Minimum length for keywords
            
        Returns:
            List of keywords
        """
        if not tokens:
            return []
            
        keywords = []
        
        for token in tokens:
            # Skip very short tokens
            if len(token) < min_length:
                continue
                
            # Skip if it's all digits (years are handled separately)
            if token.isdigit():
                continue
                
            # Skip if it's mostly punctuation
            if sum(1 for c in token if c in string.punctuation) > len(token) / 2:
                continue
            
            keywords.append(token)
        
        return keywords
    
    def process_text(self, text: str) -> ProcessedText:
        """Process text through the complete pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            ProcessedText object with all processing results
        """
        if not text:
            return ProcessedText(
                original="",
                normalized="",
                tokens=[],
                lemmatized=[],
                keywords=[]
            )
        
        # Step 1: Normalize text
        normalized = self.normalize_text(text)
        
        # Step 2: Tokenize
        tokens = self.tokenize(normalized)
        
        # Step 3: Remove stop words
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Step 4: Lemmatize
        lemmatized = self.lemmatize(filtered_tokens)
        
        # Step 5: Extract keywords
        keywords = self.extract_keywords(lemmatized)
        
        return ProcessedText(
            original=text,
            normalized=normalized,
            tokens=tokens,
            lemmatized=lemmatized,
            keywords=keywords
        )
    
    def process_movie_field(self, field_name: str, value: Any) -> List[str]:
        """Process a specific movie field for search.
        
        Args:
            field_name: Name of the movie field
            value: Field value to process
            
        Returns:
            List of processed search terms
        """
        if not value:
            return []
        
        # Handle different field types
        if field_name in ["name", "original_title", "overview"]:
            # Text fields - full processing
            processed = self.process_text(str(value))
            return processed.keywords
            
        elif field_name == "people":
            # Extract person names
            terms = []
            if isinstance(value, list):
                for person in value:
                    if isinstance(person, dict):
                        name = person.get("name", "")
                        if name:
                            processed = self.process_text(name)
                            terms.extend(processed.keywords)
                    elif isinstance(person, str):
                        processed = self.process_text(person)
                        terms.extend(processed.keywords)
            return terms
            
        elif field_name in ["genres", "tags", "taglines", "production_locations"]:
            # List fields - process each item
            terms = []
            if isinstance(value, list):
                for item in value:
                    processed = self.process_text(str(item))
                    terms.extend(processed.keywords)
            return terms
            
        elif field_name == "studios":
            # Studio objects
            terms = []
            if isinstance(value, list):
                for studio in value:
                    if isinstance(studio, dict):
                        name = studio.get("name", "")
                        if name:
                            processed = self.process_text(name)
                            terms.extend(processed.keywords)
            return terms
            
        elif field_name == "enhanced_fields":
            # LLM-generated fields
            terms = []
            if isinstance(value, dict):
                for field_value in value.values():
                    if isinstance(field_value, str):
                        processed = self.process_text(field_value)
                        terms.extend(processed.keywords)
            return terms
            
        else:
            # Default string processing
            processed = self.process_text(str(value))
            return processed.keywords
    
    def process_movie_document(self, movie_doc: Dict[str, Any]) -> Dict[str, List[str]]:
        """Process an entire movie document for search indexing.
        
        Args:
            movie_doc: Movie document dictionary
            
        Returns:
            Dictionary mapping field names to processed search terms
        """
        search_terms = {}
        
        # Define important fields for search
        search_fields = [
            "name", "original_title", "overview", "taglines",
            "people", "genres", "tags", "production_locations",
            "studios", "enhanced_fields"
        ]
        
        for field in search_fields:
            if field in movie_doc:
                terms = self.process_movie_field(field, movie_doc[field])
                if terms:
                    search_terms[field] = terms
        
        return search_terms
    
    def detect_language(self, text: str) -> str:
        """Detect text language (basic implementation).
        
        Args:
            text: Input text
            
        Returns:
            Language code (defaults to 'en')
        """
        # Basic language detection - can be enhanced later
        if not text:
            return "en"
            
        # Simple heuristics for common languages
        text_lower = text.lower()
        
        # Common non-English patterns
        if any(char in text for char in ['ñ', 'ç', 'ü', 'ö', 'ä']):
            return "es"  # Spanish or German - would need better detection
            
        # Default to English
        return "en"