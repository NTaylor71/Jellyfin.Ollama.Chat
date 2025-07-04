"""
Fuzzy matching system for movie search.
Handles typos, phonetic matching, and partial matches for movie titles and names.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
import string
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class FuzzyMatch:
    """Result of fuzzy matching operation."""
    text: str
    score: float
    match_type: str
    edit_distance: int
    matched_portion: str
    start_pos: int = 0
    end_pos: int = 0


class FuzzyMatcher:
    """Fuzzy matching system for movie search with multiple matching strategies."""
    
    def __init__(self, 
                 min_similarity: float = 0.6,
                 max_edit_distance: int = 3):
        """Initialize fuzzy matcher.
        
        Args:
            min_similarity: Minimum similarity threshold (0.0-1.0)
            max_edit_distance: Maximum edit distance for matching
        """
        self.min_similarity = min_similarity
        self.max_edit_distance = max_edit_distance
        
        # Soundex mapping for phonetic matching
        self.soundex_map = {
            'b': '1', 'f': '1', 'p': '1', 'v': '1',
            'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
            'd': '3', 't': '3',
            'l': '4',
            'm': '5', 'n': '5',
            'r': '6'
        }
        
        # Common movie title patterns
        self.movie_patterns = [
            r'\b(the|a|an)\s+',  # Articles
            r'\s*\(\d{4}\)',     # Year in parentheses
            r'\s*:\s*',          # Colons
            r'\s*-\s*',          # Dashes
            r'\s*&\s*',          # Ampersands
        ]
        
        # Common name variations
        self.name_variations = {
            'jr': 'junior',
            'sr': 'senior',
            'ii': 'second',
            'iii': 'third',
            'iv': 'fourth',
            'v': 'fifth'
        }
        
        # Phonetic equivalents for common movie terms
        self.phonetic_equivalents = {
            'ph': 'f',
            'gh': 'g',
            'kn': 'n',
            'wr': 'r',
            'mb': 'm'
        }
        
    def normalize_for_matching(self, text: str) -> str:
        """Normalize text for fuzzy matching.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove diacritics
        normalized = unicodedata.normalize('NFD', normalized)
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Apply phonetic simplifications
        for pattern, replacement in self.phonetic_equivalents.items():
            normalized = normalized.replace(pattern, replacement)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def clean_movie_title(self, title: str) -> str:
        """Clean movie title for matching.
        
        Args:
            title: Movie title to clean
            
        Returns:
            Cleaned title
        """
        if not title:
            return ""
        
        cleaned = self.normalize_for_matching(title)
        
        # Remove common movie title patterns
        for pattern in self.movie_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation except apostrophes
        cleaned = re.sub(r"[^\w\s']", ' ', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def clean_person_name(self, name: str) -> str:
        """Clean person name for matching.
        
        Args:
            name: Person name to clean
            
        Returns:
            Cleaned name
        """
        if not name:
            return ""
        
        cleaned = self.normalize_for_matching(name)
        
        # Handle name variations
        words = cleaned.split()
        normalized_words = []
        
        for word in words:
            # Remove punctuation
            word = re.sub(r'[^\w]', '', word)
            
            # Handle common abbreviations
            if word in self.name_variations:
                word = self.name_variations[word]
            
            if word:
                normalized_words.append(word)
        
        return ' '.join(normalized_words)
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance (Levenshtein distance) between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance
        """
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        
        # Create matrix
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],     # deletion
                        dp[i][j-1],     # insertion
                        dp[i-1][j-1]    # substitution
                    )
        
        return dp[m][n]
    
    def similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio (0.0-1.0)
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Use SequenceMatcher for ratio calculation
        matcher = SequenceMatcher(None, s1, s2)
        return matcher.ratio()
    
    def partial_ratio(self, s1: str, s2: str) -> float:
        """Calculate partial ratio for substring matching.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Partial similarity ratio (0.0-1.0)
        """
        if not s1 or not s2:
            return 0.0
        
        # Find best partial match
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        
        if len(shorter) == 0:
            return 0.0
        
        best_ratio = 0.0
        
        # Sliding window approach
        for i in range(len(longer) - len(shorter) + 1):
            substring = longer[i:i + len(shorter)]
            ratio = self.similarity_ratio(shorter, substring)
            if ratio > best_ratio:
                best_ratio = ratio
        
        return best_ratio
    
    def soundex(self, text: str) -> str:
        """Generate Soundex code for phonetic matching.
        
        Args:
            text: Input text
            
        Returns:
            Soundex code
        """
        if not text:
            return ""
        
        # Clean and normalize
        text = re.sub(r'[^a-zA-Z]', '', text.upper())
        
        if not text:
            return ""
        
        # First letter
        result = text[0]
        
        # Map remaining letters
        for char in text[1:]:
            if char.lower() in self.soundex_map:
                code = self.soundex_map[char.lower()]
                # Avoid duplicates
                if result[-1] != code:
                    result += code
        
        # Pad or truncate to 4 characters
        result = result[:4].ljust(4, '0')
        
        return result
    
    def phonetic_similarity(self, s1: str, s2: str) -> float:
        """Calculate phonetic similarity using Soundex.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Phonetic similarity score (0.0-1.0)
        """
        if not s1 or not s2:
            return 0.0
        
        # Split into words and compare
        words1 = s1.split()
        words2 = s2.split()
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Soundex for each word
        soundex1 = [self.soundex(word) for word in words1]
        soundex2 = [self.soundex(word) for word in words2]
        
        # Find best matching pairs
        matches = 0
        total = max(len(soundex1), len(soundex2))
        
        for s1_code in soundex1:
            for s2_code in soundex2:
                if s1_code == s2_code:
                    matches += 1
                    break
        
        return matches / total if total > 0 else 0.0
    
    def match_exact(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Find exact matches.
        
        Args:
            query: Search query
            candidates: List of candidate strings
            
        Returns:
            List of exact matches
        """
        matches = []
        query_normalized = self.normalize_for_matching(query)
        
        for candidate in candidates:
            candidate_normalized = self.normalize_for_matching(candidate)
            
            if query_normalized == candidate_normalized:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=1.0,
                    match_type="exact",
                    edit_distance=0,
                    matched_portion=candidate
                ))
        
        return matches
    
    def match_fuzzy(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Find fuzzy matches using edit distance and similarity.
        
        Args:
            query: Search query
            candidates: List of candidate strings
            
        Returns:
            List of fuzzy matches
        """
        matches = []
        query_normalized = self.normalize_for_matching(query)
        
        for candidate in candidates:
            candidate_normalized = self.normalize_for_matching(candidate)
            
            # Calculate edit distance
            distance = self.edit_distance(query_normalized, candidate_normalized)
            
            # Calculate similarity ratio
            similarity = self.similarity_ratio(query_normalized, candidate_normalized)
            
            # Check if match meets criteria
            if (distance <= self.max_edit_distance and 
                similarity >= self.min_similarity):
                
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=similarity,
                    match_type="fuzzy",
                    edit_distance=distance,
                    matched_portion=candidate
                ))
        
        return matches
    
    def match_partial(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Find partial matches (substrings).
        
        Args:
            query: Search query
            candidates: List of candidate strings
            
        Returns:
            List of partial matches
        """
        matches = []
        query_normalized = self.normalize_for_matching(query)
        
        for candidate in candidates:
            candidate_normalized = self.normalize_for_matching(candidate)
            
            # Check if query is substring of candidate
            if query_normalized in candidate_normalized:
                start_pos = candidate_normalized.find(query_normalized)
                end_pos = start_pos + len(query_normalized)
                
                # Calculate partial ratio
                partial_score = self.partial_ratio(query_normalized, candidate_normalized)
                
                if partial_score >= self.min_similarity:
                    matches.append(FuzzyMatch(
                        text=candidate,
                        score=partial_score,
                        match_type="partial",
                        edit_distance=0,
                        matched_portion=candidate[start_pos:end_pos],
                        start_pos=start_pos,
                        end_pos=end_pos
                    ))
            
            # Check if candidate is substring of query
            elif candidate_normalized in query_normalized:
                partial_score = self.partial_ratio(query_normalized, candidate_normalized)
                
                if partial_score >= self.min_similarity:
                    matches.append(FuzzyMatch(
                        text=candidate,
                        score=partial_score,
                        match_type="partial",
                        edit_distance=0,
                        matched_portion=candidate,
                        start_pos=0,
                        end_pos=len(candidate)
                    ))
        
        return matches
    
    def match_phonetic(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Find phonetic matches using Soundex.
        
        Args:
            query: Search query
            candidates: List of candidate strings
            
        Returns:
            List of phonetic matches
        """
        matches = []
        
        for candidate in candidates:
            phonetic_score = self.phonetic_similarity(query, candidate)
            
            if phonetic_score >= self.min_similarity:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=phonetic_score,
                    match_type="phonetic",
                    edit_distance=0,
                    matched_portion=candidate
                ))
        
        return matches
    
    def match_movie_title(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Match movie titles with specialized cleaning.
        
        Args:
            query: Search query
            candidates: List of movie titles
            
        Returns:
            List of matches
        """
        matches = []
        
        # Clean query
        query_cleaned = self.clean_movie_title(query)
        
        for candidate in candidates:
            candidate_cleaned = self.clean_movie_title(candidate)
            
            # Try different matching strategies
            
            # 1. Exact match on cleaned titles
            if query_cleaned == candidate_cleaned:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=1.0,
                    match_type="exact_cleaned",
                    edit_distance=0,
                    matched_portion=candidate
                ))
                continue
            
            # 2. Fuzzy match on cleaned titles
            similarity = self.similarity_ratio(query_cleaned, candidate_cleaned)
            distance = self.edit_distance(query_cleaned, candidate_cleaned)
            
            if (similarity >= self.min_similarity and 
                distance <= self.max_edit_distance):
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=similarity,
                    match_type="fuzzy_cleaned",
                    edit_distance=distance,
                    matched_portion=candidate
                ))
                continue
            
            # 3. Partial match on cleaned titles
            partial_score = self.partial_ratio(query_cleaned, candidate_cleaned)
            
            if partial_score >= self.min_similarity:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=partial_score,
                    match_type="partial_cleaned",
                    edit_distance=0,
                    matched_portion=candidate
                ))
        
        return matches
    
    def match_person_name(self, query: str, candidates: List[str]) -> List[FuzzyMatch]:
        """Match person names with specialized cleaning.
        
        Args:
            query: Search query
            candidates: List of person names
            
        Returns:
            List of matches
        """
        matches = []
        
        # Clean query
        query_cleaned = self.clean_person_name(query)
        
        for candidate in candidates:
            candidate_cleaned = self.clean_person_name(candidate)
            
            # Try different matching strategies
            
            # 1. Exact match on cleaned names
            if query_cleaned == candidate_cleaned:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=1.0,
                    match_type="exact_name",
                    edit_distance=0,
                    matched_portion=candidate
                ))
                continue
            
            # 2. Check for name part matches (first/last name)
            query_parts = query_cleaned.split()
            candidate_parts = candidate_cleaned.split()
            
            if query_parts and candidate_parts:
                # Check if any query part matches any candidate part
                for query_part in query_parts:
                    for candidate_part in candidate_parts:
                        if query_part == candidate_part:
                            matches.append(FuzzyMatch(
                                text=candidate,
                                score=0.8,  # Lower score for partial name match
                                match_type="name_part",
                                edit_distance=0,
                                matched_portion=candidate_part
                            ))
                            break
                    else:
                        continue
                    break
            
            # 3. Fuzzy match on cleaned names
            similarity = self.similarity_ratio(query_cleaned, candidate_cleaned)
            distance = self.edit_distance(query_cleaned, candidate_cleaned)
            
            if (similarity >= self.min_similarity and 
                distance <= self.max_edit_distance):
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=similarity,
                    match_type="fuzzy_name",
                    edit_distance=distance,
                    matched_portion=candidate
                ))
                continue
            
            # 4. Phonetic match
            phonetic_score = self.phonetic_similarity(query_cleaned, candidate_cleaned)
            
            if phonetic_score >= self.min_similarity:
                matches.append(FuzzyMatch(
                    text=candidate,
                    score=phonetic_score,
                    match_type="phonetic_name",
                    edit_distance=0,
                    matched_portion=candidate
                ))
        
        return matches
    
    def find_best_matches(self, 
                         query: str, 
                         candidates: List[str],
                         match_type: str = "general",
                         top_k: int = 10) -> List[FuzzyMatch]:
        """Find best matches using all strategies.
        
        Args:
            query: Search query
            candidates: List of candidate strings
            match_type: Type of matching ("general", "movie_title", "person_name")
            top_k: Number of top matches to return
            
        Returns:
            List of best matches
        """
        if not query or not candidates:
            return []
        
        all_matches = []
        
        # Choose matching strategy
        if match_type == "movie_title":
            matches = self.match_movie_title(query, candidates)
        elif match_type == "person_name":
            matches = self.match_person_name(query, candidates)
        else:
            # General matching - try all strategies
            matches = []
            matches.extend(self.match_exact(query, candidates))
            matches.extend(self.match_fuzzy(query, candidates))
            matches.extend(self.match_partial(query, candidates))
            matches.extend(self.match_phonetic(query, candidates))
        
        all_matches.extend(matches)
        
        # Remove duplicates and sort by score
        seen = set()
        unique_matches = []
        
        for match in all_matches:
            if match.text not in seen:
                seen.add(match.text)
                unique_matches.append(match)
        
        # Sort by score (descending)
        unique_matches.sort(key=lambda x: x.score, reverse=True)
        
        return unique_matches[:top_k]
    
    def match_score_threshold(self, score: float) -> str:
        """Get match quality description based on score.
        
        Args:
            score: Match score (0.0-1.0)
            
        Returns:
            Quality description
        """
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "very_good"
        elif score >= 0.75:
            return "good"
        elif score >= 0.65:
            return "fair"
        else:
            return "poor"