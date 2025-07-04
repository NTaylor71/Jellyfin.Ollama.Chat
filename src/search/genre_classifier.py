"""
Genre Classifier - Multi-label genre classification with hierarchy and subgenre detection.

This module handles intelligent genre classification, understanding relationships between genres,
subgenres, and providing confidence scoring for genre assignments.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class GenreConfidence(Enum):
    """Confidence levels for genre classification."""
    HIGH = "high"       # 0.8+
    MEDIUM = "medium"   # 0.5-0.79
    LOW = "low"         # 0.3-0.49
    VERY_LOW = "very_low"  # <0.3


@dataclass
class GenreMatch:
    """A genre match with confidence and reasoning."""
    genre: str
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    subgenres: List[str] = field(default_factory=list)
    parent_genres: List[str] = field(default_factory=list)


@dataclass
class GenreClassification:
    """Complete genre classification result."""
    primary_genres: List[GenreMatch]
    secondary_genres: List[GenreMatch]
    subgenres: List[GenreMatch]
    genre_score: float
    consistency_score: float


class GenreClassifier:
    """Intelligent multi-label genre classifier with hierarchy support."""
    
    def __init__(self):
        self._init_genre_hierarchy()
        self._init_genre_patterns()
        self._init_subgenre_mappings()
        self._init_keyword_associations()
    
    def _init_genre_hierarchy(self):
        """Initialize genre hierarchy and relationships."""
        # Primary genre categories
        self.primary_genres = {
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror', 'Music',
            'Mystery', 'Romance', 'Science Fiction', 'Thriller', 'War', 'Western'
        }
        
        # Genre hierarchy - parent -> children
        self.genre_hierarchy = {
            'Action': ['Action Thriller', 'Martial Arts', 'Superhero', 'Spy'],
            'Adventure': ['Epic Adventure', 'Survival', 'Treasure Hunt'],
            'Comedy': ['Romantic Comedy', 'Dark Comedy', 'Slapstick', 'Parody', 'Satire'],
            'Crime': ['Film Noir', 'Heist', 'Police Procedural', 'Gangster'],
            'Drama': ['Period Drama', 'Legal Drama', 'Medical Drama', 'Family Drama'],
            'Fantasy': ['Epic Fantasy', 'Urban Fantasy', 'Dark Fantasy', 'Fairy Tale'],
            'Horror': ['Slasher', 'Psychological Horror', 'Supernatural Horror', 'Body Horror', 'Zombie'],
            'Mystery': ['Detective', 'Whodunit', 'Cozy Mystery', 'Police Procedural'],
            'Romance': ['Romantic Comedy', 'Romantic Drama', 'Period Romance'],
            'Science Fiction': ['Cyberpunk', 'Space Opera', 'Dystopian', 'Time Travel', 'Alien'],
            'Thriller': ['Psychological Thriller', 'Action Thriller', 'Spy Thriller', 'Political Thriller'],
            'War': ['World War II', 'Vietnam War', 'Modern Warfare', 'Historical War']
        }
        
        # Reverse mapping: child -> parent
        self.parent_genres = {}
        for parent, children in self.genre_hierarchy.items():
            for child in children:
                self.parent_genres[child] = parent
        
        # Genre combinations that work well together
        self.compatible_genres = {
            'Action': ['Adventure', 'Thriller', 'Crime', 'Science Fiction'],
            'Comedy': ['Romance', 'Family', 'Adventure'],
            'Drama': ['Romance', 'Biography', 'Crime', 'War'],
            'Horror': ['Thriller', 'Mystery', 'Supernatural'],
            'Science Fiction': ['Action', 'Adventure', 'Thriller'],
            'Thriller': ['Action', 'Crime', 'Mystery', 'Psychological']
        }
        
        # Mutually exclusive genres
        self.exclusive_genres = {
            ('Comedy', 'Horror'): 0.3,  # Some overlap (dark comedy/horror comedy)
            ('Family', 'Horror'): 0.1,  # Very rare overlap
            ('Documentary', 'Fantasy'): 0.0,  # No overlap
            ('Animation', 'Documentary'): 0.8  # Some overlap (animated docs)
        }
    
    def _init_genre_patterns(self):
        """Initialize regex patterns for genre detection."""
        self.genre_patterns = {
            # Primary genres
            'Action': [
                r'\b(action|fighting|combat|martial arts|kung fu|karate|boxing)\b',
                r'\b(chase|explosion|gunfight|battle|warfare)\b',
                r'\b(adrenaline|intense|fast-paced|high-octane)\b'
            ],
            'Adventure': [
                r'\b(adventure|quest|journey|expedition|exploration)\b',
                r'\b(treasure|discovery|survival|wilderness)\b',
                r'\b(epic|grand|sweeping|heroic)\b'
            ],
            'Animation': [
                r'\b(animated|animation|cartoon|anime|cgi|pixar|disney)\b',
                r'\b(stop-motion|claymation|computer-animated)\b'
            ],
            'Biography': [
                r'\b(biography|biopic|bio|life story|true story)\b',
                r'\b(based on|real life|historical figure)\b'
            ],
            'Comedy': [
                r'\b(comedy|funny|hilarious|humor|comedic|laugh)\b',
                r'\b(satire|parody|slapstick|witty|amusing)\b',
                r'\b(rom-com|romantic comedy|dark comedy)\b'
            ],
            'Crime': [
                r'\b(crime|criminal|heist|robbery|theft|murder)\b',
                r'\b(detective|police|investigation|noir|gangster)\b',
                r'\b(mafia|organized crime|underworld)\b'
            ],
            'Documentary': [
                r'\b(documentary|doc|real|factual|non-fiction)\b',
                r'\b(behind the scenes|making of|investigation)\b'
            ],
            'Drama': [
                r'\b(drama|dramatic|emotional|serious|intense)\b',
                r'\b(character study|relationship|family|personal)\b'
            ],
            'Family': [
                r'\b(family|kids|children|all ages|wholesome)\b',
                r'\b(disney|pixar|dreamworks|pg rated)\b'
            ],
            'Fantasy': [
                r'\b(fantasy|magical|magic|wizard|dragon|fairy)\b',
                r'\b(mythical|supernatural|enchanted|mystical)\b',
                r'\b(lord of the rings|harry potter|game of thrones)\b'
            ],
            'Horror': [
                r'\b(horror|scary|frightening|terrifying|nightmare)\b',
                r'\b(zombie|vampire|ghost|demon|monster)\b',
                r'\b(slasher|gore|blood|haunted|possessed)\b'
            ],
            'Music': [
                r'\b(musical|music|song|singing|dance|broadway)\b',
                r'\b(concert|performance|musician|band)\b'
            ],
            'Mystery': [
                r'\b(myster(?:y|ies)|detective|investigation|clue|solve|solving)\b',
                r'\b(whodunit|puzzle|crime solving|murder mystery|a mystery)\b'
            ],
            'Romance': [
                r'\b(romance|romantic|love|relationship|dating)\b',
                r'\b(wedding|marriage|valentine|passionate)\b'
            ],
            'Science Fiction': [
                r'\b(sci-?fi|science fiction|futuristic|cyberpunk)\b',
                r'\b(space|alien|robot|time travel|dystopian)\b',
                r'\b(star wars|star trek|blade runner|matrix)\b'
            ],
            'Thriller': [
                r'\b(thriller|suspense|tension|edge of seat)\b',
                r'\b(psychological|mind-bending|twist|paranoia)\b'
            ],
            'War': [
                r'\b(war|military|battle|combat|soldier|army)\b',
                r'\b(world war|vietnam|iraq|afghanistan|conflict)\b'
            ],
            'Western': [
                r'\b(western|cowboy|frontier|wild west|gunslinger)\b',
                r'\b(saloon|sheriff|outlaw|ranch|desert)\b'
            ]
        }
    
    def _init_subgenre_mappings(self):
        """Initialize subgenre detection patterns."""
        self.subgenre_patterns = {
            # Action subgenres
            'Martial Arts': [
                r'\b(martial arts|kung fu|karate|judo|taekwondo|boxing|mma)\b',
                r'\b(fighter|tournament|sensei|dojo)\b'
            ],
            'Superhero': [
                r'\b(superhero|super hero|comic book|marvel|dc comics)\b',
                r'\b(powers|abilities|cape|mask|villain)\b'
            ],
            'Spy': [
                r'\b(spy|espionage|agent|secret service|cia|mi6)\b',
                r'\b(undercover|mission|intelligence|bond)\b'
            ],
            
            # Comedy subgenres
            'Romantic Comedy': [
                r'\b(rom-?com|romantic comedy|love comedy)\b',
                r'\b(meet cute|wedding|dating|relationship comedy)\b'
            ],
            'Dark Comedy': [
                r'\b(dark comedy|black comedy|black humor)\b',
                r'\b(satirical|cynical|mordant|gallows humor)\b'
            ],
            'Parody': [
                r'\b(parody|spoof|satire|mockumentary|send-up)\b',
                r'\b(comedy tribute|comedy homage)\b'
            ],
            
            # Horror subgenres
            'Slasher': [
                r'\b(slasher|serial killer|masked killer|final girl)\b',
                r'\b(halloween|friday the 13th|scream|nightmare)\b'
            ],
            'Psychological Horror': [
                r'\b(psychological horror|mind horror|mental)\b',
                r'\b(paranoia|madness|reality|perception)\b'
            ],
            'Supernatural Horror': [
                r'\b(supernatural|ghost|demon|spirit|haunted)\b',
                r'\b(possession|exorcism|poltergeist|paranormal)\b'
            ],
            'Zombie': [
                r'\b(zombie|undead|living dead|walker|infected)\b',
                r'\b(apocalypse|outbreak|virus|plague)\b'
            ],
            
            # Sci-Fi subgenres
            'Cyberpunk': [
                r'\b(cyberpunk|cyber|digital|virtual reality|matrix)\b',
                r'\b(hacker|cyberspace|neural|augmented)\b'
            ],
            'Space Opera': [
                r'\b(space opera|galactic|empire|federation)\b',
                r'\b(star wars|star trek|epic space|interstellar)\b'
            ],
            'Dystopian': [
                r'\b(dystopian|dystopia|totalitarian|oppressive)\b',
                r'\b(dystopian future|dystopian society)\b',
                r'\b(orwell|brave new world|hunger games)\b'
            ],
            'Time Travel': [
                r'\b(time travel|time machine|temporal|paradox)\b',
                r'\b(past|future|timeline|loop)\b'
            ],
            
            # Thriller subgenres
            'Psychological Thriller': [
                r'\b(psychological thriller|psychological thrillers)\b',
                r'\bpsychological\b.*\bthriller',
                r'\b(mind game|mental|paranoia|obsession|identity|reality)\b'
            ],
            'Political Thriller': [
                r'\b(political|conspiracy|government|election)\b',
                r'\b(corruption|cover-up|scandal|espionage)\b'
            ]
        }
    
    def _init_keyword_associations(self):
        """Initialize keyword to genre associations with weights."""
        self.keyword_weights = {
            # High confidence keywords
            'superhero': {'Action': 0.9, 'Adventure': 0.3, 'Fantasy': 0.4},
            'romantic': {'Romance': 0.9, 'Comedy': 0.3, 'Drama': 0.4},
            'zombie': {'Horror': 0.9, 'Action': 0.3, 'Thriller': 0.4},
            'detective': {'Mystery': 0.9, 'Crime': 0.7, 'Thriller': 0.5},
            'space': {'Science Fiction': 0.8, 'Adventure': 0.3},
            'cowboy': {'Western': 0.9, 'Adventure': 0.2},
            
            # Medium confidence keywords
            'fight': {'Action': 0.6, 'Crime': 0.3, 'Drama': 0.2},
            'funny': {'Comedy': 0.7, 'Family': 0.3},
            'scary': {'Horror': 0.8, 'Thriller': 0.4},
            'love': {'Romance': 0.6, 'Drama': 0.4, 'Comedy': 0.3},
            'magic': {'Fantasy': 0.7, 'Adventure': 0.3, 'Family': 0.2},
            
            # Theme-based associations
            'psychological': {'Thriller': 0.6, 'Horror': 0.5, 'Drama': 0.4},
            'epic': {'Adventure': 0.5, 'Fantasy': 0.5, 'War': 0.4},
            'dark': {'Horror': 0.4, 'Thriller': 0.4, 'Crime': 0.3},
            'family': {'Family': 0.8, 'Comedy': 0.3, 'Adventure': 0.2}
        }
    
    def classify_genres(self, text: str, existing_genres: List[str] = None) -> GenreClassification:
        """Classify genres from text with confidence scoring."""
        logger.debug(f"Classifying genres from text: '{text[:100]}...'")
        
        # Normalize text
        normalized_text = text.lower()
        
        # Score all genres
        genre_scores = self._score_genres(normalized_text)
        
        # Apply existing genre boost if provided
        if existing_genres:
            genre_scores = self._boost_existing_genres(genre_scores, existing_genres)
        
        # Detect subgenres
        subgenre_matches = self._detect_subgenres(normalized_text)
        
        # Create genre matches
        genre_matches = []
        for genre, score in genre_scores.items():
            if score > 0.2:  # Minimum threshold
                match = GenreMatch(
                    genre=genre,
                    confidence=score,
                    reasoning=self._get_genre_reasoning(genre, normalized_text),
                    subgenres=[sg.genre for sg in subgenre_matches if self.parent_genres.get(sg.genre) == genre]
                )
                genre_matches.append(match)
        
        # Sort by confidence
        genre_matches.sort(key=lambda x: x.confidence, reverse=True)
        
        # Separate primary and secondary genres
        primary_genres = [g for g in genre_matches if g.confidence >= 0.5]
        secondary_genres = [g for g in genre_matches if 0.3 <= g.confidence < 0.5]
        
        # Calculate scores
        genre_score = self._calculate_genre_score(genre_matches)
        consistency_score = self._calculate_consistency_score(primary_genres)
        
        classification = GenreClassification(
            primary_genres=primary_genres,
            secondary_genres=secondary_genres,
            subgenres=subgenre_matches,
            genre_score=genre_score,
            consistency_score=consistency_score
        )
        
        logger.debug(f"Classification result: {len(primary_genres)} primary, {len(secondary_genres)} secondary genres")
        return classification
    
    def _score_genres(self, text: str) -> Dict[str, float]:
        """Score all genres based on text content."""
        scores = {genre: 0.0 for genre in self.primary_genres}
        
        # Pattern-based scoring
        for genre, patterns in self.genre_patterns.items():
            if genre in scores:
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    scores[genre] += matches * 0.2
        
        # Keyword-based scoring
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in self.keyword_weights:
                for genre, weight in self.keyword_weights[word].items():
                    if genre in scores:
                        scores[genre] += weight
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {genre: min(score / max_score, 1.0) for genre, score in scores.items()}
        
        return scores
    
    def _boost_existing_genres(self, scores: Dict[str, float], existing_genres: List[str]) -> Dict[str, float]:
        """Boost scores for existing genres to maintain consistency."""
        boosted_scores = scores.copy()
        
        for genre in existing_genres:
            if genre in boosted_scores:
                # Boost existing genres by 50%
                boosted_scores[genre] = min(boosted_scores[genre] * 1.5, 1.0)
        
        return boosted_scores
    
    def _detect_subgenres(self, text: str) -> List[GenreMatch]:
        """Detect subgenres in the text."""
        subgenre_matches = []
        
        for subgenre, patterns in self.subgenre_patterns.items():
            confidence = 0.0
            reasoning = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    confidence += len(matches) * 0.3
                    reasoning.extend(matches)
            
            if confidence > 0.3:
                match = GenreMatch(
                    genre=subgenre,
                    confidence=min(confidence, 1.0),
                    reasoning=reasoning,
                    parent_genres=[self.parent_genres.get(subgenre, '')]
                )
                subgenre_matches.append(match)
        
        return sorted(subgenre_matches, key=lambda x: x.confidence, reverse=True)
    
    def _get_genre_reasoning(self, genre: str, text: str) -> List[str]:
        """Get reasoning for why a genre was detected."""
        reasoning = []
        
        if genre in self.genre_patterns:
            for pattern in self.genre_patterns[genre]:
                matches = re.findall(pattern, text, re.IGNORECASE)
                reasoning.extend(matches)
        
        return reasoning[:3]  # Limit to top 3 reasons
    
    def _calculate_genre_score(self, genre_matches: List[GenreMatch]) -> float:
        """Calculate overall genre classification score."""
        if not genre_matches:
            return 0.0
        
        # Weight by confidence and number of genres
        total_confidence = sum(match.confidence for match in genre_matches)
        avg_confidence = total_confidence / len(genre_matches)
        
        # Penalty for too many genres
        genre_penalty = max(0, len(genre_matches) - 3) * 0.1
        
        return max(0.0, avg_confidence - genre_penalty)
    
    def _calculate_consistency_score(self, primary_genres: List[GenreMatch]) -> float:
        """Calculate consistency score for genre combinations."""
        if len(primary_genres) <= 1:
            return 1.0
        
        consistency = 1.0
        genre_names = [g.genre for g in primary_genres]
        
        # Check compatibility
        for i, genre1 in enumerate(genre_names):
            for genre2 in genre_names[i+1:]:
                # Check if genres are compatible
                if genre1 in self.compatible_genres:
                    if genre2 not in self.compatible_genres[genre1]:
                        consistency -= 0.2
                
                # Check for mutual exclusivity
                pair = tuple(sorted([genre1, genre2]))
                for exclusive_pair, penalty in self.exclusive_genres.items():
                    if pair == exclusive_pair:
                        consistency -= penalty
        
        return max(0.0, consistency)
    
    def get_genre_hierarchy(self, genre: str) -> Dict[str, List[str]]:
        """Get the hierarchy information for a genre."""
        result = {
            'parent': None,
            'children': [],
            'siblings': []
        }
        
        # Check if it's a parent genre
        if genre in self.genre_hierarchy:
            result['children'] = self.genre_hierarchy[genre]
        
        # Check if it's a child genre
        if genre in self.parent_genres:
            parent = self.parent_genres[genre]
            result['parent'] = parent
            result['siblings'] = [g for g in self.genre_hierarchy.get(parent, []) if g != genre]
        
        return result
    
    def suggest_related_genres(self, genres: List[str]) -> List[str]:
        """Suggest related genres based on input genres."""
        suggestions = set()
        
        for genre in genres:
            # Add compatible genres
            if genre in self.compatible_genres:
                suggestions.update(self.compatible_genres[genre])
            
            # Add parent/child relationships
            hierarchy = self.get_genre_hierarchy(genre)
            if hierarchy['parent']:
                suggestions.add(hierarchy['parent'])
            suggestions.update(hierarchy['children'])
        
        # Remove input genres
        suggestions = suggestions - set(genres)
        
        return list(suggestions)
    
    def normalize_genre_name(self, genre: str) -> str:
        """Normalize genre name to standard format."""
        # Common mappings
        mappings = {
            'sci-fi': 'Science Fiction',
            'sci fi': 'Science Fiction',
            'rom-com': 'Romantic Comedy',
            'romcom': 'Romantic Comedy',
            'bio': 'Biography',
            'biopic': 'Biography',
            'doc': 'Documentary',
            'docs': 'Documentary'
        }
        
        normalized = genre.lower().strip()
        
        if normalized in mappings:
            return mappings[normalized]
        
        # Title case for standard genres
        title_cased = genre.title()
        if title_cased in self.primary_genres:
            return title_cased
        
        return genre


# Convenience function for quick classification
def classify_movie_genres(text: str, existing_genres: List[str] = None) -> GenreClassification:
    """Classify genres from movie text."""
    classifier = GenreClassifier()
    return classifier.classify_genres(text, existing_genres)