"""
Movie Query Analyzer - Intelligent natural language query parsing for movie search.

This module analyzes user queries to extract intent, entities, and structured search parameters.
Handles complex queries like "action movies with Tom Cruise from the 2000s".
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents for movie search."""
    SIMPLE_SEARCH = "simple_search"  # "The Matrix"
    GENRE_SEARCH = "genre_search"    # "action movies"
    ACTOR_SEARCH = "actor_search"    # "Tom Cruise movies"
    DIRECTOR_SEARCH = "director_search"  # "Christopher Nolan films"
    YEAR_SEARCH = "year_search"      # "2000s sci-fi"
    SIMILARITY_SEARCH = "similarity_search"  # "movies like Blade Runner"
    COMPLEX_SEARCH = "complex_search"  # "dark psychological thrillers from 2010s"
    RECOMMENDATION = "recommendation"  # "good action movies"


@dataclass
class EntityExtraction:
    """Extracted entities from a movie query."""
    # People
    actors: List[str] = field(default_factory=list)
    directors: List[str] = field(default_factory=list)
    
    # Content
    genres: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    
    # Time
    years: List[int] = field(default_factory=list)
    decades: List[str] = field(default_factory=list)
    
    # Quality/Rating
    ratings: List[str] = field(default_factory=list)
    quality_terms: List[str] = field(default_factory=list)
    
    # Similarity
    reference_movies: List[str] = field(default_factory=list)
    
    # Technical
    countries: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


@dataclass
class QueryAnalysis:
    """Complete analysis of a movie search query."""
    original_query: str
    intent: QueryIntent
    entities: EntityExtraction
    confidence: float
    processed_terms: List[str]
    search_weights: Dict[str, float] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


class MovieQueryAnalyzer:
    """Analyzes movie search queries to extract intent and entities."""
    
    def __init__(self):
        self._init_patterns()
        self._init_knowledge_base()
    
    def _init_patterns(self):
        """Initialize regex patterns for entity extraction."""
        # Genre patterns
        self.genre_patterns = {
            'action': r'\b(actions?|thriller|adventure|martial arts|kung fu)\b',
            'comedy': r'\b(comed(?:y|ies)|funny|humor|comedic|hilarious)\b',
            'drama': r'\b(dramas?|dramatic|serious|emotional)\b',
            'horror': r'\b(horrors?|scary|frightening|terrifying|slasher)\b',
            'sci-fi': r'\b(sci-?fi|science fiction|futuristic|cyberpunk|dystopian)\b',
            'fantasy': r'\b(fantas(?:y|ies)|magical|magic|wizards|dragons)\b',
            'romance': r'\b(romances?|romantic|love stor(?:y|ies)|date movie)\b',
            'thriller': r'\b(thrillers?|suspense|suspenseful|tense)\b',
            'mystery': r'\b(myster(?:y|ies)|detective|crime|noir|investigation|solving)\b',
            'animation': r'\b(animated|animation|cartoon|anime)\b',
            'documentary': r'\b(documentar(?:y|ies)|doc|real life|true stor(?:y|ies))\b',
            'musical': r'\b(musicals?|music|songs|singing)\b',
            'western': r'\b(westerns?|cowboy|wild west|frontier)\b',
            'war': r'\b(wars?|military|battle|combat|soldier)\b',
            'biography': r'\b(biograph(?:y|ies)|biopic|bio|life stor(?:y|ies))\b'
        }
        
        # Time patterns
        self.time_patterns = {
            'year': r'\b(19|20)\d{2}\b',
            'decade': r'\b(19|20)\d{2}s\b',
            'era': r'\b(classic|old|vintage|recent|new|modern|contemporary)\b'
        }
        
        # Quality patterns
        self.quality_patterns = {
            'positive': r'\b(good|great|excellent|amazing|best|top|quality|acclaimed|award)\b',
            'negative': r'\b(bad|terrible|worst|awful|low quality)\b',
            'rating': r'\b(rated|pg|pg-13|r|nc-17|unrated)\b'
        }
        
        # Intent patterns
        self.intent_patterns = {
            'similarity': r'\b(like|similar to|reminds me of|in the style of)\b',
            'recommendation': r'\b(recommend|suggest|what should|good|best)\b',
            'actor_search': r'\b(with|starring|featuring|actor|actress)\b',
            'director_search': r'\b(directed by|director|by|from)\b'
        }
    
    def _init_knowledge_base(self):
        """Initialize knowledge base of movie-related terms."""
        # Common movie terminology
        self.movie_terms = {
            'action', 'adventure', 'thriller', 'drama', 'comedy', 'horror',
            'sci-fi', 'fantasy', 'romance', 'mystery', 'animation', 'documentary',
            'musical', 'western', 'war', 'biography', 'crime', 'family'
        }
        
        # Thematic keywords
        self.theme_keywords = {
            'psychological': ['mind', 'psycho', 'mental', 'brain', 'consciousness'],
            'dark': ['dark', 'grim', 'noir', 'bleak', 'sinister'],
            'epic': ['epic', 'grand', 'massive', 'huge', 'sweeping'],
            'indie': ['indie', 'independent', 'art house', 'arthouse'],
            'classic': ['classic', 'timeless', 'legendary', 'iconic'],
            'underground': ['underground', 'cult', 'obscure', 'hidden gem']
        }
        
        # Common actor/director indicators
        self.person_indicators = {
            'actor': ['actor', 'actress', 'star', 'starring', 'featuring', 'with'],
            'director': ['director', 'directed', 'filmmaker', 'by', 'from']
        }
        
        # Decade mappings
        self.decades = {
            '1980s': ['80s', '1980s', 'eighties'],
            '1990s': ['90s', '1990s', 'nineties'],
            '2000s': ['2000s', 'aughts', 'early 2000s'],
            '2010s': ['2010s', 'twenty tens', 'recent']
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a movie search query and return structured analysis."""
        logger.debug(f"Analyzing query: '{query}'")
        
        # Normalize query
        normalized_query = self._normalize_query(query)
        
        # Extract entities
        entities = self._extract_entities(normalized_query)
        
        # Determine intent
        intent = self._determine_intent(normalized_query, entities)
        
        # Calculate confidence
        confidence = self._calculate_confidence(normalized_query, entities, intent)
        
        # Process terms for search
        processed_terms = self._process_search_terms(normalized_query, entities)
        
        # Generate search weights
        search_weights = self._generate_search_weights(intent, entities)
        
        # Generate filters
        filters = self._generate_filters(entities)
        
        analysis = QueryAnalysis(
            original_query=query,
            intent=intent,
            entities=entities,
            confidence=confidence,
            processed_terms=processed_terms,
            search_weights=search_weights,
            filters=filters
        )
        
        logger.debug(f"Query analysis: intent={intent.value}, confidence={confidence:.2f}")
        return analysis
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for processing."""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        normalized = re.sub(r'\bsci-?fi\b', 'science fiction', normalized)
        normalized = re.sub(r'\brom-?com\b', 'romantic comedy', normalized)
        
        return normalized
    
    def _extract_entities(self, query: str) -> EntityExtraction:
        """Extract entities from the normalized query."""
        entities = EntityExtraction()
        
        # Extract genres
        entities.genres = self._extract_genres(query)
        
        # Extract time information
        entities.years = self._extract_years(query)
        entities.decades = self._extract_decades(query)
        
        # Extract quality terms
        entities.quality_terms = self._extract_quality_terms(query)
        entities.ratings = self._extract_ratings(query)
        
        # Extract themes
        entities.themes = self._extract_themes(query)
        
        # Extract reference movies (for similarity search)
        entities.reference_movies = self._extract_reference_movies(query)
        
        # Extract people (actors/directors)
        entities.actors, entities.directors = self._extract_people(query)
        
        # Extract countries/languages
        entities.countries = self._extract_countries(query)
        entities.languages = self._extract_languages(query)
        
        # Extract general keywords
        entities.keywords = self._extract_keywords(query, entities)
        
        return entities
    
    def _extract_genres(self, query: str) -> List[str]:
        """Extract genre information from query."""
        found_genres = []
        
        for genre, pattern in self.genre_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                found_genres.append(genre)
        
        return found_genres
    
    def _extract_years(self, query: str) -> List[int]:
        """Extract year information from query."""
        years = []
        
        # Direct year matches - find complete 4-digit years
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        for year_str in year_matches:
            try:
                year = int(year_str)
                if 1900 <= year <= datetime.now().year + 5:  # Allow future years
                    years.append(year)
            except ValueError:
                continue
        
        return years
    
    def _extract_decades(self, query: str) -> List[str]:
        """Extract decade information from query."""
        decades = []
        
        # Direct decade matches - find complete decade patterns
        decade_matches = re.findall(r'\b(19\d{2}s|20\d{2}s)\b', query)
        for decade_str in decade_matches:
            decades.append(decade_str)
        
        # Named decades
        for decade_name, aliases in self.decades.items():
            for alias in aliases:
                if alias in query:
                    decades.append(decade_name)
                    break
        
        return list(set(decades))
    
    def _extract_quality_terms(self, query: str) -> List[str]:
        """Extract quality-related terms from query."""
        quality_terms = []
        
        for quality_type, pattern in self.quality_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            quality_terms.extend(matches)
        
        return quality_terms
    
    def _extract_ratings(self, query: str) -> List[str]:
        """Extract rating information from query."""
        ratings = []
        
        # MPAA ratings
        rating_matches = re.findall(r'\b(pg-13|pg|r|nc-17|unrated)\b', query, re.IGNORECASE)
        ratings.extend([rating.upper() for rating in rating_matches])
        
        return ratings
    
    def _extract_themes(self, query: str) -> List[str]:
        """Extract thematic keywords from query."""
        themes = []
        
        for theme, keywords in self.theme_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    themes.append(theme)
                    break
        
        return themes
    
    def _extract_reference_movies(self, query: str) -> List[str]:
        """Extract reference movies for similarity search."""
        reference_movies = []
        
        # Look for "like [movie]" or "similar to [movie]" patterns
        similarity_patterns = [
            r'like\s+([A-Z][^,\.!?\n]*)',
            r'similar to\s+([A-Z][^,\.!?\n]*)',
            r'reminds me of\s+([A-Z][^,\.!?\n]*)'
        ]
        
        for pattern in similarity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            reference_movies.extend([match.strip() for match in matches])
        
        return reference_movies
    
    def _extract_people(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract actor and director names from query."""
        actors = []
        directors = []
        
        # Look for "with [person]" or "starring [person]" patterns
        actor_patterns = [
            r'with\s+([a-zA-Z]+\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+from|\s*$)',
            r'starring\s+([a-zA-Z]+\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+from|\s*$)',
            r'featuring\s+([a-zA-Z]+\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s+from|\s*$)'
        ]
        
        for pattern in actor_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Title case the match since we extracted it case-insensitively
                clean_match = match.strip().title()
                if self._looks_like_person_name(clean_match):
                    actors.append(clean_match)
        
        # Look for "directed by [person]" or "by [person]" patterns  
        director_patterns = [
            r'directed by\s+([a-zA-Z]+\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s*$)',
            r'director\s+([a-zA-Z]+\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*?)(?:\s*$)'
        ]
        
        for pattern in director_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Title case the match since we extracted it case-insensitively  
                clean_match = match.strip().title()
                if self._looks_like_person_name(clean_match):
                    directors.append(clean_match)
        
        return actors, directors
    
    def _looks_like_person_name(self, name: str) -> bool:
        """Check if a string looks like a person name."""
        if not name or len(name) < 3:
            return False
        
        # Should have at least 2 words (first and last name)
        words = name.split()
        if len(words) < 2:
            return False
        
        # Each word should start with a capital letter
        for word in words:
            if not word[0].isupper():
                return False
        
        # Shouldn't contain movie-related terms
        movie_terms = {'movie', 'movies', 'film', 'films', 'from', 'the', 'and', 'or'}
        if any(word.lower() in movie_terms for word in words):
            return False
        
        return True
    
    def _extract_countries(self, query: str) -> List[str]:
        """Extract country/nationality information from query."""
        countries = []
        
        # Common country/nationality patterns
        country_patterns = {
            'American': ['american', 'usa', 'us', 'hollywood'],
            'Australian': ['australian', 'aussie'],
            'British': ['british', 'uk', 'england', 'english', 'scotland', 'scottish', 'wales', 'welsh', 'ireland', 'irish'],
            'French': ['french', 'france'],
            'German': ['german', 'germany'],
            'Japanese': ['japanese', 'japan'],
            'Korean': ['korean', 'korea'],
            'Italian': ['italian', 'italy'],
            'Spanish': ['spanish', 'spain'],
            'Finnish': ['finnish', 'finland'],
            'Swedish': ['swedish', 'sweden'],
            'Norwegian': ['norwegian', 'norway'],
            'Danish': ['danish', 'denmark']
        }
        
        for country, patterns in country_patterns.items():
            for pattern in patterns:
                if pattern in query.lower():
                    countries.append(country)
                    break
        
        return countries
    
    def _extract_languages(self, query: str) -> List[str]:
        """Extract language information from query."""
        languages = []
        
        # Look for language indicators
        language_patterns = [
            r'\b(english|french|german|spanish|italian|japanese|korean|chinese|finnish|swedish)\b'
        ]
        
        for pattern in language_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            languages.extend([lang.title() for lang in matches])
        
        return languages
    
    def _extract_keywords(self, query: str, entities: EntityExtraction) -> List[str]:
        """Extract general search keywords, excluding already extracted entities."""
        # Get all words from query
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out common words and already extracted entities
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'about', 'like', 'similar', 'movie', 'movies', 'film', 'films'
        }
        
        # Combine all extracted entity terms
        extracted_terms = set()
        extracted_terms.update([g.lower() for g in entities.genres])
        extracted_terms.update([str(y) for y in entities.years])
        extracted_terms.update([d.lower() for d in entities.decades])
        extracted_terms.update([t.lower() for t in entities.themes])
        extracted_terms.update([a.lower() for a in entities.actors])
        extracted_terms.update([d.lower() for d in entities.directors])
        
        # Filter keywords
        keywords = []
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                word not in extracted_terms and
                not word.isdigit()):
                keywords.append(word)
        
        return keywords
    
    def _determine_intent(self, query: str, entities: EntityExtraction) -> QueryIntent:
        """Determine the primary intent of the query."""
        # Count different types of entities to detect complex searches first
        entity_types = 0
        if entities.genres: entity_types += 1
        if entities.themes: entity_types += 1
        if entities.years or entities.decades: entity_types += 1
        if entities.quality_terms: entity_types += 1
        if entities.countries: entity_types += 1
        if entities.actors: entity_types += 1
        if entities.directors: entity_types += 1
        
        # Check for similarity search (highest priority)
        if (entities.reference_movies or 
            any(pattern in query for pattern in ['like', 'similar to', 'reminds me of'])):
            return QueryIntent.SIMILARITY_SEARCH
        
        # Check for recommendation request (high priority)
        if any(pattern in query for pattern in ['recommend', 'suggest', 'what should', 'good', 'best']):
            return QueryIntent.RECOMMENDATION
        
        # Check for complex search (multiple entity types)
        if entity_types >= 2:
            return QueryIntent.COMPLEX_SEARCH
        
        # Check for actor search (only if single entity type)
        if entities.actors:
            return QueryIntent.ACTOR_SEARCH
        
        # Check for director search (only if single entity type)
        if entities.directors:
            return QueryIntent.DIRECTOR_SEARCH
        
        # Check for year/decade search (only if single entity type)
        if entities.years or entities.decades:
            return QueryIntent.YEAR_SEARCH
        
        # Check for genre search (only if single entity type)
        if entities.genres:
            return QueryIntent.GENRE_SEARCH
        
        # Default to simple search
        return QueryIntent.SIMPLE_SEARCH
    
    def _calculate_confidence(self, query: str, entities: EntityExtraction, intent: QueryIntent) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on extracted entities
        if entities.genres:
            confidence += 0.1 * len(entities.genres)
        if entities.actors:
            confidence += 0.15 * len(entities.actors)
        if entities.directors:
            confidence += 0.15 * len(entities.directors)
        if entities.years or entities.decades:
            confidence += 0.1
        if entities.themes:
            confidence += 0.1 * len(entities.themes)
        if entities.reference_movies:
            confidence += 0.2 * len(entities.reference_movies)
        
        # Boost confidence based on intent-specific patterns
        intent_patterns = {
            QueryIntent.SIMILARITY_SEARCH: ['like', 'similar', 'reminds'],
            QueryIntent.RECOMMENDATION: ['recommend', 'suggest', 'good', 'best'],
            QueryIntent.ACTOR_SEARCH: ['with', 'starring', 'featuring'],
            QueryIntent.DIRECTOR_SEARCH: ['directed', 'director', 'filmmaker']
        }
        
        if intent in intent_patterns:
            for pattern in intent_patterns[intent]:
                if pattern in query:
                    confidence += 0.1
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def _process_search_terms(self, query: str, entities: EntityExtraction) -> List[str]:
        """Process query into search terms."""
        terms = []
        
        # Add original keywords
        terms.extend(entities.keywords)
        
        # Add genre terms
        terms.extend(entities.genres)
        
        # Add theme terms
        terms.extend(entities.themes)
        
        # Add quality terms
        terms.extend(entities.quality_terms)
        
        # Add people names
        terms.extend(entities.actors)
        terms.extend(entities.directors)
        
        # Add reference movies
        terms.extend(entities.reference_movies)
        
        return terms
    
    def _generate_search_weights(self, intent: QueryIntent, entities: EntityExtraction) -> Dict[str, float]:
        """Generate field weights based on intent and entities."""
        weights = {
            'name': 1.0,
            'overview': 1.0,
            'genres': 1.0,
            'people': 1.0,
            'enhanced_fields': 1.0
        }
        
        # Adjust weights based on intent
        if intent == QueryIntent.ACTOR_SEARCH:
            weights['people'] = 3.0
            weights['name'] = 0.5
        elif intent == QueryIntent.DIRECTOR_SEARCH:
            weights['people'] = 3.0
            weights['name'] = 0.5
        elif intent == QueryIntent.GENRE_SEARCH:
            weights['genres'] = 2.5
            weights['enhanced_fields'] = 1.5
        elif intent == QueryIntent.SIMILARITY_SEARCH:
            weights['enhanced_fields'] = 2.0
            weights['overview'] = 1.5
        elif intent == QueryIntent.RECOMMENDATION:
            weights['enhanced_fields'] = 1.5
            weights['genres'] = 1.5
        
        return weights
    
    def _generate_filters(self, entities: EntityExtraction) -> Dict[str, Any]:
        """Generate MongoDB filters based on extracted entities."""
        filters = {}
        
        # Year and decade filters (combine them)
        year_values = []
        
        # Add explicit years
        if entities.years:
            year_values.extend(entities.years)
        
        # Add decade ranges
        if entities.decades:
            for decade in entities.decades:
                if decade.endswith('s'):
                    start_year = int(decade[:4])
                    year_values.extend(range(start_year, start_year + 10))
        
        # Set the filter if we have any year values
        if year_values:
            if len(year_values) == 1:
                filters['production_year'] = year_values[0]
            else:
                filters['production_year'] = {'$in': list(set(year_values))}
        
        # Genre filters
        if entities.genres:
            # Map analyzed genres to MongoDB genre format
            genre_mapping = {
                'sci-fi': 'Science Fiction',
                'rom-com': 'Romance',
                'action': 'Action',
                'comedy': 'Comedy',
                'drama': 'Drama',
                'horror': 'Horror',
                'thriller': 'Thriller'
            }
            
            mapped_genres = []
            for genre in entities.genres:
                mapped_genres.append(genre_mapping.get(genre, genre.title()))
            
            filters['genres'] = {'$in': mapped_genres}
        
        # Rating filters
        if entities.ratings:
            filters['official_rating'] = {'$in': entities.ratings}
        
        # Country filters
        if entities.countries:
            filters['production_locations'] = {'$in': entities.countries}
        
        return filters


# Convenience function for quick analysis
def analyze_movie_query(query: str) -> QueryAnalysis:
    """Analyze a movie query and return structured analysis."""
    analyzer = MovieQueryAnalyzer()
    return analyzer.analyze_query(query)