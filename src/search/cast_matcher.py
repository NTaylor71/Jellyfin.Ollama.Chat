"""
Cast/Crew Matcher - Advanced actor and director matching with normalization and relationships.

This module handles intelligent matching of cast and crew members, including:
- Name normalization and aliases
- Character name to actor mapping
- Collaborative relationship detection
- Fuzzy matching for variations
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class PersonMatch:
    """A matched person with confidence and metadata."""
    name: str
    normalized_name: str
    confidence: float
    person_type: str  # Actor, Director, Writer, Producer
    character_role: Optional[str] = None
    jellyfin_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    match_reason: str = ""


@dataclass
class CollaborationMatch:
    """A collaboration relationship between people."""
    person1: str
    person2: str
    collaboration_type: str  # actor-director, actor-actor, etc.
    confidence: float
    shared_projects: List[str] = field(default_factory=list)


@dataclass
class CastSearchResult:
    """Result of cast/crew search operation."""
    matches: List[PersonMatch]
    collaborations: List[CollaborationMatch]
    search_confidence: float
    suggested_searches: List[str] = field(default_factory=list)


class CastMatcher:
    """Advanced cast and crew matching system."""
    
    def __init__(self):
        self._init_name_patterns()
        self._init_common_aliases()
        self._init_character_mappings()
        self._init_collaboration_patterns()
    
    def _init_name_patterns(self):
        """Initialize patterns for name normalization."""
        # Common name prefixes and suffixes
        self.name_prefixes = {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sir', 'dame',
            'von', 'van', 'de', 'del', 'da', 'du', 'le', 'la'
        }
        
        self.name_suffixes = {
            'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'
        }
        
        # Unicode normalization patterns
        self.diacritic_mappings = {
            'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a', 'å': 'a',
            'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
            'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
            'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ø': 'o',
            'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
            'ý': 'y', 'ÿ': 'y',
            'ñ': 'n', 'ç': 'c', 'ß': 'ss'
        }
    
    def _init_common_aliases(self):
        """Initialize common name aliases and variations."""
        # These would ideally be loaded from a database
        self.common_aliases = {
            # Example aliases - in production, this would be much larger
            'tom cruise': ['thomas cruise mapother iv', 'thomas mapother'],
            'will smith': ['willard smith jr', 'willard carroll smith jr'],
            'robert downey jr': ['robert john downey jr', 'rdj'],
            'the rock': ['dwayne johnson', 'dwayne the rock johnson'],
            'vin diesel': ['mark sinclair', 'mark sinclair vincent'],
            'natalie portman': ['natalie hershlag', 'neta-lee hershlag'],
            'michael caine': ['maurice micklewhite', 'sir michael caine'],
            'helen mirren': ['helen lydia mironoff', 'dame helen mirren'],
            
            # Directors
            'christopher nolan': ['chris nolan'],
            'quentin tarantino': ['qt', 'tarantino'],
            'martin scorsese': ['marty scorsese'],
            'steven spielberg': ['spielberg'],
            'james cameron': ['jim cameron']
        }
        
        # Reverse mapping
        self.alias_to_canonical = {}
        for canonical, aliases in self.common_aliases.items():
            self.alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self.alias_to_canonical[alias] = canonical
    
    def _init_character_mappings(self):
        """Initialize character name to actor mappings."""
        # Famous character mappings
        self.character_mappings = {
            # Example mappings - would be much larger in production
            'james bond': ['daniel craig', 'pierce brosnan', 'timothy dalton', 'roger moore', 'sean connery'],
            'batman': ['christian bale', 'michael keaton', 'val kilmer', 'george clooney', 'ben affleck'],
            'superman': ['christopher reeve', 'brandon routh', 'henry cavill'],
            'spider-man': ['tobey maguire', 'andrew garfield', 'tom holland'],
            'iron man': ['robert downey jr'],
            'wolverine': ['hugh jackman'],
            'john wick': ['keanu reeves'],
            'ethan hunt': ['tom cruise'],
            'luke skywalker': ['mark hamill'],
            'han solo': ['harrison ford'],
            'indiana jones': ['harrison ford'],
            'rocky balboa': ['sylvester stallone'],
            'rambo': ['sylvester stallone'],
            'terminator': ['arnold schwarzenegger'],
            'ellen ripley': ['sigourney weaver'],
            'sarah connor': ['linda hamilton'],
            'lara croft': ['angelina jolie', 'alicia vikander']
        }
    
    def _init_collaboration_patterns(self):
        """Initialize patterns for detecting collaborations."""
        # Common collaboration indicators
        self.collaboration_indicators = {
            'frequent_collaborators': [
                ('leonardo dicaprio', 'martin scorsese'),
                ('johnny depp', 'tim burton'),
                ('robert de niro', 'martin scorsese'),
                ('samuel l jackson', 'quentin tarantino'),
                ('helena bonham carter', 'tim burton'),
                ('michael caine', 'christopher nolan'),
                ('scarlett johansson', 'woody allen'),
                ('bill murray', 'wes anderson')
            ],
            'franchise_actors': {
                'marvel': ['robert downey jr', 'chris evans', 'chris hemsworth', 'scarlett johansson'],
                'dc': ['henry cavill', 'ben affleck', 'gal gadot'],
                'star wars': ['mark hamill', 'harrison ford', 'carrie fisher'],
                'fast and furious': ['vin diesel', 'paul walker', 'michelle rodriguez'],
                'mission impossible': ['tom cruise', 'ving rhames', 'simon pegg']
            }
        }
    
    def normalize_person_name(self, name: str) -> str:
        """Normalize a person's name for consistent matching."""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle diacritics
        normalized = self._remove_diacritics(normalized)
        
        # Remove common prefixes and suffixes
        words = normalized.split()
        filtered_words = []
        
        for word in words:
            # Skip prefixes
            if word.rstrip('.') not in self.name_prefixes:
                # Handle suffixes
                if word.rstrip('.') in self.name_suffixes:
                    # Keep suffixes but normalize them
                    if word in ['jr.', 'jr']:
                        filtered_words.append('jr')
                    elif word in ['sr.', 'sr']:
                        filtered_words.append('sr')
                    else:
                        filtered_words.append(word.rstrip('.'))
                else:
                    filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text."""
        # First try Unicode normalization
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
        
        # Apply custom mappings for characters that don't normalize well
        for accented, plain in self.diacritic_mappings.items():
            ascii_text = ascii_text.replace(accented, plain)
        
        return ascii_text
    
    def find_person_matches(self, query_name: str, people_data: List[Dict[str, Any]], 
                           person_type: Optional[str] = None, min_confidence: float = 0.3) -> List[PersonMatch]:
        """Find matching people in the database."""
        logger.debug(f"Finding matches for: '{query_name}' (type: {person_type})")
        
        normalized_query = self.normalize_person_name(query_name)
        matches = []
        
        for person in people_data:
            person_name = person.get('name', '')
            person_role = person.get('role', '')
            person_type_db = person.get('type', '')
            person_id = person.get('id', '')
            
            # Skip if type filter specified and doesn't match
            if person_type and person_type_db.lower() != person_type.lower():
                continue
            
            # Calculate match confidence
            confidence = self._calculate_person_match_confidence(
                normalized_query, person_name, person_role
            )
            
            if confidence >= min_confidence:
                match = PersonMatch(
                    name=person_name,
                    normalized_name=self.normalize_person_name(person_name),
                    confidence=confidence,
                    person_type=person_type_db,
                    character_role=person_role if person_type_db.lower() == 'actor' else None,
                    jellyfin_id=person_id,
                    match_reason=self._get_match_reason(normalized_query, person_name, person_role)
                )
                matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.debug(f"Found {len(matches)} matches for '{query_name}'")
        return matches
    
    def _calculate_person_match_confidence(self, query: str, name: str, role: str = "") -> float:
        """Calculate confidence score for person match."""
        normalized_name = self.normalize_person_name(name)
        
        # Exact match
        if query == normalized_name:
            return 1.0
        
        # Check aliases
        canonical_query = self.alias_to_canonical.get(query, query)
        canonical_name = self.alias_to_canonical.get(normalized_name, normalized_name)
        
        if canonical_query == canonical_name:
            return 0.95
        
        # Partial name matching
        query_parts = set(query.split())
        name_parts = set(normalized_name.split())
        
        # All query parts in name
        if query_parts.issubset(name_parts):
            return 0.9
        
        # Fuzzy string matching
        similarity = SequenceMatcher(None, query, normalized_name).ratio()
        
        # Boost for partial matches
        if any(part in normalized_name for part in query_parts):
            similarity += 0.2
        
        # Character role matching
        if role and query in role.lower():
            similarity += 0.3
        
        # Check character mappings
        if query in self.character_mappings:
            if normalized_name in [self.normalize_person_name(actor) for actor in self.character_mappings[query]]:
                return 0.85
        
        return min(similarity, 1.0)
    
    def _get_match_reason(self, query: str, name: str, role: str = "") -> str:
        """Get human-readable reason for the match."""
        normalized_name = self.normalize_person_name(name)
        
        if query == normalized_name:
            return "exact name match"
        
        if query in self.alias_to_canonical and normalized_name in self.alias_to_canonical:
            if self.alias_to_canonical[query] == self.alias_to_canonical[normalized_name]:
                return "alias match"
        
        query_parts = set(query.split())
        name_parts = set(normalized_name.split())
        
        if query_parts.issubset(name_parts):
            return "partial name match"
        
        if role and query in role.lower():
            return f"character role match ({role})"
        
        if query in self.character_mappings:
            if normalized_name in [self.normalize_person_name(actor) for actor in self.character_mappings[query]]:
                return f"character mapping ({query})"
        
        similarity = SequenceMatcher(None, query, normalized_name).ratio()
        if similarity > 0.7:
            return "fuzzy name match"
        
        return "partial match"
    
    def find_collaborations(self, people: List[str], movies_data: List[Dict[str, Any]]) -> List[CollaborationMatch]:
        """Find collaboration relationships between people."""
        collaborations = []
        
        # Look for direct collaborations in the data
        for i, person1 in enumerate(people):
            for person2 in people[i+1:]:
                collab = self._find_collaboration_between(person1, person2, movies_data)
                if collab:
                    collaborations.append(collab)
        
        # Check known frequent collaborators
        for person1 in people:
            for person2 in people:
                if person1 != person2:
                    collab = self._check_known_collaborations(person1, person2)
                    if collab and collab not in collaborations:
                        collaborations.append(collab)
        
        return collaborations
    
    def _find_collaboration_between(self, person1: str, person2: str, 
                                  movies_data: List[Dict[str, Any]]) -> Optional[CollaborationMatch]:
        """Find collaboration between two specific people."""
        norm_person1 = self.normalize_person_name(person1)
        norm_person2 = self.normalize_person_name(person2)
        
        shared_movies = []
        collaboration_types = set()
        
        for movie in movies_data:
            people_in_movie = movie.get('people', [])
            
            person1_roles = []
            person2_roles = []
            
            for person in people_in_movie:
                norm_name = self.normalize_person_name(person.get('name', ''))
                
                if self._names_match(norm_person1, norm_name):
                    person1_roles.append(person.get('type', ''))
                elif self._names_match(norm_person2, norm_name):
                    person2_roles.append(person.get('type', ''))
            
            if person1_roles and person2_roles:
                shared_movies.append(movie.get('name', 'Unknown'))
                
                # Determine collaboration type
                for role1 in person1_roles:
                    for role2 in person2_roles:
                        if role1.lower() == 'actor' and role2.lower() == 'director':
                            collaboration_types.add('actor-director')
                        elif role1.lower() == 'director' and role2.lower() == 'actor':
                            collaboration_types.add('director-actor')
                        elif role1.lower() == 'actor' and role2.lower() == 'actor':
                            collaboration_types.add('actor-actor')
                        elif role1.lower() == 'director' and role2.lower() == 'director':
                            collaboration_types.add('director-director')
                        else:
                            collaboration_types.add(f'{role1.lower()}-{role2.lower()}')
        
        if shared_movies:
            # Calculate confidence based on number of collaborations
            confidence = min(len(shared_movies) * 0.3, 1.0)
            
            return CollaborationMatch(
                person1=person1,
                person2=person2,
                collaboration_type=list(collaboration_types)[0] if collaboration_types else 'unknown',
                confidence=confidence,
                shared_projects=shared_movies
            )
        
        return None
    
    def _check_known_collaborations(self, person1: str, person2: str) -> Optional[CollaborationMatch]:
        """Check if people are known frequent collaborators."""
        norm_person1 = self.normalize_person_name(person1)
        norm_person2 = self.normalize_person_name(person2)
        
        # Check frequent collaborators list
        for collab_pair in self.collaboration_indicators['frequent_collaborators']:
            norm_pair = (self.normalize_person_name(collab_pair[0]), 
                        self.normalize_person_name(collab_pair[1]))
            
            if ((norm_person1, norm_person2) == norm_pair or 
                (norm_person2, norm_person1) == norm_pair):
                return CollaborationMatch(
                    person1=person1,
                    person2=person2,
                    collaboration_type='frequent-collaborators',
                    confidence=0.8,
                    shared_projects=['Multiple known collaborations']
                )
        
        # Check franchise associations
        for franchise, actors in self.collaboration_indicators['franchise_actors'].items():
            norm_actors = [self.normalize_person_name(actor) for actor in actors]
            
            if norm_person1 in norm_actors and norm_person2 in norm_actors:
                return CollaborationMatch(
                    person1=person1,
                    person2=person2,
                    collaboration_type=f'{franchise}-franchise',
                    confidence=0.6,
                    shared_projects=[f'{franchise.title()} franchise']
                )
        
        return None
    
    def _names_match(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two normalized names match."""
        if name1 == name2:
            return True
        
        # Check aliases
        canonical1 = self.alias_to_canonical.get(name1, name1)
        canonical2 = self.alias_to_canonical.get(name2, name2)
        
        if canonical1 == canonical2:
            return True
        
        # Fuzzy matching
        similarity = SequenceMatcher(None, name1, name2).ratio()
        return similarity >= threshold
    
    def search_cast_and_crew(self, query: str, movies_data: List[Dict[str, Any]], 
                           search_type: str = 'all') -> CastSearchResult:
        """Search for cast and crew across movie database."""
        logger.debug(f"Searching cast/crew: '{query}' (type: {search_type})")
        
        all_people = []
        for movie in movies_data:
            all_people.extend(movie.get('people', []))
        
        # Remove duplicates based on normalized names
        unique_people = {}
        for person in all_people:
            norm_name = self.normalize_person_name(person.get('name', ''))
            if norm_name not in unique_people:
                unique_people[norm_name] = person
        
        # Find matches
        person_type = None
        if search_type == 'actor':
            person_type = 'Actor'
        elif search_type == 'director':
            person_type = 'Director'
        
        matches = self.find_person_matches(
            query, list(unique_people.values()), person_type
        )
        
        # Find collaborations among top matches
        top_match_names = [match.name for match in matches[:5]]
        collaborations = self.find_collaborations(top_match_names, movies_data)
        
        # Calculate search confidence
        search_confidence = max([match.confidence for match in matches]) if matches else 0.0
        
        # Generate suggested searches
        suggested_searches = self._generate_suggestions(query, matches)
        
        result = CastSearchResult(
            matches=matches,
            collaborations=collaborations,
            search_confidence=search_confidence,
            suggested_searches=suggested_searches
        )
        
        logger.debug(f"Cast search result: {len(matches)} matches, {len(collaborations)} collaborations")
        return result
    
    def _generate_suggestions(self, query: str, matches: List[PersonMatch]) -> List[str]:
        """Generate suggested searches based on matches."""
        suggestions = []
        
        # If we have good matches, suggest related searches
        if matches and matches[0].confidence > 0.7:
            top_match = matches[0]
            
            # Suggest character searches if it's an actor
            if top_match.person_type.lower() == 'actor' and top_match.character_role:
                suggestions.append(f"movies with {top_match.character_role}")
            
            # Suggest collaboration searches
            canonical_name = self.alias_to_canonical.get(
                self.normalize_person_name(top_match.name), 
                self.normalize_person_name(top_match.name)
            )
            
            for collab_pair in self.collaboration_indicators['frequent_collaborators']:
                norm_pair = [self.normalize_person_name(name) for name in collab_pair]
                
                if canonical_name in norm_pair:
                    other_person = collab_pair[0] if canonical_name == norm_pair[1] else collab_pair[1]
                    suggestions.append(f"{other_person} movies")
        
        # If no good matches, suggest alternatives
        elif not matches or matches[0].confidence < 0.5:
            # Check if it might be a character name
            if query in self.character_mappings:
                suggestions.extend([f"{actor} movies" for actor in self.character_mappings[query][:3]])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def get_person_filmography_summary(self, person_name: str, movies_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of a person's filmography."""
        normalized_name = self.normalize_person_name(person_name)
        
        filmography = {
            'total_movies': 0,
            'roles': set(),
            'genres': set(),
            'collaborators': set(),
            'movies': []
        }
        
        for movie in movies_data:
            for person in movie.get('people', []):
                if self._names_match(normalized_name, self.normalize_person_name(person.get('name', ''))):
                    filmography['total_movies'] += 1
                    filmography['roles'].add(person.get('type', 'Unknown'))
                    filmography['genres'].update(movie.get('genres', []))
                    filmography['movies'].append({
                        'title': movie.get('name', ''),
                        'year': movie.get('production_year'),
                        'role': person.get('role', ''),
                        'type': person.get('type', '')
                    })
                    
                    # Add collaborators
                    for other_person in movie.get('people', []):
                        if other_person.get('name') != person.get('name'):
                            filmography['collaborators'].add(other_person.get('name', ''))
        
        # Convert sets to lists for JSON serialization
        filmography['roles'] = list(filmography['roles'])
        filmography['genres'] = list(filmography['genres'])
        filmography['collaborators'] = list(filmography['collaborators'])[:10]  # Limit collaborators
        
        return filmography


# Convenience functions
def find_actor_matches(query: str, people_data: List[Dict[str, Any]]) -> List[PersonMatch]:
    """Find actor matches for a query."""
    matcher = CastMatcher()
    return matcher.find_person_matches(query, people_data, 'Actor')


def find_director_matches(query: str, people_data: List[Dict[str, Any]]) -> List[PersonMatch]:
    """Find director matches for a query."""
    matcher = CastMatcher()
    return matcher.find_person_matches(query, people_data, 'Director')


def search_movie_cast(query: str, movies_data: List[Dict[str, Any]]) -> CastSearchResult:
    """Search for cast and crew in movie database."""
    matcher = CastMatcher()
    return matcher.search_cast_and_crew(query, movies_data)