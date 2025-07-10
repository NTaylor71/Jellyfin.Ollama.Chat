"""
Media Types - Media-agnostic type definitions and base classes.

This module provides the foundation for a media-agnostic system that can handle
movies, books, music, TV shows, comics, audiobooks, and future media types
without brittle hard-coded assumptions.
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Supported media types for the system."""
    MOVIE = "movie"
    BOOK = "book"
    MUSIC = "music"
    TV_SHOW = "tv_show"
    COMIC = "comic"
    AUDIOBOOK = "audiobook"
    PODCAST = "podcast"
    GAME = "game"


class MediaAgnosticAnalyzer:
    """Media-agnostic analyzer that uses configuration files instead of hard-coded patterns."""
    
    def __init__(self, media_type: MediaType):
        self.media_type = media_type
        self.patterns = self._load_patterns(media_type)
    
    def _load_patterns(self, media_type: MediaType) -> Dict:

        return load_config(f"patterns/{media_type.value}.yml")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent.parent / "config" / config_path
    
    if not config_file.exists():

        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


@dataclass
class MediaPerson:
    """A person associated with media content (actor, author, director, etc.)."""
    name: str
    normalized_name: str
    role: str
    character: Optional[str] = None
    id: Optional[str] = None


@dataclass
class MediaGenre:
    """A genre classification for media content."""
    name: str
    confidence: float
    parent_genre: Optional[str] = None
    subgenres: List[str] = field(default_factory=list)


@dataclass 
class MediaAnalysis:
    """Comprehensive analysis of media content."""
    media_type: MediaType
    title: str
    primary_creators: List[MediaPerson] = field(default_factory=list)
    performers: List[MediaPerson] = field(default_factory=list)
    genres: List[MediaGenre] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    temporal_info: Dict[str, Any] = field(default_factory=dict)
    quality_indicators: List[str] = field(default_factory=list)
    technical_info: Dict[str, Any] = field(default_factory=dict)


class MediaConfigProvider(Protocol):
    """Protocol for providing media-specific configuration."""
    
    def get_genre_patterns(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get genre detection patterns for media type."""
        ...
    
    def get_person_roles(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get person role indicators for media type."""
        ...
    
    def get_quality_indicators(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get quality indicator patterns for media type."""
        ...


class MediaAnalyzer(ABC):
    """Abstract base class for media-agnostic content analysis."""
    
    def __init__(self, config_provider: MediaConfigProvider):
        self.config_provider = config_provider
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze_content(self, content: Dict[str, Any], media_type: MediaType) -> MediaAnalysis:
        """Analyze media content and return structured analysis."""
        pass
    
    @abstractmethod
    def analyze_query(self, query: str, media_type: MediaType) -> MediaAnalysis:
        """Analyze user query for given media type."""
        pass
    
    def get_supported_media_types(self) -> List[MediaType]:
        """Get list of supported media types."""
        return list(MediaType)


class ConfigurableMediaAnalyzer(MediaAnalyzer):
    """Media analyzer that uses configuration files instead of hard-coded patterns."""
    
    def __init__(self, config_provider: MediaConfigProvider):
        super().__init__(config_provider)
        self._pattern_cache: Dict[MediaType, Dict[str, Any]] = {}
    
    def _get_patterns(self, media_type: MediaType) -> Dict[str, Any]:
        """Get cached patterns for media type."""
        if media_type not in self._pattern_cache:
            self._pattern_cache[media_type] = {
                'genres': self.config_provider.get_genre_patterns(media_type),
                'person_roles': self.config_provider.get_person_roles(media_type),
                'quality_indicators': self.config_provider.get_quality_indicators(media_type)
            }
        return self._pattern_cache[media_type]
    
    def analyze_content(self, content: Dict[str, Any], media_type: MediaType) -> MediaAnalysis:
        """Analyze media content using configurable patterns."""
        self.logger.debug(f"Analyzing {media_type.value} content")
        
        patterns = self._get_patterns(media_type)
        

        title = self._extract_title(content, media_type)
        

        primary_creators, performers = self._extract_people(content, media_type, patterns)
        

        genres = self._extract_genres(content, media_type, patterns)
        

        themes, keywords = self._extract_themes_and_keywords(content, media_type, patterns)
        

        temporal_info = self._extract_temporal_info(content, media_type)
        

        quality_indicators = self._extract_quality_indicators(content, media_type, patterns)
        

        technical_info = self._extract_technical_info(content, media_type)
        
        return MediaAnalysis(
            media_type=media_type,
            title=title,
            primary_creators=primary_creators,
            performers=performers,
            genres=genres,
            themes=themes,
            keywords=keywords,
            temporal_info=temporal_info,
            quality_indicators=quality_indicators,
            technical_info=technical_info
        )
    
    def analyze_query(self, query: str, media_type: MediaType) -> MediaAnalysis:
        """Analyze user query using media-type-specific patterns."""
        self.logger.debug(f"Analyzing {media_type.value} query: '{query}'")
        
        patterns = self._get_patterns(media_type)
        


        
        return MediaAnalysis(
            media_type=media_type,
            title=query,

        )
    
    def _extract_title(self, content: Dict[str, Any], media_type: MediaType) -> str:
        """Extract title based on media type."""
        title_fields = {
            MediaType.MOVIE: ['name', 'title', 'original_title'],
            MediaType.BOOK: ['title', 'name'],
            MediaType.MUSIC: ['title', 'album', 'track_name'],
            MediaType.TV_SHOW: ['name', 'series_name', 'title'],
            MediaType.COMIC: ['title', 'issue_title', 'series_name'],
            MediaType.AUDIOBOOK: ['title', 'book_title'],
            MediaType.PODCAST: ['title', 'episode_title'],
            MediaType.GAME: ['title', 'game_name']
        }
        
        for field in title_fields.get(media_type, ['name', 'title']):
            if field in content:
                return content[field]
        
        return "Unknown Title"
    
    def _extract_people(self, content: Dict[str, Any], media_type: MediaType, 
                       patterns: Dict[str, Any]) -> tuple[List[MediaPerson], List[MediaPerson]]:
        """Extract people based on media type."""
        primary_creators = []
        performers = []
        
        people_data = content.get('people', [])
        

        creator_roles = {
            MediaType.MOVIE: ['director', 'producer'],
            MediaType.BOOK: ['author', 'editor'],
            MediaType.MUSIC: ['composer', 'producer', 'songwriter'],
            MediaType.TV_SHOW: ['creator', 'director', 'producer'],
            MediaType.COMIC: ['writer', 'creator'],
            MediaType.AUDIOBOOK: ['author', 'narrator'],
            MediaType.PODCAST: ['host', 'producer'],
            MediaType.GAME: ['developer', 'designer', 'director']
        }
        
        performer_roles = {
            MediaType.MOVIE: ['actor', 'actress'],
            MediaType.BOOK: ['narrator'],
            MediaType.MUSIC: ['artist', 'performer', 'vocalist'],
            MediaType.TV_SHOW: ['actor', 'actress'],
            MediaType.COMIC: ['artist', 'illustrator'],
            MediaType.AUDIOBOOK: ['narrator'],
            MediaType.PODCAST: ['guest', 'co-host'],
            MediaType.GAME: ['voice_actor', 'performer']
        }
        
        creators = creator_roles.get(media_type, [])
        performers_list = performer_roles.get(media_type, [])
        
        for person in people_data:
            role = person.get('type', '').lower()
            
            media_person = MediaPerson(
                name=person.get('name', ''),
                normalized_name=person.get('name', '').lower().strip(),
                role=role,
                character=person.get('role'),
                id=person.get('id'),
                type=person.get('type')
            )
            
            if role in creators:
                primary_creators.append(media_person)
            elif role in performers_list:
                performers.append(media_person)
        
        return primary_creators, performers
    
    def _extract_genres(self, content: Dict[str, Any], media_type: MediaType, 
                       patterns: Dict[str, Any]) -> List[MediaGenre]:
        """Extract genres based on media type patterns."""
        genres = []
        
        
        content_genres = content.get('genres', [])
        
        for genre_name in content_genres:
            genre = MediaGenre(
                name=genre_name,
                confidence=1.0
            )
            genres.append(genre)
        
        return genres
    
    def _extract_themes_and_keywords(self, content: Dict[str, Any], media_type: MediaType, 
                                   patterns: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Extract themes and keywords based on media type."""
        themes = []
        keywords = []
        

        text_fields = ['overview', 'description', 'plot', 'summary']
        text_content = ""
        
        for field in text_fields:
            if field in content:
                text_content += f" {content[field]}"
        

        enhanced_fields = content.get('enhanced_fields', {})
        if enhanced_fields:
            text_content += f" {enhanced_fields.get('summary', '')}"
        

        words = text_content.lower().split()
        keywords = [word for word in words if len(word) > 3][:10]
        
        return themes, keywords
    
    def _extract_temporal_info(self, content: Dict[str, Any], media_type: MediaType) -> Dict[str, Any]:
        """Extract temporal information based on media type."""
        temporal_info = {}
        

        year_fields = {
            MediaType.MOVIE: ['production_year', 'year', 'release_year'],
            MediaType.BOOK: ['publication_year', 'year'],
            MediaType.MUSIC: ['release_year', 'year'],
            MediaType.TV_SHOW: ['first_air_date', 'year'],
            MediaType.COMIC: ['publication_year', 'issue_year'],
            MediaType.AUDIOBOOK: ['publication_year', 'recording_year'],
            MediaType.PODCAST: ['publish_date', 'year'],
            MediaType.GAME: ['release_year', 'year']
        }
        
        for field in year_fields.get(media_type, ['year']):
            if field in content:
                temporal_info['year'] = content[field]
                break
        
        return temporal_info
    
    def _extract_quality_indicators(self, content: Dict[str, Any], media_type: MediaType, 
                                  patterns: Dict[str, Any]) -> List[str]:
        """Extract quality indicators based on media type."""
        quality_indicators = []
        

        rating_fields = {
            MediaType.MOVIE: ['official_rating', 'mpaa_rating'],
            MediaType.BOOK: ['age_rating', 'content_rating'],
            MediaType.MUSIC: ['explicit_rating'],
            MediaType.TV_SHOW: ['content_rating', 'tv_rating'],
            MediaType.COMIC: ['age_rating'],
            MediaType.AUDIOBOOK: ['content_rating'],
            MediaType.PODCAST: ['explicit_rating'],
            MediaType.GAME: ['esrb_rating', 'age_rating']
        }
        
        for field in rating_fields.get(media_type, ['rating']):
            if field in content:
                quality_indicators.append(content[field])
        
        return quality_indicators
    
    def _extract_technical_info(self, content: Dict[str, Any], media_type: MediaType) -> Dict[str, Any]:
        """Extract technical information based on media type."""
        technical_info = {}
        

        if media_type == MediaType.MOVIE:
            technical_info.update({
                'container': content.get('container'),
                'width': content.get('width'),
                'height': content.get('height'),
                'has_subtitles': content.get('has_subtitles'),
                'is_hd': content.get('is_hd')
            })
        elif media_type == MediaType.MUSIC:
            technical_info.update({
                'duration': content.get('duration'),
                'bitrate': content.get('bitrate'),
                'format': content.get('format')
            })
        
        
        return {k: v for k, v in technical_info.items() if v is not None}



def detect_media_type(content: Dict[str, Any]) -> MediaType:
    """Detect media type from content structure."""

    if 'production_year' in content or 'movie_id' in content:
        return MediaType.MOVIE
    elif 'album' in content or 'artist' in content:
        return MediaType.MUSIC
    elif 'isbn' in content or 'author' in content:
        return MediaType.BOOK
    elif 'series_name' in content or 'episode' in content:
        return MediaType.TV_SHOW
    else:
        return MediaType.MOVIE


def get_media_type_from_string(media_type_str: str) -> MediaType:
    """Convert string to MediaType enum."""
    try:
        return MediaType(media_type_str.lower())
    except ValueError:
        return MediaType.MOVIE