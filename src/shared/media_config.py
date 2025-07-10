"""
Media Configuration System - Configurable patterns for different media types.

This module provides configuration-based patterns to replace hard-coded 
movie-specific patterns with media-agnostic, configurable alternatives.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from src.shared.media_types import MediaType, MediaConfigProvider

logger = logging.getLogger(__name__)


class FileBasedMediaConfig(MediaConfigProvider):
    """Configuration provider that loads patterns from YAML/JSON files."""
    
    def __init__(self, config_dir: str = "config/media"):
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[MediaType, Dict[str, Any]] = {}
        self._ensure_config_dir()
    
    def _ensure_config_dir(self):
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, media_type: MediaType) -> Dict[str, Any]:
        """Load configuration for a media type."""
        if media_type in self._config_cache:
            return self._config_cache[media_type]
        
        config_file = self.config_dir / f"{media_type.value}.yml"
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}. Creating default config.")
            self._create_default_config(media_type)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            self._config_cache[media_type] = config
            return config
        
        except Exception as e:
            logger.error(f"Error loading config for {media_type.value}: {e}")
            return self._get_fallback_config(media_type)
    
    def _create_default_config(self, media_type: MediaType):
        """Create default configuration file for a media type."""
        config = self._get_default_patterns(media_type)
        config_file = self.config_dir / f"{media_type.value}.yml"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Created default config: {config_file}")
        except Exception as e:
            logger.error(f"Error creating default config for {media_type.value}: {e}")
    
    def get_genre_patterns(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get genre detection patterns for media type."""
        config = self._load_config(media_type)
        return config.get('genres', {})
    
    def get_person_roles(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get person role indicators for media type."""
        config = self._load_config(media_type)
        return config.get('person_roles', {})
    
    def get_quality_indicators(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get quality indicator patterns for media type."""
        config = self._load_config(media_type)
        return config.get('quality_indicators', {})
    
    def get_temporal_patterns(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get temporal expression patterns for media type."""
        config = self._load_config(media_type)
        return config.get('temporal_patterns', {})
    
    def get_theme_keywords(self, media_type: MediaType) -> Dict[str, List[str]]:
        """Get theme keyword mappings for media type."""
        config = self._load_config(media_type)
        return config.get('theme_keywords', {})
    
    def _get_default_patterns(self, media_type: MediaType) -> Dict[str, Any]:
        """Get default patterns for a media type."""
        if media_type == MediaType.MOVIE:
            return self._get_movie_patterns()
        elif media_type == MediaType.BOOK:
            return self._get_book_patterns()
        elif media_type == MediaType.MUSIC:
            return self._get_music_patterns()
        elif media_type == MediaType.TV_SHOW:
            return self._get_tv_patterns()
        else:
            return self._get_generic_patterns()
    
    def _get_movie_patterns(self) -> Dict[str, Any]:
        """Get default movie patterns (migrated from hard-coded patterns)."""
        return {
            'genres': {
                'action': ['action', 'fighting', 'combat', 'martial arts', 'chase', 'explosion'],
                'adventure': ['adventure', 'quest', 'journey', 'expedition', 'exploration'],
                'animation': ['animated', 'animation', 'cartoon', 'anime', 'cgi'],
                'biography': ['biography', 'biopic', 'bio', 'life story', 'true story'],
                'comedy': ['comedy', 'funny', 'hilarious', 'humor', 'comedic', 'laugh'],
                'crime': ['crime', 'criminal', 'heist', 'robbery', 'detective', 'noir'],
                'documentary': ['documentary', 'doc', 'real', 'factual', 'non-fiction'],
                'drama': ['drama', 'dramatic', 'emotional', 'serious', 'character study'],
                'family': ['family', 'kids', 'children', 'all ages', 'wholesome'],
                'fantasy': ['fantasy', 'magical', 'magic', 'wizard', 'dragon', 'fairy'],
                'horror': ['horror', 'scary', 'frightening', 'zombie', 'vampire', 'ghost'],
                'mystery': ['mystery', 'detective', 'investigation', 'clue', 'whodunit'],
                'romance': ['romance', 'romantic', 'love', 'relationship', 'wedding'],
                'science_fiction': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien'],
                'thriller': ['thriller', 'suspense', 'tension', 'psychological', 'paranoia'],
                'war': ['war', 'military', 'battle', 'combat', 'soldier', 'army'],
                'western': ['western', 'cowboy', 'frontier', 'wild west', 'gunslinger']
            },
            'person_roles': {
                'creator': ['director', 'filmmaker', 'directed by'],
                'performer': ['actor', 'actress', 'star', 'starring', 'featuring', 'with']
            },
            'quality_indicators': {
                'positive': ['good', 'great', 'excellent', 'amazing', 'best', 'acclaimed'],
                'negative': ['bad', 'terrible', 'worst', 'awful', 'low quality'],
                'rating': ['rated', 'pg', 'pg-13', 'r', 'nc-17', 'unrated']
            },
            'temporal_patterns': {
                'decade': ['80s', '90s', '2000s', '2010s', 'eighties', 'nineties'],
                'era': ['classic', 'old', 'vintage', 'recent', 'new', 'modern']
            },
            'theme_keywords': {
                'psychological': ['mind', 'psycho', 'mental', 'brain', 'consciousness'],
                'dark': ['dark', 'grim', 'noir', 'bleak', 'sinister'],
                'epic': ['epic', 'grand', 'massive', 'huge', 'sweeping'],
                'indie': ['indie', 'independent', 'art house', 'arthouse']
            }
        }
    
    def _get_book_patterns(self) -> Dict[str, Any]:
        """Get default book patterns."""
        return {
            'genres': {
                'fiction': ['novel', 'story', 'fiction', 'narrative'],
                'non_fiction': ['biography', 'history', 'science', 'memoir', 'essay'],
                'mystery': ['mystery', 'detective', 'crime', 'thriller', 'suspense'],
                'romance': ['romance', 'love story', 'romantic', 'passion'],
                'fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'dragon'],
                'science_fiction': ['sci-fi', 'science fiction', 'futuristic', 'space'],
                'horror': ['horror', 'scary', 'ghost', 'vampire', 'supernatural'],
                'young_adult': ['young adult', 'ya', 'teen', 'teenager'],
                'children': ['children', 'kids', 'picture book', 'early reader'],
                'poetry': ['poetry', 'poems', 'verse', 'haiku', 'sonnet'],
                'drama': ['drama', 'play', 'theater', 'theatrical'],
                'self_help': ['self-help', 'motivational', 'personal development']
            },
            'person_roles': {
                'creator': ['author', 'writer', 'novelist', 'poet'],
                'performer': ['narrator', 'reader', 'voice actor']
            },
            'quality_indicators': {
                'positive': ['bestseller', 'award-winning', 'acclaimed', 'classic'],
                'negative': ['poorly written', 'bad reviews'],
                'rating': ['age appropriate', 'mature content', 'explicit']
            },
            'temporal_patterns': {
                'period': ['victorian', 'medieval', 'renaissance', 'modern', 'contemporary'],
                'publication': ['recently published', 'new release', 'classic', 'vintage']
            },
            'theme_keywords': {
                'literary': ['literary', 'literature', 'prose', 'narrative'],
                'historical': ['historical', 'period', 'era', 'ancient'],
                'philosophical': ['philosophy', 'meaning', 'existence', 'ethics']
            }
        }
    
    def _get_music_patterns(self) -> Dict[str, Any]:
        """Get default music patterns."""
        return {
            'genres': {
                'rock': ['rock', 'guitar', 'drums', 'electric'],
                'pop': ['pop', 'popular', 'catchy', 'mainstream'],
                'hip_hop': ['hip hop', 'rap', 'beats', 'rhymes'],
                'electronic': ['electronic', 'edm', 'synthesizer', 'techno'],
                'classical': ['classical', 'orchestra', 'symphony', 'opera'],
                'jazz': ['jazz', 'improvisation', 'swing', 'blues'],
                'country': ['country', 'folk', 'acoustic', 'rural'],
                'r_and_b': ['r&b', 'soul', 'rhythm and blues', 'vocals'],
                'metal': ['metal', 'heavy', 'aggressive', 'distorted'],
                'reggae': ['reggae', 'caribbean', 'jamaica', 'rastafarian']
            },
            'person_roles': {
                'creator': ['composer', 'songwriter', 'producer', 'arranger'],
                'performer': ['artist', 'musician', 'singer', 'vocalist', 'band']
            },
            'quality_indicators': {
                'positive': ['chart-topping', 'platinum', 'gold', 'grammy-winning'],
                'negative': ['poorly produced', 'off-key'],
                'rating': ['explicit lyrics', 'clean version', 'radio edit']
            },
            'temporal_patterns': {
                'era': ['60s rock', '80s pop', '90s grunge', '2000s indie'],
                'period': ['golden age', 'british invasion', 'disco era']
            },
            'theme_keywords': {
                'emotional': ['love', 'heartbreak', 'joy', 'sadness'],
                'social': ['protest', 'political', 'social justice', 'revolution'],
                'party': ['dance', 'club', 'party', 'celebration']
            }
        }
    
    def _get_tv_patterns(self) -> Dict[str, Any]:
        """Get default TV show patterns."""
        return {
            'genres': {
                'drama': ['drama', 'dramatic', 'serious', 'character-driven'],
                'comedy': ['comedy', 'sitcom', 'funny', 'humorous'],
                'reality': ['reality', 'documentary', 'unscripted', 'real life'],
                'news': ['news', 'current events', 'journalism', 'reporting'],
                'talk_show': ['talk show', 'interview', 'discussion', 'chat'],
                'game_show': ['game show', 'competition', 'contest', 'quiz'],
                'soap_opera': ['soap opera', 'melodrama', 'continuing story'],
                'animated': ['animated', 'cartoon', 'animation'],
                'documentary': ['documentary', 'educational', 'informative'],
                'variety': ['variety', 'entertainment', 'musical', 'comedy sketches']
            },
            'person_roles': {
                'creator': ['creator', 'showrunner', 'producer', 'director'],
                'performer': ['actor', 'actress', 'host', 'presenter']
            },
            'quality_indicators': {
                'positive': ['emmy-winning', 'critically acclaimed', 'popular'],
                'negative': ['cancelled', 'poorly rated'],
                'rating': ['tv-ma', 'tv-14', 'tv-pg', 'tv-g']
            },
            'temporal_patterns': {
                'era': ['golden age', 'modern tv', 'streaming era'],
                'season': ['current season', 'new episodes', 'season finale']
            },
            'theme_keywords': {
                'procedural': ['procedural', 'case-of-the-week', 'investigation'],
                'serialized': ['serialized', 'continuing story', 'mythology'],
                'anthology': ['anthology', 'different stories', 'standalone']
            }
        }
    
    def _get_generic_patterns(self) -> Dict[str, Any]:
        """Get generic patterns for unknown media types."""
        return {
            'genres': {
                'entertainment': ['entertaining', 'fun', 'enjoyable'],
                'educational': ['educational', 'informative', 'learning'],
                'artistic': ['artistic', 'creative', 'expressive']
            },
            'person_roles': {
                'creator': ['creator', 'maker', 'producer'],
                'performer': ['performer', 'participant', 'contributor']
            },
            'quality_indicators': {
                'positive': ['good', 'excellent', 'quality'],
                'negative': ['bad', 'poor quality'],
                'rating': ['appropriate', 'mature']
            },
            'temporal_patterns': {
                'era': ['recent', 'old', 'classic', 'modern']
            },
            'theme_keywords': {
                'general': ['interesting', 'engaging', 'compelling']
            }
        }
    
    def _get_fallback_config(self, media_type: MediaType) -> Dict[str, Any]:
        """Get fallback configuration if file loading fails."""
        logger.warning(f"Using fallback config for {media_type.value}")
        return self._get_generic_patterns()


class AdaptiveThresholds:
    """Adaptive thresholds based on hardware capabilities."""
    
    def __init__(self, cpu_cores: int = 4, memory_gb: float = 8.0):
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
    
    def get_max_expansion_terms(self) -> int:
        """Get maximum expansion terms based on memory."""
        base_terms = 10
        memory_multiplier = max(1, self.memory_gb / 4)
        return min(int(base_terms * memory_multiplier), 50)
    
    def get_parallel_tasks(self) -> int:
        """Get optimal parallel tasks based on CPU cores."""
        return max(1, self.cpu_cores // 2)
    
    def get_processing_timeout(self) -> float:
        """Get processing timeout based on resources."""
        base_timeout = 5.0
        if self.cpu_cores >= 8 and self.memory_gb >= 16:
            return base_timeout * 2
        elif self.cpu_cores <= 2:
            return base_timeout * 0.5
        return base_timeout



_global_config: Optional[FileBasedMediaConfig] = None


def get_media_config() -> FileBasedMediaConfig:
    """Get global media configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = FileBasedMediaConfig()
    return _global_config


def initialize_media_config(config_dir: str = "config/media") -> FileBasedMediaConfig:
    """Initialize global media configuration with custom directory."""
    global _global_config
    _global_config = FileBasedMediaConfig(config_dir)
    return _global_config