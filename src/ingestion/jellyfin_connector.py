"""
Jellyfin data ingestion connector.
"""

import asyncio
import re
import unicodedata
from typing import List, Optional, Dict, Any, Union
import httpx
import structlog

from ..shared.config import get_settings
from ..data.mongo_client import get_mongo_client
from ..data.models import MovieCreate, Movie, Person, Studio, Chapter, ExternalUrl

logger = structlog.get_logger(__name__)


def to_ascii(txt: Union[str, None]) -> str:
    if not txt:
        return ""
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode("ascii")

def ascii_list(vals: List[str] | None) -> List[str]:
    return [to_ascii(v) for v in vals or []]


class JellyfinConnector:
    """Connector for ingesting movie data from Jellyfin."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.mongo_client = None
        
    async def connect(self) -> None:
        """Initialize connections."""
        self.client = httpx.AsyncClient(timeout=30.0)
        self.mongo_client = await get_mongo_client()
        
        # Test Jellyfin connection
        await self._test_jellyfin_connection()
        
    async def disconnect(self) -> None:
        """Close connections."""
        if self.client:
            await self.client.aclose()
    
    async def _test_jellyfin_connection(self) -> None:
        """Test connection to Jellyfin."""
        try:
            url = f"{self.settings.JELLYFIN_URL}/System/Info"
            headers = {"X-Emby-Token": self.settings.JELLYFIN_API_KEY}
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            system_info = response.json()
            logger.info("Connected to Jellyfin", 
                       server_name=system_info.get("ServerName", "Unknown"),
                       version=system_info.get("Version", "Unknown"))
            
        except Exception as e:
            logger.error("Failed to connect to Jellyfin", error=str(e))
            raise
    
    async def get_movies(self, limit: int = 100, start_index: int = 0) -> List[Dict[str, Any]]:
        """Get movies from Jellyfin (single chunk)."""
        try:
            url = f"{self.settings.JELLYFIN_URL}/Items"
            headers = {"X-Emby-Token": self.settings.JELLYFIN_API_KEY}
            
            params = {
                "IncludeItemTypes": "Movie",
                "Recursive": True,
                "Fields": "Overview,Genres,People,MediaStreams,OriginalTitle,PremiereDate,ProductionYear,Taglines,CommunityRating,CriticRating,OfficialRating,RunTimeTicks,ProviderIds,Studios,ProductionLocations,Chapters,ExternalUrls,ImageTags,BackdropImageTags,PrimaryImageAspectRatio,Container,HasSubtitles,IsHD,VideoType,Width,Height,SortName,Tags,DateCreated,Path,ParentId",
                "Limit": limit,
                "StartIndex": start_index
            }
            
            if self.settings.JELLYFIN_USER_ID:
                params["UserId"] = self.settings.JELLYFIN_USER_ID
            
            response = await self.client.get(url, headers=headers, params=params, timeout=60.0)
            response.raise_for_status()
            
            data = response.json()
            return data.get("Items", [])
            
        except Exception as e:
            logger.error("Failed to get movies from Jellyfin", 
                        start_index=start_index, limit=limit, error=str(e))
            raise

    async def get_movies_chunked(self, total_limit: int = 1000, chunk_size: int = 100) -> List[Dict[str, Any]]:
        """Get movies from Jellyfin using chunked requests (like your implementation)."""
        all_items = []
        total_chunks = (total_limit + chunk_size - 1) // chunk_size
        
        logger.info("Starting chunked movie fetch", 
                   total_limit=total_limit, chunk_size=chunk_size, total_chunks=total_chunks)
        
        for chunk_index in range(total_chunks):
            start_index = chunk_index * chunk_size
            current_limit = min(chunk_size, total_limit - start_index)
            
            try:
                chunk_items = await self.get_movies(limit=current_limit, start_index=start_index)
                all_items.extend(chunk_items)
                
                logger.info("Fetched chunk", 
                           chunk=f"{chunk_index + 1}/{total_chunks}",
                           items_in_chunk=len(chunk_items),
                           total_items=len(all_items))
                
                # Small delay to avoid overwhelming Jellyfin
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Chunk failed, continuing", 
                           chunk=chunk_index + 1, start_index=start_index, error=str(e))
                continue
        
        logger.info("Chunked fetch complete", total_items=len(all_items))
        return all_items
    
    def _extract_movie_data(self, jellyfin_item: Dict[str, Any]) -> MovieCreate:
        """Extract enhanced movie data from Jellyfin item with ASCII sanitization."""
        # Core identification (Jellyfin field names)
        name = to_ascii(jellyfin_item.get("Name", "Unknown"))
        original_title = to_ascii(jellyfin_item.get("OriginalTitle")) if jellyfin_item.get("OriginalTitle") != name else None
        sort_name = to_ascii(jellyfin_item.get("SortName"))
        production_year = jellyfin_item.get("ProductionYear")
        premiere_date = jellyfin_item.get("PremiereDate")
        
        # Content
        overview = to_ascii(jellyfin_item.get("Overview", ""))
        
        # Handle taglines (it's a list in Jellyfin)
        taglines = []
        taglines_list = jellyfin_item.get("Taglines", [])
        for tagline_item in taglines_list:
            if isinstance(tagline_item, str):
                tagline_text = to_ascii(tagline_item)
            else:
                tagline_text = to_ascii(tagline_item.get("Text", ""))
            if tagline_text:
                taglines.append(tagline_text)
        
        # Extract people with full structure
        people = []
        people_list = jellyfin_item.get("People", [])
        
        for person in people_list:
            person_name = to_ascii(person.get("Name"))
            if person_name:
                people.append(Person(
                    name=person_name,
                    id=person.get("Id"),
                    role=to_ascii(person.get("Role")),
                    type=person.get("Type", ""),
                    primary_image_tag=person.get("PrimaryImageTag")
                ))
        
        # Classification & ratings
        genres = ascii_list(jellyfin_item.get("Genres"))
        official_rating = to_ascii(jellyfin_item.get("OfficialRating"))
        community_rating = jellyfin_item.get("CommunityRating")
        critic_rating = jellyfin_item.get("CriticRating")
        run_time_ticks = jellyfin_item.get("RunTimeTicks")
        
        # Location & Production
        production_locations = ascii_list(jellyfin_item.get("ProductionLocations"))
        
        # Studios
        studios = []
        studios_list = jellyfin_item.get("Studios", [])
        for studio in studios_list:
            studios.append(Studio(
                name=to_ascii(studio.get("Name", "")),
                id=studio.get("Id")
            ))
        
        # Technical details
        container = jellyfin_item.get("Container")
        media_streams = jellyfin_item.get("MediaStreams", [])
        has_subtitles = jellyfin_item.get("HasSubtitles")
        is_hd = jellyfin_item.get("IsHD")
        video_type = jellyfin_item.get("VideoType")
        width = jellyfin_item.get("Width")
        height = jellyfin_item.get("Height")
        
        # External references
        provider_ids = jellyfin_item.get("ProviderIds", {})
        
        external_urls = []
        external_urls_list = jellyfin_item.get("ExternalUrls", [])
        for url in external_urls_list:
            external_urls.append(ExternalUrl(
                name=url.get("Name", ""),
                url=url.get("Url", "")
            ))
        
        # Chapters
        chapters = []
        chapters_list = jellyfin_item.get("Chapters", [])
        for chapter in chapters_list:
            chapters.append(Chapter(
                start_position_ticks=chapter.get("StartPositionTicks", 0),
                name=to_ascii(chapter.get("Name", "")),
                image_tag=chapter.get("ImageTag")
            ))
        
        tags = ascii_list(jellyfin_item.get("Tags"))
        
        # Image information
        image_tags = jellyfin_item.get("ImageTags", {})
        backdrop_image_tags = jellyfin_item.get("BackdropImageTags", [])
        primary_image_aspect_ratio = jellyfin_item.get("PrimaryImageAspectRatio")
        
        # Jellyfin metadata
        jellyfin_id = jellyfin_item.get("Id")
        server_id = jellyfin_item.get("ServerId")
        etag = jellyfin_item.get("Etag")
        date_created = jellyfin_item.get("DateCreated")
        path = jellyfin_item.get("Path")
        parent_id = jellyfin_item.get("ParentId")
        
        # Generate search keywords
        search_keywords = []
        if name:
            search_keywords.extend(name.lower().split())
        if original_title and original_title != name:
            search_keywords.extend(original_title.lower().split())
        search_keywords.extend([genre.lower() for genre in genres])
        # Add cast and crew names
        for person in people[:10]:  # Top 10 people
            search_keywords.extend(person.name.lower().split())
        
        # Remove duplicates and empty strings
        search_keywords = list(set([kw for kw in search_keywords if kw and len(kw) > 2]))
        
        return MovieCreate(
            name=name,
            original_title=original_title,
            production_year=production_year,
            overview=overview,
            taglines=taglines,
            people=people,
            genres=genres,
            official_rating=official_rating,
            community_rating=community_rating,
            critic_rating=critic_rating,
            run_time_ticks=run_time_ticks,
            media_streams=media_streams,
            provider_ids=provider_ids,
            jellyfin_id=jellyfin_id,
            search_keywords=search_keywords
        )
    
    async def _enhance_movie_data(self, movie_data: MovieCreate) -> MovieCreate:
        """Enhance movie data using plugins."""
        try:
            # Import plugin system
            from ..plugins.examples.movie_summary_enhancer import MovieSummaryEnhancerPlugin
            from ..plugins.base import PluginExecutionContext
            
            # Create plugin instance
            plugin = MovieSummaryEnhancerPlugin()
            await plugin.initialize({})
            
            # Create execution context
            context = PluginExecutionContext(
                request_id=f"ingest_{movie_data.jellyfin_id}",
                user_id="system"
            )
            
            # Convert MovieCreate to dict for plugin processing
            movie_dict = movie_data.dict()
            
            # Enhance with plugin
            enhanced_dict = await plugin.embellish_embed_data(movie_dict, context)
            
            # Create new MovieCreate with enhanced data
            enhanced_movie_data = MovieCreate(**enhanced_dict)
            
            logger.info("Enhanced movie data with plugins", 
                       movie=movie_data.name,
                       has_enhanced_summary=bool(enhanced_dict.get("enhanced_fields", {}).get("summary")))
            
            return enhanced_movie_data
            
        except Exception as e:
            logger.warning("Failed to enhance movie data with plugins, using original", 
                          movie=movie_data.name,
                          error=str(e))
            # Return original data if plugin enhancement fails
            return movie_data
    
    async def ingest_movie(self, jellyfin_item: Dict[str, Any]) -> Optional[Movie]:
        """Ingest a single movie from Jellyfin (create or update)."""
        try:
            jellyfin_id = jellyfin_item.get("Id")
            if not jellyfin_id:
                logger.warning("Movie has no Jellyfin ID", item=jellyfin_item.get("Name"))
                return None
            
            # Extract movie data from Jellyfin
            movie_data = self._extract_movie_data(jellyfin_item)
            
            # Enhance movie data with plugins
            enhanced_movie_data = await self._enhance_movie_data(movie_data)
            
            # Check if movie already exists
            existing_movie = await self.mongo_client.get_movie_by_jellyfin_id(jellyfin_id)
            if existing_movie:
                # Update existing movie with fresh data
                from ..data.models import MovieUpdate
                update_data = MovieUpdate(**enhanced_movie_data.dict(exclude={"jellyfin_id"}))
                updated_movie = await self.mongo_client.update_movie(str(existing_movie.id), update_data)
                
                logger.info("Updated movie", name=updated_movie.name, year=updated_movie.production_year)
                return updated_movie
            else:
                # Create new movie
                movie = await self.mongo_client.create_movie(enhanced_movie_data)
                logger.info("Created movie", name=movie.name, year=movie.production_year)
                return movie
            
        except Exception as e:
            logger.error("Failed to ingest movie", 
                        title=jellyfin_item.get("Name", "Unknown"),
                        error=str(e))
            return None
    
    async def ingest_movies_batch(self, limit: int = 100, start_index: int = 0) -> List[Movie]:
        """Ingest a batch of movies from Jellyfin."""
        try:
            jellyfin_movies = await self.get_movies(limit=limit, start_index=start_index)
            ingested_movies = []
            
            for jellyfin_item in jellyfin_movies:
                movie = await self.ingest_movie(jellyfin_item)
                if movie:
                    ingested_movies.append(movie)
            
            logger.info("Ingested movie batch", 
                       count=len(ingested_movies),
                       total_items=len(jellyfin_movies))
            
            return ingested_movies
            
        except Exception as e:
            logger.error("Failed to ingest movie batch", error=str(e))
            raise
    
    async def ingest_all_movies(self, batch_size: int = 100) -> int:
        """Ingest all movies from Jellyfin."""
        total_ingested = 0
        start_index = 0
        
        while True:
            movies = await self.ingest_movies_batch(limit=batch_size, start_index=start_index)
            
            if not movies:
                break
                
            total_ingested += len(movies)
            start_index += batch_size
            
            # Add small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        logger.info("Completed full ingestion", total_movies=total_ingested)
        return total_ingested
    
    async def sync_movie_updates(self) -> int:
        """Sync updates for existing movies."""
        updated_count = 0
        
        # Get all movies from MongoDB
        mongo_movies = await self.mongo_client.list_movies(limit=1000)
        
        for movie in mongo_movies:
            if not movie.jellyfin_id:
                continue
                
            try:
                # Get fresh data from Jellyfin
                jellyfin_movies = await self.get_movies(limit=1)
                # In a real implementation, you'd filter by the specific movie ID
                # For now, this is a simplified version
                
                logger.debug("Checked movie for updates", name=movie.name)
                
            except Exception as e:
                logger.error("Failed to sync movie", 
                           name=movie.name,
                           error=str(e))
        
        logger.info("Completed sync", updated_count=updated_count)
        return updated_count