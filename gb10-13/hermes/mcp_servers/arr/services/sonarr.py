"""Service for interacting with Sonarr API."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import httpx
import json
import logging
from models import SonarrConfig


@dataclass
class Statistics:
    """Statistics for a TV series."""
    episode_file_count: int
    episode_count: int
    total_episode_count: int
    size_on_disk: int
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Statistics':
        """Create a Statistics object from a dictionary."""
        return cls(
            episode_file_count=data.get('episodeFileCount', 0),
            episode_count=data.get('episodeCount', 0),
            total_episode_count=data.get('totalEpisodeCount', 0),
            size_on_disk=data.get('sizeOnDisk', 0)
        )


@dataclass
class Series:
    """TV Series data class."""
    id: Optional[int]
    title: str
    year: Optional[int]
    overview: str
    status: str
    network: str
    tags: List[int]
    genres: List[str]
    tvdb_id: Optional[int]
    tmdb_id: Optional[int]
    statistics: Optional[Statistics]
    monitored: bool
    data: Dict[str, Any]  # Store original data for reference
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Series':
        """Create a Series object from a dictionary."""
        statistics = None
        if data.get('statistics'):
            statistics = Statistics.from_dict(data['statistics'])

        tvdb = data.get('tvdbId')
        tmdb = data.get('tmdbId')
        series_id = data.get('id', None)

        return cls(
            id=series_id,
            title=(data.get('title') or ''),
            year=data.get('year'),
            overview=(data.get('overview') or ''),
            status=(data.get('status') or ''),
            network=(data.get('network') or ''),
            tags=data.get('tags', []),
            genres=data.get('genres', []),
            tvdb_id=tvdb,
            tmdb_id=tmdb,
            statistics=statistics,
            monitored=data.get('monitored', False) if series_id is not None else False,
            data=data
        )


@dataclass
class Episode:
    """TV Episode data class."""
    id: int
    series_id: int
    episode_file_id: Optional[int]
    season_number: int
    episode_number: int
    title: str
    air_date: Optional[str]
    has_file: bool
    monitored: bool
    overview: str
    data: Dict[str, Any]  # Store original data for reference
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create an Episode object from a dictionary."""
        return cls(
            id=data['id'],
            series_id=data['seriesId'],
            episode_file_id=data.get('episodeFileId'),
            season_number=data['seasonNumber'],
            episode_number=data['episodeNumber'],
            title=data.get('title', ''),
            air_date=data.get('airDate'),
            has_file=data.get('hasFile', False),
            monitored=data.get('monitored', True),
            overview=data.get('overview', ''),
            data=data
        )


class SonarrService:
    """Service for interacting with Sonarr API."""
    
    def __init__(self, config: SonarrConfig):
        """Initialize the Sonarr service with configuration."""
        self.config = config
        self.api_key = config.api_key
        self.logger = logging.getLogger(__name__)
        # Use header for API key as required by Sonarr
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=30.0,
            headers={"X-Api-Key": self.api_key},
        )
    
    async def get_all_series(self) -> List[Series]:
        """Fetch all TV series from Sonarr."""
        try:
            response = await self._client.get("/series")
            response.raise_for_status()

            series_list = []
            for series_data in response.json():
                series_list.append(Series.from_dict(series_data))

            return series_list
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error fetching series from Sonarr: {e}")
            raise Exception(f"Failed to fetch seriesNs from Sonarr: {e}")
    
    async def get_episodes(self, series_id: int) -> List[Episode]:
        """Fetch episodes for a TV series."""
        try:
            response = await self._client.get(
                "/episode",
                params={"seriesId": series_id}
            )
            response.raise_for_status()

            episodes = []
            for episode_data in response.json():
                episodes.append(Episode.from_dict(episode_data))

            return episodes
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error fetching episodes for series ID {series_id}: {e}")
            raise Exception(f"Failed to fetch episodes: {e}")

    async def add_series(self, tvdb_id: str, quality_profile_id: int, root_dir_path: str = "/tv", 
                  language_profile_id: int = 1, season_folder: bool = True, 
                  monitor: bool = True, search_for_missing: bool = True) -> Any:
        """Add a series to Sonarr using TMDB ID.

        Args:
            tvdb_id (str): The TMDB ID of the series to add
            quality_profile_id (int): The ID of the quality profile to use
            root_dir_path (str): The Root Directory path in Sonarr. Defaults to "/tv"
            language_profile_id (int, optional): The ID of the language profile. Defaults to 1=English
            season_folder (bool, optional): Whether to create season folders. Defaults to True
            monitor (bool, optional): Defaults to True
            search_for_missing (bool, optional): Whether to search for missing episodes. Defaults to True
            
        Returns:
            Any: The API response object from Sonarr (propagated from `_make_request`).
        """
        # Lookup series by TMDB id using the lookup_series helper. If the
        # lookup returns a list, use the first result's original data dict.
        results = await self.lookup_series("", tvdb_id=tvdb_id)
        if not results:
            raise Exception(f"No lookup results for tmdb id {tvdb_id}")

        series_obj = results[0] if isinstance(results, list) else results
        series_data = series_obj.data if hasattr(series_obj, 'data') else series_obj

        data = {
            **series_data,
            "title": series_data.get("title", ""),
            "tmdbId": tvdb_id,
            "qualityProfileId": quality_profile_id,
            "rootFolderPath": root_dir_path,
            "languageProfileId": language_profile_id,
            "seasonFolder": season_folder,
            "monitored": monitor,
            "addOptions": {
                "searchForMissingEpisodes": search_for_missing,
            },
            "titleSlug": series_data.get("titleSlug"),
            "images": series_data.get("images"),
            "seasons": series_data.get("seasons"),
        }

        self.logger.info(f"Sonarr request: {json.dumps(data, indent=2)}")
        return await self._make_request("POST", "series", data=data)

    async def lookup_series(self, term: str, tvdb_id: Optional[str] = None) -> List[Series]:
        """Look up series by search term or TVDb ID.

        If `tvdb_id` is provided it will be used instead of `term` and formatted
        as `tvdbid:<id>` to match Sonarr's lookup API.
        """
        try:
            lookup_params = {"term": term}
            if tvdb_id is not None:
                lookup_params = {"term": f"tvdbid:{tvdb_id}"}

            response = await self._client.get(
                "/series/lookup",
                params=lookup_params
            )
            response.raise_for_status()

            results: List[Series] = []
            body = response.json()
            if not body:
                return []

            for s in body:
                results.append(Series.from_dict(s))

            return results
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error looking up series in Sonarr: {e}")
            raise Exception(f"Failed to lookup series in Sonarr: {e}")

    async def get_quality_profiles(self, default_profile_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch all quality profiles from Sonarr and return simplified profiles.

        Each simplified profile contains:
            - id: profile ID
            - name: profile name
            - allowed_qualities: list of allowed quality names
            - is_default: whether this profile matches `default_profile_id`
        """
        try:
            response = await self._client.get("/qualityprofile")
            response.raise_for_status()

            data = response.json()
            if not data or not isinstance(data, list):
                import logging
                logging.warning("No quality profiles data received or data is not a list.")
                return []

            simplified_profiles: List[Dict[str, Any]] = []
            for profile in data:
                allowed_qualities: List[str] = []
                for item in profile.get('items', []):
                    if item.get('allowed', False):
                        quality = item.get('quality') or {}
                        quality_name = (
                            quality.get('name')
                            or item.get('name')
                            or (f"Resolution {quality.get('resolution')}p" if quality.get('resolution') is not None else "Unknown")
                        )
                        allowed_qualities.append(quality_name)

                simplified_profiles.append({
                    'id': profile.get('id'),
                    'name': profile.get('name'),
                    'allowed_qualities': allowed_qualities,
                    'is_default': (profile.get('id') == default_profile_id) if default_profile_id is not None else False
                })

            return simplified_profiles
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error fetching quality profiles from Sonarr: {e}")
            raise Exception(f"Failed to fetch quality profiles: {e}")

