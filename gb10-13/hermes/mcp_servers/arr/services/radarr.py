"""Service for interacting with Radarr API."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import httpx
from models import RadarrConfig


@dataclass
class Movie:
    """Movie data class."""
    id: Optional[int]
    tmdb_id: Optional[int]
    title: str
    year: Optional[int]
    overview: str
    status: str
    tags: List[int]
    genres: List[str]
    monitored: bool
    data: Dict[str, Any]  # Store original data for reference
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Movie':
        """Create a Movie object from a dictionary."""
        tmdb = data.get('tmdbId')
        movie_id = data.get('id', None)

        return cls(
            id=movie_id,
            tmdb_id=tmdb,
            title=(data.get('title') or ''),
            year=data.get('year'),
            overview=(data.get('overview') or ''),
            status=(data.get('status') or ''),
            tags=data.get('tags', []),
            genres=data.get('genres', []),
            monitored=data.get('monitored', False) if movie_id is not None else False,
            data=data
        )


class RadarrService:
    """Service for interacting with Radarr API."""
    
    def __init__(self, config: RadarrConfig):
        """Initialize the Radarr service with configuration."""
        self.config = config
        self.api_key = config.api_key
        # Use header for API key as required by Radarr
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=30.0,
            headers={"X-Api-Key": self.api_key},
        )
    
    async def get_all_movies(self) -> List[Movie]:
        """Fetch all movies from Radarr."""
        try:
            response = await self._client.get("/movie")
            response.raise_for_status()

            movies = []
            for movie_data in response.json():
                movies.append(Movie.from_dict(movie_data))

            return movies
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error fetching movies from Radarr: {e}")
            raise Exception(f"Failed to fetch movies from Radarr: {e}")
    
    async def lookup_movie(self, term: str, tmdb_id: Optional[int] = None) -> List[Movie]:
        """Look up movies by search term or TMDb ID.

        If `tmdb_id` is provided, it will be used instead of `term` and formatted
        as `tmdb:<id>` per Radarr's lookup API.
        """
        try:
            lookup_params = {"term": term}
            if tmdb_id is not None:
                lookup_params = {"term": f"tmdb:{tmdb_id}"}

            response = await self._client.get(
                "/movie/lookup",
                params=lookup_params
            )
            response.raise_for_status()

            movies = []
            for movie_data in response.json():
                movies.append(Movie.from_dict(movie_data))

            return movies
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error looking up movie in Radarr: {e}")
            raise Exception(f"Failed to lookup movie in Radarr: {e}")
    
    async def get_movie_file(self, movie_id: int) -> Dict[str, Any]:
        """Get the file information for a movie."""
        try:
            response = await self._client.get(
                "/moviefile",
                params={"movieId": movie_id}
            )
            response.raise_for_status()

            return response.json()
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error fetching movie file for ID {movie_id}: {e}")
            raise Exception(f"Failed to fetch movie file: {e}")

    async def get_quality_profiles(self, default_profile_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch all quality profiles from Radarr and return simplified profiles.

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
                logging.warning("No quality profiles data received from Radarr or data is not a list.")
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
            logging.error(f"Error fetching quality profiles from Radarr: {e}")
            raise Exception(f"Failed to fetch quality profiles: {e}")

    async def add_movie(self, tmdb_id: int, quality_profile_id: int, root_dir_path: str = "/movies",
                        monitor: bool = True, search_for_movie: bool = True) -> Dict[str, Any]:
        """Add a movie to Radarr using a TMDb ID.

        This follows the module's async/httpx style: it looks up the movie via the
        Radarr lookup endpoint and sends a POST to the `/movie` endpoint with the
        assembled payload.
        """
        try:
            # Lookup movie details by TMDb id using Radarr's lookup endpoint
            lookup_resp = await self._client.get(
                "/movie/lookup",
                params={"term": f"tmdb:{tmdb_id}"}
            )
            lookup_resp.raise_for_status()

            lookup_data = lookup_resp.json()
            if not lookup_data:
                raise Exception(f"No lookup results for tmdb id {tmdb_id}")

            # Use the first lookup result
            movie_info = lookup_data[0] if isinstance(lookup_data, list) else lookup_data

            data = {
                **movie_info,
                "qualityProfileId": quality_profile_id,
                "rootFolderPath": root_dir_path,
                "monitored": monitor,
                "addOptions": {"searchForMovie": search_for_movie},
            }

            import json, logging
            logging.info(f"Radarr add movie request: {json.dumps(data, indent=2)}")

            response = await self._client.post(
                "/movie",
                json=data
            )
            response.raise_for_status()

            return response.json()
        except httpx.RequestError as e:
            import logging
            logging.error(f"Error adding movie to Radarr (tmdb id {tmdb_id}): {e}")
            raise Exception(f"Failed to add movie to Radarr: {e}")
