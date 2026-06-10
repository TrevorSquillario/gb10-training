#!/usr/bin/env python
"""Main MCP server implementation for Radarr/Sonarr, async style like `jellyfin.py`."""

import os
import json
import logging
from typing import Optional
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

from fastmcp import FastMCP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent / ".env")

# Load configuration
def load_config():
    if os.environ.get('RADARR_API_KEY') or os.environ.get('SONARR_API_KEY'):
        logger.info("Loading configuration from environment variables...")
        nas_ip = os.environ.get('NAS_IP', '10.0.0.23')
        return {
            "nasConfig": {"ip": nas_ip, "port": os.environ.get('RADARR_PORT', '7878')},
            "radarrConfig": {
                "apiKey": os.environ.get('RADARR_API_KEY', ''),
                "basePath": os.environ.get('RADARR_BASE_PATH', '/api/v3'),
                "port": os.environ.get('RADARR_PORT', '7878'),
                "defaultQualityProfileId": int(os.environ.get('RADARR_DEFAULT_QUALITY_PROFILE_ID')) if os.environ.get('RADARR_DEFAULT_QUALITY_PROFILE_ID') else None,
                "rootFolderPath": os.environ.get('RADARR_ROOT_FOLDER_PATH', '/movies')
            },
            "sonarrConfig": {
                "apiKey": os.environ.get('SONARR_API_KEY', ''),
                "basePath": os.environ.get('SONARR_BASE_PATH', '/api/v3'),
                "port": os.environ.get('SONARR_PORT', '8989'),
                "defaultQualityProfileId": int(os.environ.get('SONARR_DEFAULT_QUALITY_PROFILE_ID')) if os.environ.get('SONARR_DEFAULT_QUALITY_PROFILE_ID') else None,
                "rootFolderPath": os.environ.get('SONARR_ROOT_FOLDER_PATH', '/tv')
            },
            "server": {"port": int(os.environ.get('MCP_SERVER_PORT', '3000'))}
        }
    else:
        config_path = 'config.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
            return {
                "nasConfig": {"ip": "10.0.0.23", "port": "7878"},
                "radarrConfig": {"apiKey": "", "basePath": "/api/v3", "port": "7878", "defaultQualityProfileId": None, "rootFolderPath": "/movies"},
                "sonarrConfig": {"apiKey": "", "basePath": "/api/v3", "port": "8989", "defaultQualityProfileId": None, "rootFolderPath": "/tv"},
                "server": {"port": 3000}
            }


# Small helpers to build service config objects expected by service constructors
def _radarr_ns(config: dict) -> SimpleNamespace:
    nas_ip = config["nasConfig"]["ip"]
    port = config["radarrConfig"]["port"]
    base_path = config["radarrConfig"]["basePath"]
    base_url = f"http://{nas_ip}:{port}{base_path}"
    return SimpleNamespace(api_key=config["radarrConfig"].get("apiKey", ""), base_url=base_url)


def _sonarr_ns(config: dict) -> SimpleNamespace:
    nas_ip = config["nasConfig"]["ip"]
    port = config["sonarrConfig"]["port"]
    base_path = config["sonarrConfig"]["basePath"]
    base_url = f"http://{nas_ip}:{port}{base_path}"
    return SimpleNamespace(api_key=config["sonarrConfig"].get("apiKey", ""), base_url=base_url)


# Instantiate MCP server (module-level so FastMCP can discover it)
config = load_config()
mcp = FastMCP("radarr-sonarr")


# Sonarr Tools
@mcp.tool()
async def get_available_series(year: Optional[int] = None,
                               downloaded: Optional[bool] = None,
                               watched: Optional[bool] = None,
                               actors: Optional[str] = None) -> dict:
    from services.sonarr import SonarrService

    svc = SonarrService(_sonarr_ns(config))
    all_series = await svc.get_all_series()
    filtered_series = all_series

    if year is not None:
        filtered_series = [s for s in filtered_series if s.year == year]

    if downloaded is not None:
        filtered_series = [
            s for s in filtered_series
            if (s.statistics and s.statistics.episode_file_count > 0) == downloaded
        ]

    if actors:
        filtered_series = [
            s for s in filtered_series
            if s.data.get("credits") and any(
                actors.lower() in cast.get("name", "").lower()
                for cast in s.data.get("credits", {}).get("cast", [])
            )
        ]

    return {
        "count": len(filtered_series),
        "series": [
            {
                "id": s.id,
                "title": s.title,
                "year": s.year,
                "overview": s.overview,
                "status": s.status,
                "network": s.network,
                "genres": s.genres,
            }
            for s in filtered_series
        ]
    }


@mcp.tool()
async def lookup_series(term: str) -> dict:
    from services.sonarr import SonarrService

    svc = SonarrService(_sonarr_ns(config))
    results = await svc.lookup_series(term)
    top_5 = results[:5]
    return {
        "count": len(results),
        "series": [
            {
                "id": s.id,
                "tvdb_id": s.tvdb_id,
                "title": s.title,
                "year": s.year,
                "overview": s.overview,
                "status": s.status,
                "network": s.network,
                "monitored": s.monitored
            }
            for s in top_5
        ]
    }

@mcp.tool()
async def add_series(tvdb_id: str, quality_profile_id: Optional[int] = None, root_dir_path: Optional[str] = None,
                     language_profile_id: int = 1, season_folder: bool = True,
                     monitor: bool = True, search_for_missing: bool = True) -> dict:
    """Add a TV series to Sonarr by TVDB id.

    Parameters:
    - tvdb_id (str): The TVDB id for the series to add (passed to Sonarr).
    - quality_profile_id (Optional[int]): Sonarr quality profile id to assign.
        If omitted the configured default `sonarrConfig.defaultQualityProfileId`
        will be used. If neither is available a `ValueError` is raised.
    - root_dir_path (Optional[str]): Filesystem root path where the series
        should be stored. Falls back to `sonarrConfig.rootFolderPath` from config
        (default '/tv').
    - language_profile_id (int): Sonarr language profile id (defaults to 1).
    - season_folder (bool): Whether to create a season subfolder for the
        series (defaults to True).
    - monitor (bool): Whether Sonarr should monitor the series for new
        episodes (defaults to True).
    - search_for_missing (bool): If True, Sonarr will perform an initial
        search to locate existing episodes after adding the series (defaults to True).
    """
    from services.sonarr import SonarrService

    svc = SonarrService(_sonarr_ns(config))
    # Resolve quality profile id and root folder from provided args or config defaults
    qid = quality_profile_id if quality_profile_id is not None else config.get('sonarrConfig', {}).get('defaultQualityProfileId')
    root = root_dir_path if root_dir_path is not None else config.get('sonarrConfig', {}).get('rootFolderPath', '/tv')

    if qid is None:
        raise ValueError('No quality_profile_id provided and no default configured for Sonarr')

    result = await svc.add_series(tvdb_id=tvdb_id,
                                  quality_profile_id=int(qid),
                                  root_dir_path=root,
                                  language_profile_id=language_profile_id,
                                  season_folder=season_folder,
                                  monitor=monitor,
                                  search_for_missing=search_for_missing)
    return {"result": result}

@mcp.tool()
async def get_series_quality_profiles(default_profile_id: Optional[int] = None) -> dict:
    """Return simplified quality profiles from Sonarr."""
    from services.sonarr import SonarrService

    svc = SonarrService(_sonarr_ns(config))
    profiles = await svc.get_quality_profiles(default_profile_id)
    return {"count": len(profiles), "profiles": profiles}

# Radarr Tools
@mcp.tool()
async def lookup_movie(term: str) -> dict:
    from services.radarr import RadarrService

    svc = RadarrService(_radarr_ns(config))
    results = await svc.lookup_movie(term)
    top_5 = results[:5]
    return {
        "count": len(results),
        "movies": [
            {
                "id": m.id,
                "tmdb_id": m.tmdb_id,
                "title": m.title,
                "year": m.year,
                "overview": m.overview,
                "status": m.status,
                "monitored": m.monitored
            }
            for m in top_5
        ]
    }
    
@mcp.tool()
async def get_available_movies(year: Optional[int] = None,
                               downloaded: Optional[bool] = None,
                               watched: Optional[bool] = None,
                               actors: Optional[str] = None) -> dict:
    from services.radarr import RadarrService

    svc = RadarrService(_radarr_ns(config))
    all_movies = await svc.get_all_movies()
    filtered_movies = all_movies

    if year is not None:
        filtered_movies = [m for m in filtered_movies if m.get("year") == year]

    if downloaded is not None:
        filtered_movies = [m for m in filtered_movies if m.get("hasFile") == downloaded]

    if actors:
        filtered_movies = [
            m for m in filtered_movies
            if m.get("credits") and any(
                actors.lower() in cast.get("name", "").lower()
                for cast in m.get("credits", {}).get("cast", [])
            )
        ]

    return {
        "count": len(filtered_movies),
        "movies": [
            {
                "id": m.get("id"),
                "title": m.get("title"),
                "year": m.get("year"),
                "overview": m.get("overview"),
                "hasFile": m.get("hasFile"),
                "status": m.get("status"),
                "genres": m.get("genres", []),
            }
            for m in filtered_movies
        ]
    }

@mcp.tool()
async def get_movies_quality_profiles(default_profile_id: Optional[int] = None) -> dict:
    """Return simplified quality profiles from Radarr."""
    from services.radarr import RadarrService

    svc = RadarrService(_radarr_ns(config))
    profiles = await svc.get_quality_profiles(default_profile_id)
    return {"count": len(profiles), "profiles": profiles}


@mcp.tool()
async def add_movie(tmdb_id: int, quality_profile_id: Optional[int] = None, root_dir_path: Optional[str] = None,
                    monitor: bool = True, search_for_movie: bool = True) -> dict:
    """Add a movie to Radarr by TMDb id.

    Parameters:
    - tmdb_id (int): The TMDb id of the movie to add.
    - quality_profile_id (Optional[int]): Radarr quality profile id to assign.
        If omitted the configured default `radarrConfig.defaultQualityProfileId`
        will be used. If neither is available a `ValueError` is raised.
    - root_dir_path (Optional[str]): Filesystem root path where the movie
        should be stored. Falls back to `radarrConfig.rootFolderPath` from config
        (default '/movies').
    - monitor (bool): Whether Radarr should monitor the movie for upgrades
        and availability (defaults to True).
    - search_for_movie (bool): If True, Radarr will perform an immediate
        search for the movie after adding (defaults to True).
    """
    from services.radarr import RadarrService

    svc = RadarrService(_radarr_ns(config))
    # Resolve quality profile id and root folder from provided args or config defaults
    qid = quality_profile_id if quality_profile_id is not None else config.get('radarrConfig', {}).get('defaultQualityProfileId')
    root = root_dir_path if root_dir_path is not None else config.get('radarrConfig', {}).get('rootFolderPath', '/movies')

    if qid is None:
        raise ValueError('No quality_profile_id provided and no default configured for Radarr')

    result = await svc.add_movie(tmdb_id=tmdb_id,
                                 quality_profile_id=int(qid),
                                 root_dir_path=root,
                                 monitor=monitor,
                                 search_for_movie=search_for_movie)
    return {"result": result}


if __name__ == "__main__":
    mcp.run(transport="stdio")