"""Jellyfin MCP Server — curated tools for managing Jellyfin from Claude Code."""

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

JELLYFIN_URL = os.environ.get("JELLYFIN_URL", "").rstrip("/")
JELLYFIN_API_KEY = os.environ.get("JELLYFIN_API_KEY", "")
JELLYFIN_USERNAME = os.environ.get("JELLYFIN_USERNAME", "")

if not JELLYFIN_URL or not JELLYFIN_API_KEY or not JELLYFIN_USERNAME:
    raise RuntimeError(
        "JELLYFIN_URL, JELLYFIN_API_KEY, and JELLYFIN_USERNAME environment variables must be set"
    )

AUTH_HEADER = (
    f'MediaBrowser Token="{JELLYFIN_API_KEY}", '
    f'Client="JellyfinMCP", Version="1.0.0", '
    f'DeviceId="jellyfin-mcp", Device="Jellyfin MCP Server"'
)

mcp = FastMCP("jellyfin")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=JELLYFIN_URL,
            headers={"Authorization": AUTH_HEADER},
            timeout=30.0,
        )
    return _client


async def _get(path: str, params: dict | None = None) -> Any:
    r = await _get_client().get(path, params=params)
    r.raise_for_status()
    return r.json()


async def _post(path: str, json: dict | None = None, params: dict | None = None) -> Any:
    r = await _get_client().post(path, json=json, params=params)
    r.raise_for_status()
    if r.status_code == 204 or not r.content:
        return None
    return r.json()


async def _delete(path: str, params: dict | None = None) -> None:
    r = await _get_client().delete(path, params=params)
    r.raise_for_status()

# ---------------------------------------------------------------------------
# User ID (cached, lazy)
# ---------------------------------------------------------------------------

_user_id: str | None = None


async def _get_user_id(username: str | None = None) -> str:
    """Resolve a username to a user ID.

    If no username is provided, uses the configured JELLYFIN_USERNAME.
    API-key auth doesn't support /Users/Me, so we look up the user by name.
    The result is cached for the lifetime of the process.
    """
    global _user_id
    if _user_id is None:
        target_username = username or JELLYFIN_USERNAME
        users = await _get("/Users")
        match = next(
            (u for u in users if u.get("Name", "").lower() == target_username.lower()),
            None,
        )
        if not match:
            available = [u.get("Name", "") for u in users]
            raise RuntimeError(
                f"User '{target_username}' not found on the Jellyfin server. "
                f"Available users: {available}"
            )
        _user_id = match["Id"]
    return _user_id


# ---------------------------------------------------------------------------
# Item summarizer
# ---------------------------------------------------------------------------


def _summarize_item(item: dict) -> dict:
    """Extract only the LLM-relevant fields from a Jellyfin item."""
    summary: dict[str, Any] = {"id": item["Id"], "name": item.get("Name", "")}

    if t := item.get("Type"):
        summary["type"] = t
    if y := item.get("ProductionYear"):
        summary["year"] = y
    if artists := item.get("Artists"):
        summary["artists"] = artists
    elif artist := item.get("AlbumArtist"):
        summary["artist"] = artist
    if album := item.get("Album"):
        summary["album"] = album
    if series := item.get("SeriesName"):
        summary["series"] = series
    if sid := item.get("SeriesId"):
        summary["series_id"] = sid
    if pd := item.get("PremiereDate"):
        summary["premiere_date"] = pd
    if lp := item.get("UserData", {}).get("LastPlayedDate"):
        summary["last_played_date"] = lp
    if sn := item.get("ParentIndexNumber"):
        summary["season"] = sn
    if idx := item.get("IndexNumber"):
        if item.get("Type") == "Audio":
            summary["track"] = idx
        else:
            summary["episode"] = idx
    if ticks := item.get("RunTimeTicks"):
        summary["duration_seconds"] = round(ticks / 10_000_000)
    if overview := item.get("Overview"):
        summary["overview"] = overview[:300] + "..." if len(overview) > 300 else overview
    if da := item.get("DateCreated"):
        summary["date_added"] = da
    if (count := item.get("ChildCount")) and item.get("Type") != "Playlist":
        summary["child_count"] = count
    if genres := item.get("Genres"):
        summary["genres"] = genres
    if tags := item.get("Tags"):
        summary["tags"] = tags
    if locations := item.get("ProductionLocations"):
        summary["production_locations"] = locations
    if prov_ids := item.get("ProviderIds"):
        summary["provider_ids"] = prov_ids
    if ss := item.get("SeriesStudio"):
        summary["series_studio"] = ss
    if studios := item.get("Studios"):
        summary["studios"] = [s["Name"] for s in studios]
    if people := item.get("People"):
        summary["people"] = [
            {
                "name": p.get("Name", ""),
                "role": p.get("Role", ""),
                "type": p.get("Type", ""),
            }
            for p in people
            if p.get("Type") in ("Actor", "Writer", "Director")
        ]

    return summary


def _summarize_items(items: list[dict]) -> list[dict]:
    summary = []
    for it in items:
        summary.append(_summarize_item(it))
    return summary

# ---------------------------------------------------------------------------
# Tools — Library Rescans
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_users() -> list[dict[str, str]]:
    """Get a list of all users on the Jellyfin server, including their Name, Id, and LastActivityDate."""
    users = await _get("/Users")
    return [
        {
            "Name": u.get("Name", ""),
            "Id": u.get("Id", ""),
            "LastActivityDate": u.get("LastActivityDate", ""),
        }
        for u in users
    ]

@mcp.tool()
async def get_user_id(username: str | None = None) -> str:
    """Get the Jellyfin user ID for a given username.

    If no username is provided, it uses the default username configured in the environment.
    """
    return await _get_user_id(username)


@mcp.tool()
async def list_libraries() -> list[dict]:
    """List all Jellyfin media libraries with their IDs, names, types, and folder paths."""
    folders = await _get("/Library/VirtualFolders")
    return [
        {
            "id": f.get("ItemId", ""),
            "name": f["Name"],
            "type": f.get("CollectionType", "unknown"),
            "locations": f.get("Locations", []),
        }
        for f in folders
    ]


@mcp.tool()
async def scan_all_libraries() -> str:
    """Trigger a full rescan of all Jellyfin libraries. Returns immediately; scan runs in background."""
    await _post("/Library/Refresh")
    return "Full library scan triggered."


@mcp.tool()
async def scan_library(library_id: str) -> str:
    """Trigger a recursive rescan of a specific library.

    Args:
        library_id: The library/item ID from list_libraries().
    """
    await _post(f"/Items/{library_id}/Refresh", json={"Recursive": True})
    return f"Library scan triggered for {library_id}."


# ---------------------------------------------------------------------------
# Tools — Search & Browse
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_recently_watched(user_id: str, limit: int = 10) -> dict:
    """Get the most recently watched items.

    Args:
        user_id: The Jellyfin user ID.
        limit: Max results to return (default 10).
    """
    params = {
        "Recursive": "true",
        "IsPlayed": "true",
        "IncludeItemTypes": "Movie,Episode",    
        "SortBy": "DatePlayed",
        "SortOrder": "Descending",
        "Limit": limit,
        "Fields": "Overview,Genres,Tags,ProductionLocations,ProviderIds,SeriesStudio,Studios,People"
    }
    data = await _get(f"/Users/{user_id}/Items", params=params)
    raw_items = data.get("Items", [])

    # Keep one representative per series (for episodes) and avoid duplicate item IDs.
    seen_series: set[str] = set()
    seen_ids: set[str] = set()
    unique_items: list[dict] = []

    for item in raw_items:
        item_id = item.get("Id")
        if not item_id or item_id in seen_ids:
            continue

        if item.get("Type") == "Episode" and (series_id := item.get("SeriesId")):
            if series_id in seen_series:
                continue
            seen_series.add(series_id)

        unique_items.append(item)
        seen_ids.add(item_id)

    summarized_items: list[dict] = []
    for item in unique_items:
        if item.get("Type") == "Episode" and (series_id := item.get("SeriesId")):
            series_data = await get_item_details(series_id)
            summarized_items.append(series_data)
        else:
            summarized_items.append(_summarize_item(item))

    return {"items": summarized_items}

@mcp.tool()
async def get_item_details(item_id: str) -> dict:
    """Get detailed information for a specific Jellyfin item.

    Args:
        item_id: The item's Jellyfin ID.
    """
    params = {
        "Ids": item_id,
        "Fields": "Overview,Genres,Tags,ProductionLocations,ProviderIds,SeriesStudio,Studios,People"
    }
    try:
        data = await _get(f"/Items", params=params)
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            raise RuntimeError(f"Item '{item_id}' not found (404)") from e
        raise

    items = data.get("Items", [])
    if not items:
        raise RuntimeError(f"Item '{item_id}' not found")

    return _summarize_item(items[0])

@mcp.tool()
async def search_media(
    query: str,
    media_type: str = None,
    limit: int = 20,
    start_index: int = 0,
) -> dict:
    """Search the Jellyfin library by name/keyword.

    Args:
        query: Search term.
        media_type: Optional filter — one of Audio, MusicAlbum, MusicArtist, Movie, Series, Episode, Book, Playlist.
        limit: Max results to return (default 20).
        start_index: Offset for pagination.
    """
    params: dict[str, Any] = {
        "SearchTerm": query,
        "Limit": limit,
        "StartIndex": start_index,
        "Recursive": True,
        "Fields": "Overview,DateCreated",
    }
    if media_type:
        params["IncludeItemTypes"] = media_type
    data = await _get("/Items", params=params)
    return {
        "total_count": data.get("TotalRecordCount", 0),
        "items": _summarize_items(data.get("Items", [])),
    }

@mcp.tool()
async def browse_library(
    library_id: str = None,
    media_type: str = None,
    artist_ids: str = None,
    sort_by: str = "SortName",
    sort_order: str = "Ascending",
    limit: int = 20,
    start_index: int = 0,
) -> dict:
    """Browse items in a library with sorting and pagination.

    Args:
        library_id: Optional library ID to scope results.
        media_type: Optional type filter (Audio, MusicAlbum, MusicArtist, Movie, Series, Episode, Book).
        artist_ids: Optional comma-separated artist IDs to filter by (e.g. for albums by a specific artist).
        sort_by: Sort field — SortName, DateCreated, CommunityRating, ProductionYear, Random, etc.
        sort_order: Ascending or Descending.
        limit: Max results (default 20).
        start_index: Offset for pagination.
    """
    params: dict[str, Any] = {
        "SortBy": sort_by,
        "SortOrder": sort_order,
        "Limit": limit,
        "StartIndex": start_index,
        "Recursive": True,
        "Fields": "Overview,DateCreated",
    }
    if library_id:
        params["ParentId"] = library_id
    if media_type:
        params["IncludeItemTypes"] = media_type
    if artist_ids:
        params["ArtistIds"] = artist_ids
    data = await _get("/Items", params=params)
    return {
        "total_count": data.get("TotalRecordCount", 0),
        "items": _summarize_items(data.get("Items", [])),
    }


@mcp.tool()
async def get_recently_added(
    media_type: str = None,
    limit: int = 20,
) -> dict:
    """Get recently added items, sorted by date added (newest first).

    Args:
        media_type: Optional type filter (Audio, MusicAlbum, Movie, Series, Episode, Book).
        limit: Max results (default 20).
    """
    params: dict[str, Any] = {
        "SortBy": "DateCreated",
        "SortOrder": "Descending",
        "Limit": limit,
        "Recursive": True,
        "Fields": "Overview,DateCreated",
    }
    if media_type:
        params["IncludeItemTypes"] = media_type
    data = await _get("/Items", params=params)
    return {
        "total_count": data.get("TotalRecordCount", 0),
        "items": _summarize_items(data.get("Items", [])),
    }


@mcp.tool()
async def get_similar_items(item_id: str, limit: int = 10) -> list[dict]:
    """Get items similar to a given item (works for albums, movies, series, etc.).

    Args:
        item_id: The item's Jellyfin ID.
        limit: Max results (default 10).
    """
    data = await _get(f"/Items/{item_id}/Similar", params={"Limit": limit})
    return _summarize_items(data.get("Items", []))


# ---------------------------------------------------------------------------
# Tools — Playlist Management
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_playlists(user_id: str) -> list[dict]:
    """List all playlists on the server.

    Args:
        user_id: The Jellyfin user ID used to query playlist item counts.
    """
    data = await _get("/Items", params={
        "IncludeItemTypes": "Playlist",
        "Recursive": True,
    })
    playlists = _summarize_items(data.get("Items", []))

    async def _fetch_count(pl: dict) -> None:
        count_data = await _get(f"/Playlists/{pl['id']}/Items", params={
            "UserId": user_id,
            "Limit": 0,
        })
        pl["item_count"] = count_data.get("TotalRecordCount", 0)

    await asyncio.gather(*[_fetch_count(pl) for pl in playlists])
    return playlists


@mcp.tool()
async def create_playlist(
    name: str,
    user_id: str,
    item_ids: str = None,
    media_type: str = "Audio",
) -> dict:
    """Create a new playlist, optionally with initial items.

    Args:
        name: Playlist name.
        user_id: The Jellyfin user ID to own the playlist.
        item_ids: Optional comma-separated item IDs to add initially.
        media_type: Media type for the playlist — Audio, Video, etc. (default Audio).
    """
    body: dict[str, Any] = {
        "Name": name,
        "UserId": user_id,
        "MediaType": media_type,
    }
    if item_ids:
        body["Ids"] = [i.strip() for i in item_ids.split(",")]
    data = await _post("/Playlists", json=body)
    return {"id": data["Id"], "name": name}


@mcp.tool()
async def get_playlist_items(
    playlist_id: str,
    user_id: str,
    limit: int = 50,
    start_index: int = 0,
) -> dict:
    """Get items in a playlist. Each item includes a playlist_item_id needed for removal.

    Args:
        playlist_id: The playlist's Jellyfin ID.
        user_id: The Jellyfin user ID for the request.
        limit: Max results (default 50).
        start_index: Offset for pagination.
    """
    data = await _get(f"/Playlists/{playlist_id}/Items", params={
        "UserId": user_id,
        "Limit": limit,
        "StartIndex": start_index,
    })
    items = []
    for item in data.get("Items", []):
        s = _summarize_item(item)
        s["playlist_item_id"] = item.get("PlaylistItemId", item["Id"])
        items.append(s)
    return {
        "total_count": data.get("TotalRecordCount", 0),
        "items": items,
    }


@mcp.tool()
async def modify_playlist(
    playlist_id: str,
    user_id: str,
    add_item_ids: str = None,
    remove_item_ids: str = None,
) -> str:
    """Add and/or remove items from a playlist in a single operation.

    Args:
        playlist_id: The playlist's Jellyfin ID.
        user_id: The Jellyfin user ID used when adding items.
        add_item_ids: Comma-separated item IDs to add to the playlist.
        remove_item_ids: Comma-separated playlist-item IDs to remove (use playlist_item_id from get_playlist_items).
    """
    messages = []
    if add_item_ids:
        ids = [i.strip() for i in add_item_ids.split(",")]
        await _post(f"/Playlists/{playlist_id}/Items", params={
            "Ids": ",".join(ids),
            "UserId": user_id,
        })
        messages.append(f"Added {len(ids)} item(s).")
    if remove_item_ids:
        ids = [i.strip() for i in remove_item_ids.split(",")]
        await _delete(f"/Playlists/{playlist_id}/Items", params={
            "EntryIds": ",".join(ids),
        })
        messages.append(f"Removed {len(ids)} item(s).")
    return " ".join(messages) or "No changes requested."


@mcp.tool()
async def delete_playlist(playlist_id: str) -> str:
    """Permanently delete a playlist.

    Args:
        playlist_id: The playlist's Jellyfin ID.
    """
    await _delete(f"/Items/{playlist_id}")
    return f"Playlist {playlist_id} deleted."


# ---------------------------------------------------------------------------
# Tools — Scheduled Tasks
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_scheduled_tasks() -> list[dict]:
    """List all scheduled tasks on the Jellyfin server with their IDs, names, and states."""
    tasks = await _get("/ScheduledTasks")
    result = []
    for t in tasks:
        entry: dict[str, Any] = {
            "id": t["Id"],
            "name": t["Name"],
            "state": t["State"],
            "last_execution": t.get("LastExecutionResult", {}).get("EndTimeUtc"),
            "last_result": t.get("LastExecutionResult", {}).get("Status"),
        }
        if (pct := t.get("CurrentProgressPercentage")) is not None:
            entry["progress_percent"] = round(pct, 1)
        result.append(entry)
    return result


@mcp.tool()
async def run_scheduled_task(task_id: str) -> str:
    """Trigger a scheduled task to run immediately.

    Args:
        task_id: The task ID from list_scheduled_tasks().
    """
    await _post(f"/ScheduledTasks/Running/{task_id}")
    return f"Scheduled task {task_id} triggered."


# ---------------------------------------------------------------------------
# Tools — Server Status
# ---------------------------------------------------------------------------


@mcp.tool()
async def server_status(include: str = "info") -> dict:
    """Get Jellyfin server status information.

    Args:
        include: Comma-separated sections to include — any combination of "info", "sessions", "tasks", "activity".
                 Defaults to "info" if not specified.
    """
    sections = [s.strip() for s in include.split(",")]

    result: dict[str, Any] = {}

    if "info" in sections:
        info = await _get("/System/Info")
        result["info"] = {
            "server_name": info.get("ServerName"),
            "version": info.get("Version"),
            "os": info.get("OperatingSystem"),
            "has_pending_restart": info.get("HasPendingRestart", False),
            "local_address": info.get("LocalAddress"),
        }

    if "sessions" in sections:
        sessions = await _get("/Sessions")
        result["sessions"] = [
            {
                "user": s.get("UserName", ""),
                "client": s.get("Client", ""),
                "device": s.get("DeviceName", ""),
                "last_activity": s.get("LastActivityDate", ""),
                "now_playing": (
                    _summarize_item(s["NowPlayingItem"]) if s.get("NowPlayingItem") else None
                ),
            }
            for s in sessions
        ]

    if "tasks" in sections:
        tasks = await _get("/ScheduledTasks")
        result["tasks"] = [
            {
                "id": t["Id"],
                "name": t["Name"],
                "state": t["State"],
                "last_execution": t.get("LastExecutionResult", {}).get("EndTimeUtc"),
            }
            for t in tasks
        ]

    if "activity" in sections:
        log = await _get("/System/ActivityLog/Entries", params={"Limit": 20})
        result["activity"] = [
            {
                "date": e.get("Date", ""),
                "type": e.get("Type", ""),
                "overview": e.get("Overview", e.get("ShortOverview", "")),
                "user": e.get("UserName", ""),
            }
            for e in log.get("Items", [])
        ]

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")