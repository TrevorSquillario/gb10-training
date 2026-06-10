"""FRED MCP Server — macroeconomic data from the Federal Reserve Bank of St. Louis."""

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

try:
    from .utils import safe_float
except Exception:
    # Support running this file directly (no parent package) for pytest
    # Insert repository root into sys.path and import using absolute package names.
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from mcp_servers.finance.utils import safe_float

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("hermes.tools.fred")

mcp = FastMCP("fred")

# FRED series IDs for common US macro indicators
_SERIES = {
    "gdp": "GDP",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "interest_rate": "FEDFUNDS",
    "consumer_confidence": "UMCSENT",
    "retail_sales": "RSAFS",
    "industrial_production": "INDPRO",
    "vix": "VIXCLS",
    "yield_curve_10y_2y": "T10Y2Y",
    "housing_starts": "HOUST",
    "ppi": "PPIACO",
}

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY environment variable is not set.")
    return key


def _normalize_date(dt: Optional[str]) -> Optional[str]:
    """Accept several common date formats and return YYYY-MM-DD or None.

    Supports: YYYY-MM-DD, MM/DD/YYYY, MM/DD/YY, ISO formats.
    Raises ValueError for unrecognized non-empty strings.
    """
    if not dt:
        return None
    # Accept already-correct format first
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(dt, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Try ISO parsing as a last resort
    try:
        return datetime.fromisoformat(dt).strftime("%Y-%m-%d")
    except Exception:
        raise ValueError(f"Invalid date format: {dt}")


def _fetch_series(series_id: str, start_date: Optional[str], end_date: Optional[str]) -> dict:
    """Synchronous FRED API call — run inside asyncio.to_thread."""
    params: dict = {
        "series_id": series_id,
        "api_key": _get_api_key(),
        "file_type": "json",
    }
    # Normalize incoming dates to YYYY-MM-DD (FRED API requirement)
    start_date = _normalize_date(start_date)
    end_date = _normalize_date(end_date)
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date
    resp = requests.get(_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _compute_trend(series_data: dict) -> dict:
    """Calculate summary statistics for a single FRED series response."""
    observations = series_data.get("observations", [])
    valid_obs = [o for o in observations if o.get("value") not in (None, ".", "")]
    values: list[float] = []
    dates: list[str] = []
    for obs in valid_obs:
        v = safe_float(obs["value"])
        if v is not None:
            values.append(v)
            dates.append(obs["date"])

    if not values:
        return {
            "series_start_value": None,
            "series_start_date": None,
            "series_end_value": None,
            "series_end_date": None,
            "trend": None,
            "rate_of_change": None,
            "absolute_change": None,
            "min_value": None,
            "min_date": None,
            "max_value": None,
            "max_date": None,
            "mean_value": None,
            "std_dev": None,
        }

    start_value = values[0]
    end_value = values[-1]
    min_value = min(values)
    max_value = max(values)
    min_index = values.index(min_value)
    max_index = values.index(max_value)
    mean_value = sum(values) / len(values)
    std_dev = (
        math.sqrt(sum((v - mean_value) ** 2 for v in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )

    return {
        "series_start_value": start_value,
        "series_start_date": dates[0],
        "series_end_value": end_value,
        "series_end_date": dates[-1],
        "trend": end_value / start_value if start_value != 0 else None,
        "rate_of_change": (end_value - start_value) / start_value if start_value != 0 else None,
        "absolute_change": end_value - start_value,
        "min_value": min_value,
        "min_date": dates[min_index],
        "max_value": max_value,
        "max_date": dates[max_index],
        "mean_value": mean_value,
        "std_dev": std_dev,
    }


# ---------------------------------------------------------------------------
# Tools — Individual Series
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_fred_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Fetch raw observations for any FRED series by its series ID.

    Args:
        series_id: FRED series identifier, e.g. "GDP", "CPIAUCSL", "UNRATE".
        start_date: Start date in YYYY-MM-DD format (optional).
        end_date: End date in YYYY-MM-DD format (optional).
    """
    return await asyncio.to_thread(_fetch_series, series_id, start_date, end_date)


@mcp.tool()
async def get_economic_indicator(
    indicator: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Fetch observations for a named macroeconomic indicator.

    Args:
        indicator: One of: gdp, cpi, unemployment, interest_rate, consumer_confidence,
                   retail_sales, industrial_production, vix, yield_curve_10y_2y,
                   housing_starts, ppi.
        start_date: Start date in YYYY-MM-DD format (optional).
        end_date: End date in YYYY-MM-DD format (optional).
    """
    series_id = _SERIES.get(indicator.lower())
    if series_id is None:
        valid = ", ".join(_SERIES.keys())
        raise ValueError(f"Unknown indicator '{indicator}'. Valid options: {valid}")
    return await asyncio.to_thread(_fetch_series, series_id, start_date, end_date)


# ---------------------------------------------------------------------------
# Tools — Bulk Data
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_bulk_economic_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Fetch raw observations for all tracked macroeconomic indicators in parallel.

    Returns a dict keyed by indicator name (gdp, cpi, unemployment, interest_rate,
    consumer_confidence, retail_sales, industrial_production, vix, yield_curve_10y_2y,
    housing_starts, ppi).

    Args:
        start_date: Start date in YYYY-MM-DD format (optional).
        end_date: End date in YYYY-MM-DD format (optional).
    """
    tasks = {
        key: asyncio.to_thread(_fetch_series, series_id, start_date, end_date)
        for key, series_id in _SERIES.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {
        key: result if not isinstance(result, Exception) else {"error": str(result)}
        for key, result in zip(tasks.keys(), results)
    }


@mcp.tool()
async def get_economic_trends(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Fetch all macroeconomic indicators and return summary statistics for each.

    For each series, returns: start/end values and dates, trend ratio, rate of change,
    absolute change, min/max with dates, mean, and standard deviation.

    Args:
        start_date: Start date in YYYY-MM-DD format (optional).
        end_date: End date in YYYY-MM-DD format (optional).
    """
    logger.info("Fetching economic trends from FRED for period %s to %s", start_date, end_date)
    bulk = await get_bulk_economic_data(start_date=start_date, end_date=end_date)
    trends = {}
    for key, series_data in bulk.items():
        if "error" in series_data:
            trends[key] = {"error": series_data["error"]}
        else:
            trends[key] = _compute_trend(series_data)
            logger.debug("Processed %s trend: %s", key, trends[key])
    return trends


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
