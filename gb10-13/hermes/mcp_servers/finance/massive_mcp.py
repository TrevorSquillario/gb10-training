"""Massive MCP Server — curated tools for market data via api.massive.com."""

import asyncio
import os
import logging
import time
from pathlib import Path
from typing import Any, Optional, Dict, Tuple, List

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from models import (
    Earning,
    EarningsHistory,
)
from utils import safe_float

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")
logger = logging.getLogger("hermes.tools.finance.massive")

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
MASSIVE_BASE_URL = "https://api.massive.com"
RATE_LIMIT = 5  # requests per minute

if not MASSIVE_API_KEY:
    raise RuntimeError("MASSIVE_API_KEY environment variable must be set")

mcp = FastMCP("massive")

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_session: httpx.AsyncClient | None = None
_last_request_time: float = 0


def _get_session() -> httpx.AsyncClient:
    global _session
    if _session is None:
        _session = httpx.AsyncClient()
    return _session


async def _get(endpoint: str, params: dict | None = None) -> Any:
    """Rate-limited GET request to the Massive API."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    min_interval = 60 / RATE_LIMIT
    if elapsed < min_interval:
        await asyncio.sleep(min_interval - elapsed)

    url = f"{MASSIVE_BASE_URL}{endpoint}"
    request_params = dict(params or {})
    request_params["apiKey"] = MASSIVE_API_KEY

    session = _get_session()
    response = await session.get(url, params=request_params)
    _last_request_time = time.time()
    if response.status_code != 200:
        error_text = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {error_text}")
    data = response.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data


# ---------------------------------------------------------------------------
# Financial value extraction helpers
# ---------------------------------------------------------------------------


def _extract_value(data_dict: dict, field_name: str) -> Optional[float]:
    """Extract a numeric value from a Massive financial field dict."""
    if field_name not in data_dict:
        return None
    field_data = data_dict[field_name]
    if not isinstance(field_data, dict):
        return None
    for key in ("value", "value_usd", "amount", "raw_value"):
        if key in field_data:
            return safe_float(field_data[key])
    for key, val in field_data.items():
        if key != "order":
            result = safe_float(val)
            if result is not None:
                return result
    return None


_INCOME_FIELD_MAP = {
    "net_income": "net_income_loss",
    "operating_income": "operating_income_loss",
    "gross_profit": "gross_profit",
    "total_revenue": "revenues",
    "cost_of_revenue": "cost_of_revenue",
    "cost_of_goods_and_services_sold": "cost_of_revenue",
    "operating_expenses": "operating_expenses",
    "interest_income": "interest_income",
    "interest_expense": "interest_expense_operating",
    "income_before_tax": "income_loss_from_continuing_operations_before_tax",
    "income_tax_expense": "income_tax_expense_benefit",
    "research_and_development": "research_and_development",
    "selling_general_and_administrative": "selling_general_and_administrative_expenses",
    "depreciation": "depreciation",
    "depreciation_and_amortization": "depreciation_and_amortization",
    "net_income_from_continuing_operations": "income_loss_from_continuing_operations_after_tax",
    "comprehensive_income_net_of_tax": "comprehensive_income_loss",
    "interest_and_debt_expense": "interest_expense",
    "investment_income_net": "investment_income_net",
    "net_interest_income": "net_interest_income",
    "non_interest_income": "nonoperating_income_loss",
    "other_non_operating_income": "other_non_operating_income",
}


def _build_earning(result_item: dict, previous_result_item: dict | None = None) -> Optional[Earning]:
    """Build an Earning model from a Massive financials result item."""
    if not result_item or "financials" not in result_item:
        return None

    income = result_item["financials"].get("income_statement", {})
    balance = result_item["financials"].get("balance_sheet", {})
    cash_flow = result_item["financials"].get("cash_flow_statement", {})

    # EPS
    reported_eps = _extract_value(income, "basic_earnings_per_share")

    # Base fields
    fields: dict[str, Any] = {
        "date": result_item.get("fiscal_date", result_item.get("end_date", "")),
        "reported_eps": reported_eps,
    }

    # Map income statement fields
    for model_field, api_field in _INCOME_FIELD_MAP.items():
        val = _extract_value(income, api_field)
        if val is not None:
            fields[model_field] = val

    # Derived / balance-sheet ratios
    if balance:
        ebit = _extract_value(income, "operating_income_loss")
        revenue = _extract_value(income, "revenues")
        net_income = _extract_value(income, "net_income_loss")
        total_assets = _extract_value(balance, "assets")
        current_assets = _extract_value(balance, "current_assets")
        current_liabilities = _extract_value(balance, "current_liabilities")
        total_liabilities = _extract_value(balance, "liabilities")
        equity = _extract_value(balance, "equity")
        inventory = _extract_value(balance, "inventory")
        gross_profit = _extract_value(income, "gross_profit")
        basic_avg_shares = _extract_value(income, "basic_average_shares")

        # D&A
        da = _extract_value(income, "depreciation_and_amortization")
        if da is None:
            dep = _extract_value(income, "depreciation") or 0.0
            amt = _extract_value(income, "amortization") or 0.0
            da = dep + amt if (dep or amt) else None

        # EBIT fallback
        if ebit is None and net_income is not None:
            interest_exp = _extract_value(income, "interest_expense") or 0.0
            tax = _extract_value(income, "income_tax_expense_benefit") or 0.0
            ebit = net_income + interest_exp + tax

        # EBITDA
        ebitda = None
        if ebit is not None and da is not None:
            ebitda = ebit + da

        # Cash flow
        op_cf = _extract_value(cash_flow, "net_cash_flow_from_operating_activities")
        capex = _extract_value(cash_flow, "capital_expenditures")
        if capex is None:
            net_inv = _extract_value(cash_flow, "net_cash_flow_from_investing_activities")
            capex = -net_inv if net_inv is not None else None

        capital_employed = (
            total_assets - current_liabilities
            if total_assets is not None and current_liabilities is not None
            else None
        )

        def _ratio(num, den):
            return num / den if num is not None and den is not None and den != 0 else None

        ratios = {
            "ebit": ebit,
            "ebitda": ebitda,
            "roce": _ratio(ebit, capital_employed),
            "capital_efficiency_ratio": _ratio(revenue, capital_employed),
            "current_ratio": _ratio(current_assets, current_liabilities),
            "quick_ratio": (
                _ratio((current_assets or 0) - (inventory or 0), current_liabilities)
                if current_assets is not None and inventory is not None and current_liabilities
                else None
            ),
            "debt_to_equity": _ratio(total_liabilities, equity),
            "gross_profit_margin": _ratio(gross_profit, revenue),
            "operating_margin": _ratio(ebit, revenue),
            "return_on_assets": _ratio(net_income, total_assets),
            "return_on_equity": _ratio(net_income, equity),
            "operating_cash_flow": op_cf,
            "free_cash_flow": op_cf - capex if op_cf is not None and capex is not None else None,
            "revenue_per_share": _ratio(revenue, basic_avg_shares),
        }
        fields.update({k: v for k, v in ratios.items() if v is not None})

    return Earning(**fields)


# ---------------------------------------------------------------------------
# Tools — Short Interest / Volume
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_short_interest(
    ticker: str,
    limit: int = 10,
    sort: str = "settlement_date.desc",
) -> list[dict]:
    """Get short-interest records for a ticker.

    Args:
        ticker: Ticker symbol, e.g. "AAPL".
        limit: Number of records to return (default 10).
        sort: Sort order, e.g. "settlement_date.desc".
    """
    data = await _get("/stocks/v1/short-interest", {"ticker": ticker, "limit": limit, "sort": sort})
    if isinstance(data, dict) and "results" in data:
        return data["results"] or []
    return data if isinstance(data, list) else []


@mcp.tool()
async def get_short_volume(
    ticker: str,
    limit: int = 10,
    sort: str = "date.desc",
) -> list[dict]:
    """Get short-volume records for a ticker.

    Args:
        ticker: Ticker symbol, e.g. "AAPL".
        limit: Number of records to return (default 10).
        sort: Sort order, e.g. "date.desc".
    """
    data = await _get("/stocks/v1/short-volume", {"ticker": ticker, "limit": limit, "sort": sort})
    if isinstance(data, dict) and "results" in data:
        return data["results"] or []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Tools — Earnings
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_earnings_history(symbol: str) -> EarningsHistory:
    """Get annual and quarterly earnings history for a symbol from Massive.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
    """
    price_task = _get(f"/v2/aggs/ticker/{symbol}/prev", {})
    quarterly_task = _get("/vX/reference/financials", {
        "ticker": symbol, "timeframe": "quarterly",
        "limit": 8, "sort": "filing_date", "order": "desc",
    })
    annual_task = _get("/vX/reference/financials", {
        "ticker": symbol, "timeframe": "annual",
        "limit": 5, "sort": "filing_date", "order": "desc",
    })

    price_data, quarterly_data, annual_data = await asyncio.gather(
        price_task, quarterly_task, annual_task, return_exceptions=True
    )

    # Current price (optional, used for PE/PS enrichment)
    current_price: Optional[float] = None
    if isinstance(price_data, dict) and price_data.get("results"):
        current_price = safe_float(price_data["results"][0].get("c"))

    def _valid_results(data) -> list[dict]:
        if isinstance(data, Exception) or not isinstance(data, dict):
            return []
        return [
            r for r in data.get("results", [])
            if "financials" in r
            and "income_statement" in r.get("financials", {})
            and "balance_sheet" in r.get("financials", {})
        ]

    def _enrich(earning: Optional[Earning]) -> Optional[Earning]:
        if earning and current_price:
            if earning.reported_eps:
                earning.pe_ratio = current_price / earning.reported_eps
            if earning.total_revenue and earning.revenue_per_share:
                earning.price_to_sales = current_price / earning.revenue_per_share
        return earning

    # Build quarterly earnings (compare each quarter to same quarter prior year)
    quarterly_valid = _valid_results(quarterly_data)
    quarterly_earnings = []
    for i in range(min(4, len(quarterly_valid))):
        current = quarterly_valid[i]
        current_date = current.get("end_date", "")
        previous = None
        if current_date:
            try:
                cy, cm = int(current_date[:4]), int(current_date[5:7])
                best, best_diff = None, 12
                for q in quarterly_valid:
                    qd = q.get("end_date", "")
                    if qd and qd != current_date:
                        qy, qm = int(qd[:4]), int(qd[5:7])
                        if qy == cy - 1 and abs(qm - cm) < best_diff:
                            best_diff, best = abs(qm - cm), q
                previous = best
            except (ValueError, IndexError):
                pass
        earning = _enrich(_build_earning(current, previous))
        if earning:
            quarterly_earnings.append(earning)

    # Build annual earnings (each year vs. prior year)
    annual_valid = _valid_results(annual_data)
    annual_earnings = []
    for i in range(min(4, len(annual_valid))):
        previous = annual_valid[i + 1] if i + 1 < len(annual_valid) else None
        earning = _enrich(_build_earning(annual_valid[i], previous))
        if earning:
            annual_earnings.append(earning)

    return EarningsHistory(quarterly_earnings=quarterly_earnings, annual_earnings=annual_earnings)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
