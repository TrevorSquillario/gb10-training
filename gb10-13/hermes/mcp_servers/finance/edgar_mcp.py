"""EDGAR MCP Server — tools for SEC filings, insider trading, and institutional holdings."""

import asyncio
from gettext import find
from gettext import find
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from edgar import *
from edgar.xbrl import *
from bs4 import BeautifulSoup
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("hermes.tools.edgar")

mcp = FastMCP("edgar")

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def _get_identity() -> str:
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise ValueError("EDGAR_IDENTITY environment variable is not set.")
    return identity


def _setup_identity() -> None:
    set_identity(_get_identity())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_filings(symbol: str, forms: list[str], days: int = 90):
    """Fetch filings for a symbol filtered by form type and date range."""
    _setup_identity()
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    date_range = f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
    logger.info(f"Fetching filings {forms} for {symbol} from {start_date} to {end_date}")

    def _fetch():
        company = Company(symbol)
        return company.get_filings(form=forms, filing_date=date_range)

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        logger.error(f"Error fetching filings {forms} for {symbol}: {e}")
        return []


# ---------------------------------------------------------------------------
# Tools — Company Search
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_company(name: str) -> list[dict]:
    """Search for companies by name using EDGAR.

    Args:
        name: The name of the company to search for, e.g. "Apple".
    """
    def _find():
        return find(name)

    try:
        results = await asyncio.to_thread(_find)
        return [
            {
                "ticker": getattr(company, "ticker", None),
                "name": getattr(company, "name", None),
                "cik": getattr(company, "cik", None),
                "industry": getattr(company, "industry", None),
                "website": getattr(company, "website", None),
                "location": f"{getattr(company, 'city', 'N/A')}, {getattr(company, 'state', 'N/A')}",
                "sic_code": getattr(company, "sic", None),
                "fiscal_year_end": getattr(company, "fiscal_year_end", None),
                "exchange": getattr(company, "exchange", None),
            }
            for company in results
        ]
    except Exception as e:
        logger.error(f"Error searching for company '{name}': {e}")
        return []


# ---------------------------------------------------------------------------
# Tools — Insider Trading
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_insider_trading(symbol: str, days: int = 90) -> dict:
    """Fetch raw insider trading transactions (Form 4 and 5) for a symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        days: Number of calendar days to look back (default 90).
    """
    filings = await _get_filings(symbol, ["4", "5"], days)
    all_transactions: list[pd.DataFrame] = []

    def _parse_filing(filing):
        try:
            return filing.obj().to_dataframe()
        except Exception as e:
            logger.error(
                f"Could not parse Form 4/5 filing "
                f"{getattr(filing, 'accession_number', 'unknown')}: {e}"
            )
            return None

    parsed = await asyncio.gather(*[asyncio.to_thread(_parse_filing, f) for f in filings])
    all_transactions = [df for df in parsed if df is not None and not df.empty]

    if not all_transactions:
        return {"symbol": symbol, "days": days, "transactions": [], "count": 0}

    full_df = pd.concat(all_transactions, ignore_index=True)
    return {
        "symbol": symbol,
        "days": days,
        "count": len(full_df),
        "transactions": full_df.to_dict(orient="records"),
    }


@mcp.tool()
async def get_insider_trading_summary(
    symbol: str,
    days: int = 90,
    top_n: int = 10,
) -> dict:
    """Summarize insider buying and selling activity for a symbol.

    Returns aggregate buy/sell totals, buy-vs-sell ratio, top buyers, top sellers,
    and a monthly trend breakdown.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        days: Number of calendar days to look back (default 90).
        top_n: Number of top buyers/sellers to return (default 10).
    """
    raw = await get_insider_trading(symbol, days)
    transactions = raw.get("transactions", [])

    if not transactions:
        return {
            "symbol": symbol,
            "days": days,
            "sale_total": 0.0,
            "buy_total": 0.0,
            "buy_vs_sell_ratio": None,
            "top_sellers": [],
            "top_buyers": [],
            "monthly_trend": {},
        }

    df = pd.DataFrame(transactions)
    logger.debug(f"DataFrame columns: {df.columns.tolist()}")

    sale_mask = df["Transaction Type"].str.contains("sale", case=False, na=False)
    buy_mask = df["Transaction Type"].str.contains("purchase|buy", case=False, na=False)

    if "Value" in df.columns:
        df["value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    else:
        df["value"] = (
            pd.to_numeric(df.get("Shares", 0), errors="coerce").fillna(0.0)
            * pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0.0)
        )

    sale_total = float(df.loc[sale_mask, "value"].sum())
    buy_total = float(df.loc[buy_mask, "value"].sum())
    buy_vs_sell_ratio = (buy_total / sale_total) if sale_total != 0 else None

    group_cols = ["Insider", "Position"]

    def _top_group(mask, ascending):
        grouped = df.loc[mask].groupby(group_cols)["value"].sum().reset_index()
        top = grouped.sort_values("value", ascending=ascending).head(top_n)
        return [
            {"insider": r["Insider"], "position": r["Position"], "total_value": r["value"]}
            for _, r in top.iterrows()
        ]

    top_sellers = _top_group(sale_mask, ascending=False)
    top_buyers = _top_group(buy_mask, ascending=False)

    monthly_trend: dict = {}
    if "Date" in df.columns:
        df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str)
        for month in sorted(df["Month"].dropna().unique()):
            month_df = df[df["Month"] == month]
            s_mask = month_df["Transaction Type"].str.contains("sale", case=False, na=False)
            b_mask = month_df["Transaction Type"].str.contains("purchase|buy", case=False, na=False)
            s_total = float(month_df.loc[s_mask, "value"].sum())
            b_total = float(month_df.loc[b_mask, "value"].sum())
            monthly_trend[month] = {
                "sale_total": s_total,
                "buy_total": b_total,
                "buy_vs_sell_ratio": (b_total / s_total) if s_total != 0 else None,
            }

    return {
        "symbol": symbol,
        "days": days,
        "sale_total": sale_total,
        "buy_total": buy_total,
        "buy_vs_sell_ratio": buy_vs_sell_ratio,
        "top_sellers": top_sellers,
        "top_buyers": top_buyers,
        "monthly_trend": monthly_trend,
    }

# ---------------------------------------------------------------------------
# Tools — Institutional Holdings (13F)
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_institutional_holdings(symbol: str, days: int = 90) -> dict:
    """Fetch institutional holdings from the latest 13F-HR filing for a symbol. These are the investments the company is holding. 

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        days: Number of calendar days to look back when searching for filings (default 90).
    """
    filings = await _get_filings(symbol, ["13F-HR"], days)

    if not filings:
        logger.debug(f"No 13F-HR filings found for {symbol} in the last {days} days.")
        return {"symbol": symbol, "days": days, "count": 0, "holdings": []}

    latest_filing = filings[0]

    def _parse():
        obj = latest_filing.obj()
        if hasattr(obj, "infotable") and obj.infotable is not None:
            return obj.infotable
        return None

    try:
        infotable: Optional[pd.DataFrame] = await asyncio.to_thread(_parse)
    except Exception as e:
        logger.error(
            f"Could not parse 13F filing "
            f"{getattr(latest_filing, 'accession_number', 'unknown')}: {e}"
        )
        return {"symbol": symbol, "days": days, "count": 0, "holdings": []}

    if infotable is None or infotable.empty:
        return {"symbol": symbol, "days": days, "count": 0, "holdings": []}

    return {
        "symbol": symbol,
        "days": days,
        "accession_number": getattr(latest_filing, "accession_number", None),
        "count": len(infotable),
        "holdings": infotable.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Tools — Fundamentals (10-K / 10-Q)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_current_fundamentals(symbol: str, period: str = "annual") -> dict:
    """Fetch financial statements (income, balance sheet, cash flow) from
    the most recent 10-K (annual) or 10-Q (quarterly) filings for a symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        period: "annual" or "quarterly" (case-insensitive) or "10-K"/"10-Q".
        days: Lookback window in days to search for filings (default 730).
    """
    p = (period or "").lower()
    if p in ("annual", "10-k", "10k"):
        form = "10-K"
    elif p in ("quarterly", "10-q", "10q"):
        form = "10-Q"
    else:
        raise ValueError("period must be 'annual' or 'quarterly'")

    if form == "10-K":
        financials = Company(symbol).get_financials()
    elif form == "10-Q":
        financials = Company(symbol).get_quarterly_financials()

    if not financials:
        return {"symbol": symbol, "period": period, "fundamentals": []}

    def _parse_filing(financials):
        try:
            income = financials.income_statement().to_dataframe() if hasattr(financials, "income_statement") else None
            balance = financials.balance_sheet().to_dataframe() if hasattr(financials, "balance_sheet") else None
            cashflow = financials.cashflow_statement().to_dataframe() if hasattr(financials, "cashflow_statement") else None

            return {
                "accession_number": getattr(financials, "accession_number", None),
                "filing_date": getattr(financials, "filing_date", None),
                "income_statement": income.to_dict(orient="records") if income is not None else None,
                "balance_sheet": balance.to_dict(orient="records") if balance is not None else None,
                "cash_flow_statement": cashflow.to_dict(orient="records") if cashflow is not None else None,
            }
        except Exception as e:
            logger.error(
                f"Could not parse {form} filing " f"{getattr(financials, 'accession_number', 'unknown')}: {e}"
            )
            return None

    parsed = await asyncio.to_thread(_parse_filing, financials)

    return {
        "symbol": symbol,
        "period": period,
        "fundamentals": parsed,
    }


# ---------------------------------------------------------------------------
# Tools — Filing text extraction (Item 1, 1A, 7)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_filing_section(
    symbol: str,
    form: str = "10-K",
    days: int = 730,
    top: int = 1,
    sections: Optional[list[str]] = None,
) -> dict:
    """Fetch specified sections from the latest filing(s) (10-K or 10-Q).

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        form: Filing form to fetch, default "10-K". Accepts "10-Q" too.
        days: Lookback window in days when searching for filings (default 730).
        top: Number of most recent filings to fetch (default 1).
        sections: List of section keys to return. Defaults to ["business", "risk_factors", "management_discussion"].

    Returns a dict with requested section keys (when available) plus metadata
    (`accession_number`, `filing_date`, `symbol`, `form`, `count`). For
    `top==1` returns the sections as top-level keys; for `top>1` returns a
    `filings` list of objects containing the sections and metadata.
    """
    form_up = (form or "").upper()
    if form_up not in ("10-K", "10-Q"):
        p = (form or "").lower()
        if p in ("annual", "10-k", "10k"):
            form_up = "10-K"
        elif p in ("quarterly", "10-q", "10q"):
            form_up = "10-Q"
        else:
            raise ValueError("form must be '10-K' or '10-Q'")

    # default sections
    requested = [s for s in (sections or ["business", "risk_factors", "management_discussion"]) if s]

    filings = await _get_filings(symbol, [form_up], days)
    if not filings:
        # build empty response depending on top
        base = {"symbol": symbol, "form": form_up, "count": 0}
        if len(requested) == 1:
            # single requested section -> return that key with None
            k = requested[0]
            base.update({k: None})
            return base
        # multiple sections -> return each as None
        for k in requested:
            base.update({k: None})
        return base

    # limit to the requested top N filings
    try:
        top_n = int(top or 1)
    except Exception:
        top_n = 1
    if top_n < 1:
        top_n = 1

    filings = filings[:top_n]

    def _text(v):
        if v is None:
            return None
        if hasattr(v, "text"):
            return v.text
        if isinstance(v, (list, tuple)):
            return "\n\n".join(str(x) for x in v)
        return str(v)

    def _extract_for(filing):
        try:
            obj = filing.obj()
            # possible attribute fallbacks
            section_map = {
                "business": getattr(obj, "business", None) or getattr(obj, "item_1_business", None),
                "risk_factors": getattr(obj, "risk_factors", None) or getattr(obj, "item_1a_risk_factors", None),
                "management_discussion": getattr(obj, "management_discussion", None)
                or getattr(obj, "item_7_managements_discussion_and_analysis", None),
            }

            result = {
                "accession_number": getattr(filing, "accession_number", None),
                "filing_date": getattr(filing, "filing_date", None),
            }
            for k in requested:
                # if a user passed an unexpected key, return None for it
                v = section_map.get(k)
                result[k] = _text(v) if v is not None else None

            return result
        except Exception as e:
            logger.error(
                f"Could not parse filing text for {symbol}: {getattr(filing, 'accession_number', 'unknown')} - {e}"
            )
            return None

    # parse filings concurrently in threads
    parsed_list = await asyncio.gather(*[asyncio.to_thread(_extract_for, f) for f in filings])
    parsed_list = [p for p in parsed_list if p]

    if not parsed_list:
        base = {"symbol": symbol, "form": form_up, "count": 0}
        for k in requested:
            base.update({k: None})
        return base

    if top_n == 1:
        parsed = parsed_list[0]
        out = {"symbol": symbol, "form": form_up, "count": 1}
        out.update(parsed)
        return out
    # ---------------------------------------------------------------------------
    # Tools — Filing HTML / full text extraction
    # ---------------------------------------------------------------------------


@mcp.tool()
async def get_filing_html(symbol: str, form: str = "10-K", days: int = 730, top: int = 1) -> dict:
    """Fetch the filing text (extracted via BeautifulSoup) for the latest filing(s).

    Returns:
      - For `top==1`: a dict with `symbol`, `form`, `count`, and `text` (string or None).
      - For `top>1`: a dict with `symbol`, `form`, `count`, and `filings` (list of strings).
    """
    form_up = (form or "").upper()
    if form_up not in ("10-K", "10-Q"):
        p = (form or "").lower()
        if p in ("annual", "10-k", "10k"):
            form_up = "10-K"
        elif p in ("quarterly", "10-q", "10q"):
            form_up = "10-Q"
        else:
            raise ValueError("form must be '10-K' or '10-Q'")

    filings = await _get_filings(symbol, [form_up], days)
    try:
        top_n = int(top or 1)
    except Exception:
        top_n = 1
    if top_n < 1:
        top_n = 1

    if not filings:
        if top_n == 1:
            return {"symbol": symbol, "form": form_up, "count": 0, "text": None}
        return {"symbol": symbol, "form": form_up, "count": 0, "filings": []}

    filings = filings[:top_n]
    def _fetch_text(filing):
        try:
            html = filing.html()
            text = BeautifulSoup(html, "html.parser").get_text()
            return {
                "accession_number": getattr(filing, "accession_number", None),
                "filing_date": getattr(filing, "filing_date", None),
                "text": text,
            }
        except Exception as e:
            logger.error(
                f"Could not fetch HTML for {symbol}: {getattr(filing, 'accession_number', 'unknown')} - {e}"
            )
            return None

    parsed = await asyncio.gather(*[asyncio.to_thread(_fetch_text, f) for f in filings])
    parsed = [p for p in parsed if p is not None]

    if not parsed:
        if top_n == 1:
            return {"symbol": symbol, "form": form_up, "count": 0, "text": None, "filing_date": None}
        return {"symbol": symbol, "form": form_up, "count": 0, "filings": []}

    if top_n == 1:
        first = parsed[0]
        return {
            "symbol": symbol,
            "form": form_up,
            "count": 1,
            "text": first.get("text"),
            "filing_date": first.get("filing_date"),
            "accession_number": first.get("accession_number"),
        }

    return {"symbol": symbol, "form": form_up, "count": len(parsed), "filings": parsed}


# ---------------------------------------------------------------------------
# Tools — Compare Fundamentals (multiple filings)
# ---------------------------------------------------------------------------


@mcp.tool()
async def compare_fundamentals(
    symbol: str, form: str = "10-Q", head: int = 5, days: int = 730
) -> dict:
    """Compare fundamentals across the most-recent filings for a symbol.

    Tries to use `XBRLS.from_filings(...)` when the `xbrls` (or `xbrl`) package
    is available; otherwise falls back to per-filing parsing.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        form: Filing form to compare ("10-Q" or "10-K").
        head: Number of most-recent filings to include (default 5).
        days: Lookback window in days when searching for filings (default 730).
    """
    form_up = (form or "").upper()
    if form_up not in ("10-Q", "10-K"):
        # Allow users to pass just "quarterly"/"annual" too
        p = (form or "").lower()
        if p in ("annual", "10-k", "10k"):
            form_up = "10-K"
        elif p in ("quarterly", "10-q", "10q"):
            form_up = "10-Q"
        else:
            raise ValueError("form must be '10-Q' or '10-K'")

    filings = await _get_filings(symbol, [form_up], days)
    if not filings:
        return {"symbol": symbol, "form": form_up, "head": head, "count": 0, "comparison": {}}

    if XBRLS is not None:
        try:
            x = XBRLS.from_filings(filings)
            stmts: dict = {}
            if hasattr(x, "statements"):
                s = x.statements
                # common statement accessors used by many xbrl wrappers
                for name, accessor in (
                    ("income_statement", "income_statement"),
                    ("balance_sheet", "balance_sheet"),
                    ("cash_flow", "cashflow_statement"),
                ):
                    fn = getattr(s, accessor, None)
                    if fn is None:
                        # try alternative attribute name
                        fn = getattr(s, name, None)
                    if fn is None:
                        continue
                    try:
                        df = fn()
                        if hasattr(df, "to_dataframe"):
                            df = df.to_dataframe()
                        if isinstance(df, pd.DataFrame):
                            stmts[name] = df.fillna("").to_dict(orient="index")
                        else:
                            # attempt conversion to DataFrame for safer serialization
                            try:
                                stmts[name] = pd.DataFrame(df).fillna("").to_dict(orient="index")
                            except Exception:
                                stmts[name] = str(df)
                    except Exception as e:
                        logger.debug(f"Could not extract {name} via XBRLS: {e}")

            return {
                "symbol": symbol,
                "form": form_up,
                "head": head,
                "count": len(filings),
                "comparison": stmts,
            }
        except Exception as e:
            logger.error(f"XBRLS comparison failed for {symbol}: {e}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
