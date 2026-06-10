"""YFinance MCP Server — curated tools for market data and technical analysis."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP

try:
    from .models import (
        DividendHistory,
        Earning,
        EarningsHistory,
        MarketData,
        News,
        NewsItem,
        TechnicalIndicators,
    )
    from .utils import (
        calculate_atr,
        calculate_bollinger_bands,
        calculate_macd,
        calculate_obv,
        calculate_rsi,
        market_open,
        safe_float,
    )
except Exception:
    # Support running this file directly (no parent package) for pytest
    # Insert repository root into sys.path and import using absolute package names.
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from mcp_servers.finance.models import (
        DividendHistory,
        Earning,
        EarningsHistory,
        MarketData,
        News,
        NewsItem,
        TechnicalIndicators,
    )
    from mcp_servers.finance.utils import (
        calculate_atr,
        calculate_bollinger_bands,
        calculate_macd,
        calculate_obv,
        calculate_rsi,
        market_open,
        safe_float,
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger("hermes.tools.yfinance")

mcp = FastMCP("yfinance")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_interval(period: str) -> str:
    """Return the appropriate interval for a given period.

    Valid periods: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo
    """
    if period == "1d":
        return "1h"
    if period == "5d":
        return "5m"
    if period == "1y":
        return "1wk"
    return "1d"


async def _fetch_history(symbol: str, period: str, interval: Optional[str] = None) -> pd.DataFrame:
    """Fetch single-symbol history in a thread pool."""
    if interval is None:
        interval = _get_interval(period)
    if market_open():
        print("WARNING: ***MARKET OPEN*** — data is from previous close.")
    stock = yf.Ticker(symbol)
    return await asyncio.to_thread(stock.history, period=period, interval=interval)


# ---------------------------------------------------------------------------
# Tools — History
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_stock_history(
    symbols: str,
    period: str = "1mo",
    interval: Optional[str] = None,
) -> dict:
    """Fetch OHLCV history for one or more ticker symbols.

    Args:
        symbols: Comma-separated ticker symbol(s), e.g. "AAPL" or "AAPL,MSFT,GOOG".
        period: Lookback period — 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: Bar interval — 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo, etc. Auto-selected if omitted.
    """
    symbol_list = [s.strip() for s in symbols.split(",")]
    resolved_interval = interval or _get_interval(period)
    if market_open():
        print("WARNING: ***MARKET OPEN*** — data is from previous close.")
    hist: pd.DataFrame = await asyncio.to_thread(
        yf.download, symbol_list, period=period, interval=resolved_interval
    )
    if hist.empty:
        return {"symbols": symbol_list, "period": period, "interval": resolved_interval, "rows": 0, "data": {}}
    hist.index = hist.index.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "symbols": symbol_list,
        "period": period,
        "interval": resolved_interval,
        "rows": len(hist),
        "data": hist.to_dict(orient="index"),
    }


# ---------------------------------------------------------------------------
# Tools — Technical Indicators
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_technical_indicators(symbol: str, period: str = "6mo") -> TechnicalIndicators:
    """Calculate technical indicators for a symbol: SMA, RSI, MACD, Bollinger Bands, OBV, ATR.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        period: Lookback period — 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    """
    hist = await _fetch_history(symbol, period)
    if hist.empty or "Close" not in hist or hist["Close"].isna().all():
        raise ValueError(f"No historical close data for {symbol} with period '{period}'")

    sma_20 = hist["Close"].rolling(window=20).mean()
    sma_50 = hist["Close"].rolling(window=50).mean()
    sma_200 = hist["Close"].rolling(window=200).mean()
    rsi = calculate_rsi(hist["Close"])
    macd_data = calculate_macd(hist["Close"])
    bb_data = calculate_bollinger_bands(hist["Close"])
    obv = calculate_obv(hist["Close"], hist["Volume"])
    atr = calculate_atr(hist["High"], hist["Low"], hist["Close"])

    ohlcv_df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close"]:
        ohlcv_df[col] = ohlcv_df[col].map("{:.4f}".format)
    ohlcv_df["Volume"] = ohlcv_df["Volume"].astype(int).astype(str)
    ohlcv_df.index = ohlcv_df.index.strftime("%Y-%m-%d %H:%M:%S")

    window = min(20, len(hist["Volume"]))
    avg_recent = hist["Volume"].iloc[-5:].mean()
    avg_window = hist["Volume"].iloc[-window:].mean()
    volume_trend = float(avg_recent / avg_window) if window > 0 and avg_window != 0 else 0.0

    return TechnicalIndicators(
        current_price=float(hist["Close"].iloc[-1]),
        current_volume=int(hist["Volume"].iloc[-1]),
        sma_20=float(sma_20.iloc[-1]),
        sma_50=float(sma_50.iloc[-1]),
        sma_200=float(sma_200.iloc[-1]),
        rsi=float(rsi.iloc[-1]),
        volume_trend=volume_trend,
        macd=macd_data["macd"],
        macd_signal=macd_data["macd_signal"],
        macd_hist=macd_data["macd_hist"],
        bb_upper=bb_data["bb_upper"],
        bb_middle=bb_data["bb_middle"],
        bb_lower=bb_data["bb_lower"],
        obv=obv,
        atr=atr,
        ohlcv=ohlcv_df.to_dict(orient="index"),
    )


# ---------------------------------------------------------------------------
# Tools — Fundamentals
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_market_data(symbol: str) -> MarketData:
    """Get fundamental market data for a symbol: valuation, margins, growth, and cash flow metrics.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
    """
    stock = yf.Ticker(symbol)
    info: dict[str, Any] = await asyncio.to_thread(lambda: stock.info)
    return MarketData(
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=info.get("marketCap") or 0,
        beta=safe_float(info.get("beta")),
        pe_ratio=safe_float(info.get("trailingPE")) or 0.0,
        trailing_eps=safe_float(info.get("trailingEps")) or 0.0,
        forward_pe=safe_float(info.get("forwardPE")),
        peg_ratio=safe_float(info.get("pegRatio")),
        price_to_book=safe_float(info.get("priceToBook")) or 0.0,
        price_to_sales=safe_float(info.get("priceToSalesTrailing12Months")) or 0.0,
        profit_margin=safe_float(info.get("profitMargins")) or 0.0,
        operating_margin=safe_float(info.get("operatingMargins")) or 0.0,
        return_on_assets=safe_float(info.get("returnOnAssets")) or 0.0,
        return_on_equity=safe_float(info.get("returnOnEquity")) or 0.0,
        revenue_per_share=safe_float(info.get("revenuePerShare")) or 0.0,
    )


@mcp.tool()
async def get_dividend_history(symbol: str) -> DividendHistory:
    """Get the full dividend payment history for a symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
    """
    stock = yf.Ticker(symbol)
    div: pd.Series = await asyncio.to_thread(lambda: stock.dividends)
    if div.empty:
        return DividendHistory(dividends={})
    div.index = div.index.strftime("%Y-%m-%d")
    return DividendHistory(dividends=div.to_dict())


@mcp.tool()
async def get_earnings_history(symbol: str) -> EarningsHistory:
    """Get annual and quarterly earnings history from the income statement.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
    """
    stock = yf.Ticker(symbol)
    annual_stmt, quarterly_stmt = await asyncio.gather(
        asyncio.to_thread(lambda: stock.income_stmt),
        asyncio.to_thread(lambda: stock.quarterly_income_stmt),
    )
    return EarningsHistory(
        annual_earnings=_parse_income_stmt(annual_stmt),
        quarterly_earnings=_parse_income_stmt(quarterly_stmt),
    )


# ---------------------------------------------------------------------------
# Tools — News
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_news(symbol: str, limit: int = 10) -> News:
    """Get the latest news headlines for a symbol.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        limit: Maximum number of news items to return (default 10).
    """
    stock = yf.Ticker(symbol)
    raw_news: list = await asyncio.to_thread(lambda: stock.news)
    items = []
    for item in raw_news[:limit]:
        content = item.get("content", {})
        items.append(NewsItem(
            title=content.get("title", ""),
            publisher=content.get("provider", {}).get("displayName", ""),
            url=content.get("canonicalUrl", {}).get("url", ""),
            timestamp=content.get("pubDate", ""),
        ))
    return News(news=items)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_income_stmt(stmt: pd.DataFrame) -> list[Earning]:
    """Convert a yfinance income statement DataFrame into a list of Earning models."""
    if stmt is None or stmt.empty:
        return []
    earnings = []
    for date, row in stmt.transpose().iterrows():
        earnings.append(Earning(
            date=date.strftime("%Y-%m-%d"),
            reported_eps=safe_float(row.get("Diluted EPS")),
            net_income=safe_float(row.get("Net Income")),
            net_income_continuous_operations=safe_float(row.get("Net Income Continuous Operations")),
            operating_income=safe_float(row.get("Operating Income")),
            gross_profit=safe_float(row.get("Gross Profit")),
            total_revenue=safe_float(row.get("Total Revenue")),
            cost_of_revenue=safe_float(row.get("Cost Of Revenue")),
            selling_general_and_administrative=safe_float(row.get("Selling General Administrative")),
            research_and_development=safe_float(row.get("Research And Development")),
            operating_expenses=safe_float(row.get("Operating Expense")),
            depreciation=safe_float(row.get("Depreciation")),
            depreciation_and_amortization=safe_float(row.get("Reconciled Depreciation")),
            income_before_tax=safe_float(row.get("Pretax Income")),
            income_tax_expense=safe_float(row.get("Tax Provision")),
            interest_expense=safe_float(row.get("Interest Expense")),
            ebit=safe_float(row.get("EBIT")),
            ebitda=safe_float(row.get("EBITDA")),
        ))
    return earnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
