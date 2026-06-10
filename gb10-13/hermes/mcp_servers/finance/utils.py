import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("hermes.tools.finance.utils")

def market_open() -> bool:
    """Return True if the market is open (MST 7:30 AM - 2:00 PM), else False. Log warning if closed."""
    tz_name = os.environ.get("TZ", "America/Denver")
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    open_dt = now.replace(hour=7, minute=30, second=0, microsecond=0)
    close_dt = now.replace(hour=14, minute=0, second=0, microsecond=0)
    is_open = open_dt <= now <= close_dt and now.weekday() < 5
    return is_open

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_yoy_growth(current_value: Optional[float], previous_value: Optional[float]) -> Optional[float]:
    """Calculate year-over-year growth rate, rounded to 1 decimal place"""
    if current_value is None or previous_value is None or previous_value == 0:
        return None
    return round(((current_value - previous_value) / abs(previous_value)) * 100, 2)

def calculate_macd(close: pd.Series) -> dict:
    """Calculate MACD and signal line."""
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    return {
        'macd': float(macd.iloc[-1]),
        'macd_signal': float(signal.iloc[-1]),
        'macd_hist': float(macd.iloc[-1] - signal.iloc[-1])
    }

def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    """Calculate Bollinger Bands."""
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return {
        'bb_upper': float(upper_band.iloc[-1]),
        'bb_middle': float(sma.iloc[-1]),
        'bb_lower': float(lower_band.iloc[-1])
    }

def calculate_obv(close: pd.Series, volume: pd.Series) -> float:
    """Calculate On-Balance Volume (OBV)."""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return float(obv[-1])

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
    """Calculate Average True Range (ATR)."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return float(atr.iloc[-1])

# Helper to match by fiscalDateEnding
def find_by_date(reports, date_key):
    for r in reports:
        if r.get("fiscalDateEnding") == date_key:
            return r
    return None

def safe_float(value) -> Optional[float]:
    """Safely convert a value to float, handling None and "None" cases"""
    if value is None or value == "None" or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None