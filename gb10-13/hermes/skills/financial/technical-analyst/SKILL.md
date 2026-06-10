---
name: technical-analyst
description: Analyze technical indicators for a given symbol and period.
version: 1.2.0
---

First load the data using the yfinance get_technical_indicators tool for the given symbol and period, then analyze the technical indicators based on the provided schema.

Analyze these technical indicators for {symbol} over the last {period}:

Provide:
1.  **Current Price:** The most recent price provided.
2.  **Trend Analysis:**
    * **Short-term (20-day SMA):** Is the price above/below, and what does it suggest?
    * **Medium-term (50-day SMA):** Is the price above/below, and what does it suggest?
    * **Long-term (200-day SMA):** Is the price above/below, and what does it suggest?
    * **RSI Interpretation:** Is it overbought, oversold, or neutral? What momentum does it indicate?
    * **Volume Trend Interpretation:** Is volume confirming or diverging from price trends?
    * **MACD Interpretation:** Analyze the MACD, signal line, and histogram. Is there a bullish or bearish crossover? What does the MACD suggest about momentum and trend strength?
    * **Bollinger Bands Interpretation:** Is the price near the upper or lower band? Is volatility increasing or decreasing? Are there any squeeze or breakout signals?
    * **OBV Interpretation:** Analyze the On-Balance Volume. Is OBV rising or falling? Is it confirming price trends or diverging? What does it suggest about buying/selling pressure?
    * **ATR Interpretation:** Analyze the Average True Range. Is volatility increasing or decreasing? What does the ATR suggest about risk and potential price movement?
3.  **Support and Resistance Levels:** Identify key levels based on the provided historical OHLCV data. Provide 1-2 immediate support levels and 1-2 immediate resistance levels with brief explanations.
4.  **Technical Rating:** A single, clear rating (e.g., "Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish") based on the combined analysis.
5.  **Key Signals & Actionable Insights:** Summarize significant technical patterns, potential breakouts/breakdowns, or other critical signals derived from the indicators. Suggest potential short-term implications.