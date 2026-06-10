---
name: insidertrading-analyst
description: Analyze insider trading activity for a given symbol and provide a sentiment report.
version: 1.0.0
---

First load the insider trading data using the edgar get_insider_trades tool for the given symbol and days period.

Analyze the insider trading activity for {symbol} over the last {days} days:

Provide a report including:
1. **Overview**
    - Total buy value vs. total sell value
    - Buy/sell ratio and what it signals about insider conviction
    - Net insider sentiment: Bullish, Bearish, or Neutral

2. **Top Sellers**
    - List the top insiders by sell volume, including their title/role
    - Flag any C-suite or Board-level sellers (CEO, CFO, Directors) as higher signal
    - Distinguish between likely planned 10b5-1 sales (pre-scheduled, lower signal) and discretionary sales (higher signal) where role context allows

3. **Top Buyers**
    - List the top insiders by buy volume, including their title/role
    - Open-market purchases by executives are considered the strongest bullish signal
    - Note any absence of buying activity during a sell-heavy period

4. **Monthly Trend Analysis**
    - Summarize the monthly breakdown of buy and sell totals
    - Identify acceleration or deceleration in selling/buying activity
    - Flag any months where the buy/sell ratio shifted significantly

5. **Signal Interpretation**
    - Insider selling is often routine (diversification, taxes, 10b5-1 plans) and should not be overweighted in isolation
    - Cluster selling — multiple insiders selling in the same window — is a stronger bearish signal
    - Complete absence of buying during a period of heavy selling warrants attention
    - Compare insider activity against price action context where available

6. **Risk Flags**
    - Heavy concentration of sales from a single high-ranking insider
    - Multiple insiders selling simultaneously
    - Selling near 52-week highs or ahead of known catalysts
    - No insider purchases in 90+ days despite price pullbacks

7. **Insider Sentiment Score** (0–100, where 0 is strongly bearish and 100 is strongly bullish)
    - Weight open-market buys heavily positive
    - Weight cluster selling or C-suite discretionary selling heavily negative
    - Treat routine/planned sales as neutral
