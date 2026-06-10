---
name: short-analyst
description: Analyze short interest and short volume data for a given symbol and provide a bearish pressure report.
version: 1.0.0
---

First load both datasets in parallel using the massive tools:
- `get_short_interest` for {symbol} with limit 20, sorted by `settlement_date.desc`
- `get_short_volume` for {symbol} with limit 20, sorted by `date.desc`

Each `get_short_interest` record contains: `ticker`, `settlement_date`, `short_interest` (shares short), `avg_daily_volume`, `days_to_cover`.

Analyze the short selling activity for {symbol}:

Provide a report including:
1. **Short Interest Overview**
    - Most recent `short_interest` (total shares sold short)
    - Most recent `days_to_cover`: the number of days it would take to buy back all short positions at average daily volume
        - < 1 day: Very low short pressure
        - 1–3 days: Moderate
        - 3–7 days: Elevated — warrants attention
        - > 7 days: High short squeeze risk or strong bearish conviction
    - Most recent `avg_daily_volume` for context
    - Settlement date of the most recent reading

2. **Short Interest Trend**
    - Track `short_interest` across all returned settlement dates (oldest → newest)
    - Is short interest rising, falling, or stable?
    - Calculate the percentage change from the oldest to the most recent reading
    - Flag any sharp single-period increases (>20%) as a potential bearish escalation signal

3. **Days-to-Cover Trend**
    - Track `days_to_cover` across all settlement dates
    - A rising days-to-cover alongside rising short interest confirms growing bearish conviction
    - A rising days-to-cover with flat or falling short interest may indicate declining average volume (liquidity concern)

4. **Short Volume Analysis**
    - Summarize recent daily short volume records: date, short volume, total volume, and implied short volume ratio (short_volume / total_volume)
    - A short volume ratio above 50% on multiple consecutive days is a strong bearish signal
    - Identify any spikes or sustained elevation in short volume ratio

5. **Short Squeeze Risk Assessment**
    - High `days_to_cover` (>5) + rising short interest = elevated squeeze risk if a positive catalyst occurs
    - A sudden drop in short interest after a period of high readings may indicate short covering (potential bullish price action)
    - Cross-reference short volume spikes with the short interest trend to identify capitulation or escalation

6. **Signal Summary**
    - Combine short interest trend, days-to-cover level, and short volume ratio into a directional signal
    - Bearish signals: rising short interest, high days_to_cover, elevated short volume ratio
    - Bullish contrarian signals: short interest declining sharply (covering), squeeze setup (high days_to_cover + rising price)

7. **Short Pressure Score** (0–100, where 0 is maximum bearish short pressure and 100 is minimal/short-squeeze setup)
    - Weight days_to_cover heavily (>7 days scores low)
    - Weight short interest trend direction
    - Weight short volume ratio elevation
