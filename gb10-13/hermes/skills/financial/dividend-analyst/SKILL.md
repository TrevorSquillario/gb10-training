---
name: dividend-analyst
description: Analyze dividend data for a given symbol and provide a report.
version: 1.2.0
---

First load the dividend history using the yfinance get_dividend_history tool for the given symbol. 

Analyze the dividend history for {symbol}:

Dividend History:
{dividends}

Provide:
1. Dividend Yield: The annual dividend payment as a percentage of the stock price.
2. Dividend Payout Ratio: The percentage of earnings paid out as dividends. A sustainable payout ratio (e.g., below 60-70%) is important.
3. Dividend Growth History: A consistent history of increasing dividends can indicate financial strength and commitment to shareholders.
4. Any notable trends or risks in the dividend policy.