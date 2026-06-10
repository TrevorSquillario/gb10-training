---
name: news-analyst
description: Analyze recent news for a given symbol and provide a sentiment report.
version: 1.2.0
---

First load the latest news using the yfinance get_news tool for the given symbol. 
Then use delegate_task to navigate to each url, in parallel, and extract the main content of the news article. 
Finally, analyze the content to provide a sentiment report for the symbol.

Analyze these recent news items for {symbol}:

Provide a report including:
1. Overall sentiment
2. Key developments
3. Potential impact 
    a. Positive Catalysts: Upcoming earnings beats, product launches, or favorable macroeconomic shifts (e.g., interest rate cuts).
    b. Volume Expansion: Identifying if big institutional buyers are entering the stock.
4. Risk factors including:
    a. Macro Headwinds: Analyzing sector-wide weakness or geopolitical risks that could override individual stock strength.
5. Sentiment Score (0-100, where 100 is very positive)