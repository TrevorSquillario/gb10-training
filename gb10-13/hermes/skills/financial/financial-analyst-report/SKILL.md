---
name: financial-analyst-report
description: Organize and manage financial analyses for a given stock ticker, synthesizing insights from multiple financial skills to provide a comprehensive investment recommendation.
version: 1.2.0
---

First use delegate_task to run the following skills in parallel for the {symbol} stock ticker. Split the tasks into an explicit list of separate parallel sub-tasks in batches of 4 for delegate_task. Make note of tasks that failed or returned incomplete data, and include that in the final report.

- /technical_analyst
- /news_analyst
- /fundamental_analyst
- /dividend_analyst
- /economic_analyst
- /short_analyst
- /insidertrading_analyst

Then, synthesize the insights from each analysis to provide a comprehensive investment recommendation for DELL.

Provide:
1. Final recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
2. One year target price range including: low price target, high price target, percent gain from current to high price target
3. Confidence score (1-10)
4. Write a bull case theory focused on momentum and catalysts. Look for reasons why the price will move significantly higher over the next few weeks to months. Consider the following factors:
    Trend Confirmation: Looking for "Higher Highs" and "Higher Lows" on daily/weekly charts.
    Positive Catalysts: Upcoming earnings beats, product launches, or favorable macroeconomic shifts (e.g., interest rate cuts).
    Volume Expansion: Identifying if big institutional buyers are entering the stock.
    Mean Reversion: Checking if the stock is "oversold" on the RSI but showing a hidden bullish divergence.
5. Write a bear case theory focused on risks and potential pitfalls. Look for the "Hidden Trap" that could cause the trade to fail. Consider the following factors:
    Resistance Levels: Identifying "heavy" supply zones where sellers historically take control.
    Macro Headwinds: Analyzing sector-wide weakness or geopolitical risks that could override individual stock strength.
    Valuation Fatigue: Flagging if a stock is trading at historical extremes (e.g., an unsustainably high P/E ratio).
    Technical Breakdowns: Looking for "Head and Shoulders" patterns or momentum exhaustion (RSI overbought).
6. Include a short highlight from each analysis (technical, news, fundamental, dividend, economic) that was used to inform the final recommendation. Each highlight should be 1-2 sentences summarizing the most important insight from that analysis.