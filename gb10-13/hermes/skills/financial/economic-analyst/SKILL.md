---
name: economic-analyst
description: Analyze macroeconomic indicators and their implications for the stock market.
version: 1.2.0
---

First use the FRED get_economic_trends tool to retrieve the latest macroeconomic indicators for the last 6 months. Then, analyze the data to provide insights into the current economic environment and its implications for stock market investing and portfolio strategy.

Given the following recent US macroeconomic indicators (with trend statistics), provide an analysis of the current economic environment and its potential impact on stock market investing and portfolio strategy. Highlight any notable trends, risks, or opportunities.

GDP: {gdp}
Consumer Price Index for All Urban Consumers: All Items in U.S. City Average: {cpi}
Unemployment Rate: {unemployment}
Federal Funds Effective Rate: {interest_rate}
University of Michigan: Consumer Sentiment: {consumer_confidence}
Advance Retail Sales: Retail Trade and Food Services: {retail_sales}
Industrial Production: Total Index: {industrial_production}
CBOE Volatility Index: VIX: {vix}
10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity: {yield_curve}
New Privately-Owned Housing Units Started: Total Units: {housing_starts}
Producer Price Index by Commodity: All Commodities: {ppi}

Each indicator is a JSON summary with start/end values, trend, rate_of_change, min/max, mean, and std_dev.

Provide:
1. Summary of the current macroeconomic environment
2. Key risks and opportunities for investors
3. How these factors might influence portfolio allocation or stock selection
4. Any notable trends or signals to watch