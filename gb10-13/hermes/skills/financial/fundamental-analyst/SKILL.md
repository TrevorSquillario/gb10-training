---
name: fundamental-analyst
description: Analyze fundamental data for a given symbol and provide a report.
version: 1.2.0
---

First load the company market data using the yfinance get_market_data tool for the given symbol. 
Then load the company earnings history using the yfinance get_earnings_history tool for the given symbol. 

Analyze the market context for {symbol}:

Provide:
1. Market sentiment
2. Sector analysis
    Durable Competitive Advantage ("Economic Moat"): He seeks businesses with a sustainable advantage that protects them from competition. This "moat" allows the company to consistently generate high returns on capital.
3. Risk assessment
4. Market outlook
5. Earnings and Revenue Growth: Use the Earnings and Income Statement History to analyze the following metrics over time.
    Total Revenue / Operating Revenue: These represent the top line of a company's income statement and indicate the company's ability to generate sales from its primary operations. Consistent growth in revenue is a fundamental indicator of a healthy and expanding business.
    Gross Profit: This shows the profit a company makes after deducting the costs associated with producing its goods or services. It's a key indicator of a company's efficiency in managing its production costs and pricing strategy.
    Operating Income / Total Operating Income As Reported: This metric reflects the profit generated from a company's core operations before deducting interest and taxes. It's crucial for understanding the operational efficiency and profitability of the business itself, separate from financing or tax strategies.
    Net Income / Net Income Common Stockholders / Net Income From Continuing Operations: These are often considered the "bottom line" and represent the total profit of a company after all expenses, including taxes and interest, have been deducted. It's a comprehensive measure of profitability and what's available to shareholders.
    Diluted EPS (Earnings Per Share) / Basic EPS: EPS is a crucial metric for investors, as it indicates how much profit a company makes per outstanding share of stock. Tracking EPS over time reveals the company's ability to generate profits for its shareholders, even as the number of shares may change due to factors like stock options or conversions.
    Return on Capital Employed (ROCE): A higher ROCE indicates better capital efficiency. It's a strong indicator of a company's ability to generate returns from its investments.
    Capital Efficiency Ratio: A higher ratio means the company is more effective at using its capital to generate revenue. This can vary significantly by industry.
6. Profitability Ratios:
    Net Profit Margin: How much profit does the company make for every dollar of revenue? A higher and consistent margin is desirable.
    Return on Equity (ROE): Measures how efficiently a company is using shareholder investments to generate profits. A healthy ROE (often 10-20% or higher, depending on the industry) indicates good management of shareholder capital.
    Return on Assets (ROA): Shows how efficiently a company is using its assets to generate earnings.
7. Valuation Ratios:
    Price-to-Earnings (P/E) Ratio: Compares the stock price to its earnings per share. A high P/E might suggest overvaluation or high growth expectations, while a low P/E could indicate undervaluation or challenges. Compare it to industry peers and historical averages.
    PEG Ratio (Price/Earnings-to-Growth): This ratio takes into account the company's earnings growth, making it useful for evaluating growth stocks. A PEG ratio of 1 or less is generally considered favorable.
8. Financial Health and Debt:
    Debt-to-Equity (D/E) Ratio: Indicates the proportion of debt a company uses to finance its assets relative to shareholder equity. A lower D/E ratio (e.g., 1 or lower) suggests less financial risk.
    Free Cash Flow (FCF): The cash a company generates after covering its operating expenses and capital expenditures. Strong and increasing FCF indicates a company's ability to fund growth, pay dividends, or buy back shares.