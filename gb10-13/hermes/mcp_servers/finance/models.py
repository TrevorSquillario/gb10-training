from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import date

class TechnicalIndicators(BaseModel):
    current_price: float
    current_volume: int
    sma_20: float
    sma_50: float
    sma_200: Optional[float] = None
    rsi: float
    volume_trend: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    obv: float
    atr: float
    ohlcv: Dict[str, Dict[str, str]]

class DividendHistory(BaseModel):
    dividends: Dict[str, float]

class Earning(BaseModel):
    date: str
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    net_income: Optional[float] = None
    net_income_continuous_operations: Optional[float] = None
    operating_income: Optional[float] = None
    gross_profit: Optional[float] = None
    total_revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    cost_of_goods_and_services_sold: Optional[float] = None
    selling_general_and_administrative: Optional[float] = None
    research_and_development: Optional[float] = None
    operating_expenses: Optional[float] = None
    investment_income_net: Optional[float] = None
    net_interest_income: Optional[float] = None
    interest_income: Optional[float] = None
    interest_expense: Optional[float] = None
    non_interest_income: Optional[float] = None
    other_non_operating_income: Optional[float] = None
    depreciation: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    income_before_tax: Optional[float] = None
    income_tax_expense: Optional[float] = None
    interest_and_debt_expense: Optional[float] = None
    net_income_from_continuing_operations: Optional[float] = None
    comprehensive_income_net_of_tax: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    revenue_growth: Optional[float] = None
    eps_growth: Optional[float] = None
    roce: Optional[float] = None
    capital_efficiency_ratio: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    gross_profit_margin: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    market_cap: Optional[int] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    trailing_eps: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None
    revenue_per_share: Optional[float] = None

class EarningsHistory(BaseModel):
    quarterly_earnings: List[Earning]
    annual_earnings: List[Earning]

class MarketData(BaseModel):
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: int
    beta: Optional[float] = None
    pe_ratio: float
    trailing_eps: float
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: float
    price_to_sales: float
    profit_margin: float
    operating_margin: float
    return_on_assets: float
    return_on_equity: float
    revenue_per_share: float

class NewsItem(BaseModel):
    title: str
    publisher: str
    url: str
    timestamp: str

class News(BaseModel):
    news: List[NewsItem]

class HistoricalTrackedValues(BaseModel):
    """
    Represents a concise set of significant financial, technical, and sentiment
    values to track over time for a specific stock.
    This model is designed for historical data storage and trend analysis.
    """
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")

    final_recommendation: str = Field(..., description="Final Recommendation (e.g., Strong Buy, Buy, Sell, Strong Sell, Hold)")
    final_confidence_score: float = Field(..., description="Confidence Score (0-10)")

    # --- Key Fundamental Metrics ---
    pe_ratio: Optional[float] = Field(None, description="Price-to-Earnings Ratio")
    revenue_growth_yoy: Optional[float] = Field(None, description="Revenue Growth Rate Year-over-Year (%)")
    eps_growth_yoy: Optional[float] = Field(None, description="Earnings Per Share Growth Year-over-Year (%)")
    net_profit_margin: Optional[float] = Field(None, description="Net Profit Margin (%)")
    free_cash_flow: Optional[float] = Field(None, description="Free Cash Flow (in currency units)")
    debt_to_equity_ratio: Optional[float] = Field(None, description="Debt-to-Equity Ratio")

    # --- Key Market Sentiment Data ---
    news_sentiment_score: Optional[float] = Field(None, description="Aggregated news sentiment score (e.g., 0 to 100)")
    analyst_consensus_rating: Optional[str] = Field(None, description="Consensus analyst rating (e.g., 'Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell')")

    # --- Key Earnings Report Data (from latest report if available for the period) ---
    #latest_eps_actual: Optional[float] = Field(None, description="Actual Earnings Per Share from the most recent report")
    #latest_eps_estimate: Optional[float] = Field(None, description="Analyst consensus EPS estimate from the most recent report")
    #latest_revenue_actual: Optional[float] = Field(None, description="Actual Revenue from the most recent report")
    #latest_revenue_estimate: Optional[float] = Field(None, description="Analyst consensus Revenue estimate from the most recent report")
    #guidance_summary: Optional[str] = Field(None, description="Summary of management's forward-looking guidance from the most recent report")

    # --- Price Targets ---
    current_price: Optional[float] = Field(None, description="Current price from technical analysis")
    low_price_target: Optional[float] = Field(None, description="Low price target from analysis")
    high_price_target: Optional[float] = Field(None, description="High price target from analysis")
    price_target_percent: Optional[float] = Field(None, description="Percent gain from current to high price target in analysis")

class TechnicalHistoricalTrackedValues(BaseModel):
    # --- Key Technical Indicators ---
    close_price: Optional[float] = Field(None, description="Closing price for the period")
    volume: Optional[int] = Field(None, description="Trading volume for the period")
    sma_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-day Simple Moving Average")
    rsi: Optional[float] = Field(None, description="Relative Strength Index (0-100)")

class SwingTradePlan(BaseModel):
    """
    A Pydantic model representing a complete swing trade plan, ready for execution.
    """

    ticker: str = Field(..., description="The stock ticker symbol (e.g., 'AAPL').")
    direction: str = Field(..., description="The trade direction: 'long' or 'short'.")
    entry_price: float = Field(..., gt=0, description="The price at which to enter the trade.")
    stop_loss_price: float = Field(..., gt=0, description="The price to exit the trade to limit losses.")
    take_profit_price: float = Field(..., gt=0, description="The price at which to exit the trade to lock in profits.")

    # Risk Management
    risk_per_trade_usd: float = Field(..., gt=0, description="The maximum amount of capital to risk on this trade.")
    position_size: int = Field(..., ge=0, description="The number of shares to buy/sell.")
    risk_reward_ratio: float = Field(..., gt=0, description="The calculated risk/reward ratio for the trade.")
    
    # Rationale and Execution Plan
    entry_reason: str = Field(..., description="A brief explanation for the entry signal (e.g., 'Breakout of Bull Flag on high volume').")
    exit_reason: str = Field(..., description="A brief explanation of the exit strategy (e.g., 'Target based on pattern measurement, Stop Loss below key support').")

    def __init__(self, **data):
        super().__init__(**data)
        # Add a simple validation to ensure stop_loss_price and take_profit_price are logical
        if self.direction == 'long':
            if not (self.stop_loss_price < self.entry_price < self.take_profit_price):
                raise ValueError("For a long trade, Stop Loss must be below Entry, and Take Profit above Entry.")
        elif self.direction == 'short':
            if not (self.stop_loss_price > self.entry_price > self.take_profit_price):
                raise ValueError("For a short trade, Stop Loss must be above Entry, and Take Profit below Entry.")