---
name: chart-analyst
description: Analyze technical indicators for chart patterns a given symbol and period.
version: 1.2.0
---


First load the data using the yfinance get_technical_indicators tool for the given symbol and period.

Analyze the data for these chart patterns for {symbol} over the last {period}:

Guidelines:
    - Only include potential or identified patterns found. 
    - Provide a brief explanation of each pattern identified and what data was relevant to this determination. 

**Candlestick Patterns**
1. **Bullish Reversal Patterns:**
    - Appear after a downtrend and suggest buying pressure is taking over.
    - **Hammer:** Single candlestick with a small body (red or green) and a long lower wick (at least twice the body). Little/no upper wick. Indicates sellers pushed price down, but buyers brought it back up. Strong bullish signal at support or after a downtrend.
    - **Bullish Engulfing:** Two-candlestick pattern. First is small bearish (red), second is large bullish (green) that engulfs the first. Shows buyers overpowering sellers. Reliable at the bottom of a downtrend.
    - **Morning Star:** Three-candlestick reversal pattern:
        - First: Long bearish (red), continues downtrend.
        - Second: Small-bodied (Doji/Spinning Top), gaps down, shows indecision.
        - Third: Long bullish (green), gaps up, closes well into first candle. Bulls take control.
2. **Bearish Reversal Patterns:**
    - Appear after an uptrend and suggest selling pressure is taking over.
    - **Shooting Star:** Bearish equivalent of inverted hammer. Small body (red/green), long upper wick. Buyers push price up, sellers push it back down. Strong reversal signal at resistance or after uptrend.
    - **Bearish Engulfing:** Inverse of bullish engulfing. First is small bullish (green), second is large bearish (red) that engulfs the first. Sellers take over.
    - **Evening Star:** Three-candlestick reversal pattern:
        - First: Long bullish (green), continues uptrend.
        - Second: Small-bodied, gaps up, indecision.
        - Third: Long bearish (red), gaps down, closes well into first candle. Uptrend likely over.
3. **Indecision and Continuation Patterns:**
    - **Doji:** Open and close nearly identical, very small body. Wick length varies. Represents indecision. Not strong alone, but at trend tops/bottoms can signal reversal. Most effective as part of a larger pattern (e.g., Morning/Evening Star).
    - **Spinning Top:** Like a Doji but with a slightly larger body. Also signals indecision. Series after a strong trend can mean trend is losing momentum.

**Chart Patterns**
1. **Head and Shoulders (Reversal Pattern):**
    - Reliable bearish reversal. Uptrend may be ending. Inverse pattern is bullish.
    - **How to Identify:**
        - **Left Shoulder:** Price rises to a peak, then declines.
        - **Head:** Price rises to a higher peak, then declines to first trough.
        - **Right Shoulder:** Price rises again, forms lower peak, then declines.
        - **Neckline:** Connects two lowest points between peaks. Break below (with volume) confirms pattern and signals down move.
2. **Double Top and Double Bottom (Reversal Patterns):**
    - Signal strong reversal after price fails to break support/resistance twice.
    - **Double Top (Bearish):**
        - **Uptrend:** Pattern forms after uptrend.
        - **First Top:** Price rises to new high, faces resistance, pulls back.
        - **Second Top:** Price rallies to same resistance, fails, declines.
        - **Neckline:** Support at low point between peaks. Break below (high volume) signals downtrend.
    - **Double Bottom (Bullish):**
        - **Downtrend:** Pattern forms after downtrend.
        - **First Bottom:** Price declines to new low, finds support, rallies.
        - **Second Bottom:** Price falls to same support, fails to break lower, rises.
        - **Neckline:** Resistance at high point between troughs. Break above (high volume) signals uptrend. Pattern resembles 'W'.
3. **Cup and Handle (Continuation Pattern):**
    - Bullish continuation. Brief pause in uptrend before price continues higher. Reliable pattern.
    - **How to Identify:**
        - **Cup:** "U" shaped curve as price declines, then recovers. Bottom should be broad/rounded, not sharp/V-shaped.
        - **Handle:** After cup, price consolidates in short, downward-sloping channel/flag in top half of cup.
        - **Breakout:** Confirmed when price breaks above handle resistance on volume. Target = cup depth + breakout point.
4. **Flags and Pennants (Continuation Patterns):**
    - Short-term continuation patterns, pause in strong trend. Characterized by sharp price move, then consolidation.
    - **Bull Flag/Pennant:**
        - **Flagpole:** Sharp, vertical rally on high volume.
        - **Flag/Pennant:** Price consolidates in tight range. Flag = rectangular channel sloping against trend; pennant = small symmetrical triangle.
        - **Breakout:** Confirmed when price breaks out in direction of trend, on volume.
    - **Bear Flag/Pennant:**
        - **Flagpole:** Sharp, rapid decline on high volume.
        - **Flag/Pennant:** Price consolidates in tight range. Flag = channel sloping against trend; pennant = small triangle.
        - **Breakout:** Confirmed when price breaks out downward, on volume.