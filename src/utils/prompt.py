PROMPT_TEMPLATE = """
You are a financial analyst producing **daily** directional signals for a single U.S. stock.

{context_block}
Today's structured snapshot:
- Ticker: {ticker}
- Date: {date}
- Close: {close}, Ret_1d: {ret_1d}, Ret_5d: {ret_5d}
- Vol_20d: {vol_20d}, RSI_14: {rsi_14}
- Fundamentals: {fundamentals}

Today's News (raw, may be empty): 
\"\"\"{news}\"\"\"

Task:
1) Infer a one-day-ahead **direction** (one of: bullish / bearish / neutral).
2) Provide a **confidence** between 0 and 1.
3) VERY briefly justify with at most one sentence.

Return JSON like: {{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reason": "..."}}.
"""