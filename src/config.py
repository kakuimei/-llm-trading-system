from dataclasses import dataclass

@dataclass
class Config:
    date_col: str = "date"
    ticker_col: str = "ticker"
    price_cols: tuple = ("open", "high", "low", "close", "volume")
    news_col: str = "news"          # daily news text
    pe_col: str = "pe"              # example fundamental
    roe_col: str = "roe"            # example fundamental

    # Backtest params
    trading_cost_bps: float = 5.0   # round-trip cost in basis points (0.05% as default per trade side ~ adjustable)
    vol_target: float = 0.15        # annualized vol target
    max_dd: float = 0.25            # max drawdown cap; reduce risk if breached
    seed: int = 42
