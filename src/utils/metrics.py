import numpy as np
import pandas as pd

def annualized_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    cum = (1 + daily_returns).prod()
    years = len(daily_returns) / trading_days
    if years <= 0:
        return 0.0
    return float(cum ** (1/years) - 1)

def sharpe_ratio(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    mean = daily_returns.mean() * trading_days
    vol = daily_returns.std(ddof=1) * np.sqrt(trading_days)
    return float(mean / (vol + 1e-9))

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak.replace(0, np.nan)
    return float(dd.min())
