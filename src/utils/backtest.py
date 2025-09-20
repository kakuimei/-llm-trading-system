import pandas as pd
import numpy as np
from typing import Dict, Any
from .metrics import annualized_return, sharpe_ratio, compute_max_drawdown
from .portfolio import apply_positions, pnl_from_weights
import matplotlib.pyplot as plt

def run_backtest(df: pd.DataFrame,
                 agent,
                 trading_cost_bps: float = 5.0) -> Dict[str, Any]:
    # Assumes df is sorted by date; intra-day uses close-to-close
    all_dates = sorted(df["date"].unique())
    equity = [1.0]
    rets = []
    logs = []

    for i in range(len(all_dates)-1):
        d = all_dates[i]
        nd = all_dates[i+1]
        day_df = df[df["date"] == d].copy()
        next_df = df[df["date"] == nd].copy()

        preds = agent.predict_day(day_df)
        alloc = apply_positions(day_df, preds)
        ret = pnl_from_weights(alloc, next_df, trading_cost_bps=trading_cost_bps)
        rets.append(ret)
        equity.append(equity[-1] * (1 + ret))

        logs.append({"date": d, "ret": ret, "equity": equity[-1]})

    equity_series = pd.Series(equity, index=all_dates)
    ret_series = pd.Series(rets, index=all_dates[1:])
    stats = {
        "annual_return": annualized_return(ret_series),
        "sharpe": sharpe_ratio(ret_series),
        "max_drawdown": compute_max_drawdown(equity_series)
    }

    # Plot equity curve
    plt.figure()
    equity_series.plot(title="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    out_path = "./outputs/equity_curve.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return {"stats": stats, "equity_path": out_path, "logs": logs}
