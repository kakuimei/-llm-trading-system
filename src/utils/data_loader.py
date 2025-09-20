import pandas as pd
import numpy as np
import ast
from typing import Tuple
from .feature_engineering import add_technical_features, clean_news_text, build_prompt

def load_data(path: str,
              date_col: str = "date",
              ticker_col: str = "ticker") -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- Schema harmonization for Chinese column names ---
    zh_map = {
        "Unnamed: 0": "date",
        "thscode": "ticker",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "新闻": "news",
        "市盈率ttm": "pe",
        # Optional extras (kept if present):
        "市盈率": "pe_raw",
        "市现率ttm": "pcf_ttm",
        "市净率": "pb",
        "市销率": "ps",
        "市销率ttm": "ps_ttm",
        "昨日收盘价": "prev_close",
    }
    if any(k in df.columns for k in zh_map.keys()):
        df = df.rename(columns={k: v for k, v in zh_map.items() if k in df.columns})
        date_col = "date" if "date" in df.columns else date_col
        ticker_col = "ticker" if "ticker" in df.columns else ticker_col

    # Parse news if it is a JSON-like list of dicts in string form
    if "news" in df.columns:
        def _parse_news(cell):
            if pd.isna(cell):
                return ""
            s = str(cell)
            # try to literal_eval to list[dict]
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, (list, tuple)):
                    snippets = []
                    for it in obj:
                        if isinstance(it, dict):
                            # prefer common keys
                            parts = []
                            for key in ["title","titl","content","summary","abstract","desc","text"]:
                                if key in it and isinstance(it[key], str):
                                    parts.append(it[key])
                            if parts:
                                snippets.append(" ".join(parts))
                    if snippets:
                        return " \n ".join(snippets)
            except Exception:
                pass
            return s  # fallback raw
        df["news"] = df["news"].apply(_parse_news)

    # Basic sanitation
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    if ticker_col not in df.columns:
        raise ValueError(f"Missing ticker column: {ticker_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Fill missing numeric cols conservatively
    for col in ["open","high","low","close","volume","pe","roe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # forward-fill within each ticker for fundamentals
    for col in ["pe","roe"]:
        if col in df.columns:
            df[col] = df.groupby(ticker_col)[col].ffill()

    # News cleaning
    if "news" in df.columns:
        df["news"] = df["news"].fillna("").astype(str).apply(clean_news_text)
    else:
        df["news"] = ""

    # Add technicals (returns/vol/RSI); handle missing OHLCV gracefully
    df = add_technical_features(df)
    return df

def time_split(df: pd.DataFrame,
               start: str,
               end: str,
               valid_ratio: float = 0.15,
               test_ratio: float = 0.15,
               date_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Restrict by date window
    mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
    dfx = df.loc[mask].copy()
    dates = dfx[date_col].sort_values().unique()
    n = len(dates)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    n_train = n - n_valid - n_test
    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError("Not enough data for the requested split.")
    train_dates = dates[:n_train]
    valid_dates = dates[n_train:n_train+n_valid]
    test_dates  = dates[n_train+n_valid:]
    return (dfx[dfx[date_col].isin(train_dates)].copy(),
            dfx[dfx[date_col].isin(valid_dates)].copy(),
            dfx[dfx[date_col].isin(test_dates)].copy())


def make_training_samples(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.001):
    samples = []
    for _, row in df.iterrows():
        # 构造标签：次日收益
        ret_next = row["close_next"] / row["close"] - 1 if "close_next" in row else None
        if ret_next is None:
            continue
        if ret_next > threshold:
            direction = "bullish"
        elif ret_next < -threshold:
            direction = "bearish"
        else:
            direction = "neutral"
        confidence = min(1.0, abs(ret_next) / 0.05)  # 简单归一化

        # 构造 prompt（训练时禁止 RAG，防止泄露）
        prompt = build_prompt(row, context_docs=None, use_rag=False)

        samples.append({
            "input": prompt,
            "output": {"direction": direction, "confidence": confidence},
            "meta": {"date": str(row["date"]), "ticker": row["ticker"]}
        })
    return samples
