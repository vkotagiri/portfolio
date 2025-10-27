# app/services/technicals.py
from __future__ import annotations
import pandas as pd

def rsi14(close: pd.Series) -> float | None:
    """
    Wilder's RSI(14) using EMA-style smoothing.
    Returns the latest RSI value or None if insufficient data.
    """
    if close is None or len(close) < 15:
        return None
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    val = float(rsi.iloc[-1])
    if pd.isna(val):
        return None
    return val

def macd_12_26_9(close: pd.Series, recent_sessions: int = 2) -> dict:
    """
    Compute MACD(12,26,9) and annotate:
      - direction: 'Bullish' if MACD > Signal, else 'Bearish'
      - last_crossover: ISO date of most-recent MACD/Signal cross
      - recent_crossover: True if last cross occurred within the last `recent_sessions` bars
      - recent_crossover_type: 'bullish' or 'bearish' for that last cross
    """
    base = {
        "macd": None,
        "signal": None,
        "hist": None,
        "direction": "Data not available",
        "last_crossover": "Data not available",
        "recent_crossover": False,
        "recent_crossover_type": "unknown",
    }
    if close is None or len(close) < 35:
        return base

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    direction = "Bullish" if macd.iloc[-1] > signal.iloc[-1] else "Bearish"

    cross_up = (hist.shift(1) <= 0) & (hist > 0)   # bullish cross
    cross_dn = (hist.shift(1) >= 0) & (hist < 0)   # bearish cross

    last_up_idx = hist.index[cross_up].max() if cross_up.any() else None
    last_dn_idx = hist.index[cross_dn].max() if cross_dn.any() else None

    last_idx = None
    last_type = "unknown"
    if last_up_idx is not None and last_dn_idx is not None:
        last_idx = max(last_up_idx, last_dn_idx)
        last_type = "bullish" if last_idx == last_up_idx else "bearish"
    elif last_up_idx is not None:
        last_idx = last_up_idx
        last_type = "bullish"
    elif last_dn_idx is not None:
        last_idx = last_dn_idx
        last_type = "bearish"

    recent = False
    if last_idx is not None and len(hist.index) >= (recent_sessions + 1):
        cutoff_idx = hist.index[-recent_sessions]
        recent = last_idx >= cutoff_idx

    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "hist": float(hist.iloc[-1]),
        "direction": direction,
        "last_crossover": (last_idx.date().isoformat() if last_idx is not None else "Data not available"),
        "recent_crossover": recent,
        "recent_crossover_type": last_type,
    }
