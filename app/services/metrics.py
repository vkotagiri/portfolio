import pandas as pd

def rsi(series: pd.Series, period: int = 14):
    if len(series) < period + 1:
        return None
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

def sma(series: pd.Series, window: int):
    if len(series) < window:
        return None
    return float(series.tail(window).mean())

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    if len(series) < slow + signal:
        return None, None, None
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(hist.iloc[-1])
