# app/services/outlook_yf.py
from __future__ import annotations
import time
from typing import Dict, List, Tuple

try:
    import yfinance as yf
except Exception:
    yf = None

DEFAULT_RPM = 30   # polite default requests per minute
DEFAULT_MAX = 40   # total call cap per run

def fetch_next_earnings(
    tickers: List[str],
    *,
    rpm: int = DEFAULT_RPM,
    max_calls: int = DEFAULT_MAX
) -> Dict[str, Tuple[str | None, str]]:
    """
    Returns {ticker: (iso_date_or_None, source)} for the next earnings date.
    Throttled to ~rpm and capped by max_calls to avoid 429s.
    No exceptions leak out: failures become (None, '...-error').
    """
    if yf is None:
        return {t: (None, "Data not available") for t in tickers}

    out: Dict[str, Tuple[str | None, str]] = {}
    # seconds between calls (rpm ~ requests/min)
    interval = max(60.0 / max(rpm, 1), 0.5)
    calls = 0

    for t in tickers:
        if calls >= max_calls:
            out[t] = (None, "throttle-cap")
            continue
        try:
            # This method typically hits a lighter endpoint than quoteSummary?modules=calendarEvents
            edf = yf.Ticker(t).get_earnings_dates(limit=1)
            calls += 1
            if edf is not None and not edf.empty:
                d = edf.index[0].date().isoformat()
                out[t] = (d, "yfinance")
            else:
                out[t] = (None, "yfinance")
        except Exception:
            out[t] = (None, "yfinance-error")
        time.sleep(interval)

    return out
