# app/services/earnings.py
from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Optional
import yfinance as yf

from ..config import settings
from ..utils.rate_limit import RateLimiter

# Reuse a gentle limiter for yfinance calls
_YF_LIMITER = RateLimiter(rpm=settings.yf_rpm, burst=settings.yf_burst, rpd=None)


def _to_date(x) -> Optional[date]:
    try:
        return x.date()  # pandas Timestamp-like
    except Exception:
        try:
            return date.fromisoformat(str(x)[:10])
        except Exception:
            return None


def upcoming_earnings_next_14d(tickers: List[str], asof: date) -> List[Dict[str, object]]:
    """
    Best-effort upcoming earnings using yfinance calendars.
    Returns: [{ticker, event_date(iso), source}]
    """
    results: List[Dict[str, object]] = []
    end = asof + timedelta(days=14)

    for t in tickers:
        try:
            _YF_LIMITER.wait()
            cal = yf.Ticker(t).calendar
            if cal is None or cal.empty:
                continue

            ev = None
            if "Earnings Date" in cal.index:
                row = cal.loc["Earnings Date"]
                ev = _to_date(row[0]) if len(row) > 0 else _to_date(row)
            elif "EarningsDate" in cal.index:
                row = cal.loc["EarningsDate"]
                ev = _to_date(row[0]) if len(row) > 0 else _to_date(row)

            if ev and asof <= ev <= end:
                results.append({"ticker": t, "event_date": ev.isoformat(), "source": "yfinance"})
        except Exception:
            continue

    return sorted(results, key=lambda r: (r["event_date"], r["ticker"]))
