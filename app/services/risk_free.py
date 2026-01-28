# app/services/risk_free.py
from __future__ import annotations
import os, requests
from datetime import date
import pandas as pd

# FRED: 1-Week Treasury Bill Secondary Market Rate (DTB1WK)
FRED_SERIES = "DTB1WK"

# Fallback rate when FRED data is unavailable (e.g., future dates in simulation)
# Updated periodically based on Fed Funds rate environment
FALLBACK_WEEKLY_RATE_PCT = 5.25  # ~5.25% annualized as of late 2024

def fred_api_key() -> str | None:
    return os.getenv("FRED_API_KEY")

def latest_1w_tbill(as_of: date) -> dict:
    """
    Returns: {"value": float_percent_or_None, "date": "YYYY-MM-DD", "source": "FRED" or "fallback"}
    If no key or fetch fails, uses fallback rate.
    """
    key = fred_api_key()
    if not key:
        return {"value": FALLBACK_WEEKLY_RATE_PCT, "date": as_of.isoformat(), "source": "fallback (no API key)"}

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": FRED_SERIES,
        "api_key": key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 30,  # Get recent observations
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty or "value" not in df:
            return {"value": FALLBACK_WEEKLY_RATE_PCT, "date": as_of.isoformat(), "source": "fallback (no FRED data)"}
        df = df[df["value"].apply(lambda v: v not in (None, ".", ""))]
        if df.empty:
            return {"value": FALLBACK_WEEKLY_RATE_PCT, "date": as_of.isoformat(), "source": "fallback (no FRED data)"}
        # Use most recent available observation
        row = df.iloc[0]
        return {"value": float(row["value"]), "date": row["date"], "source": "FRED"}
    except Exception:
        return {"value": FALLBACK_WEEKLY_RATE_PCT, "date": as_of.isoformat(), "source": "fallback (API error)"}
