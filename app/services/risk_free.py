# app/services/risk_free.py
from __future__ import annotations
import os, requests
from datetime import date
import pandas as pd

# FRED: 1-Week Treasury Bill Secondary Market Rate (DTB1WK)
FRED_SERIES = "DTB1WK"

def fred_api_key() -> str | None:
    return os.getenv("FRED_API_KEY")

def latest_1w_tbill(as_of: date) -> dict:
    """
    Returns: {"value": float_percent_or_None, "date": "YYYY-MM-DD", "source": "FRED" or "Data not available"}
    If no key or fetch fails, value will be None (caller should print 'Data not available').
    """
    key = fred_api_key()
    if not key:
        return {"value": None, "date": as_of.isoformat(), "source": "Data not available"}

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": FRED_SERIES,
        "api_key": key,
        "file_type": "json",
        "observation_start": (as_of.replace(day=1)).isoformat(),  # pull this month's data
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty or "value" not in df:
            return {"value": None, "date": as_of.isoformat(), "source": "FRED"}
        df = df[df["value"].apply(lambda v: v not in (None, ".", ""))]
        if df.empty:
            return {"value": None, "date": as_of.isoformat(), "source": "FRED"}
        row = df.iloc[-1]
        return {"value": float(row["value"]), "date": row["date"], "source": "FRED"}
    except Exception:
        return {"value": None, "date": as_of.isoformat(), "source": "FRED"}
