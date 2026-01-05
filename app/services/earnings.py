# app/services/earnings.py
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Dict, List, Optional

import requests
from ..config import settings

_YF_BASE = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{sym}"
_YF_PARAMS = {
    "modules": "calendarEvents",
    "corsDomain": "finance.yahoo.com",
    "formatted": "false",
    "crumb": "Edge: Too Many Requests",
}
_YF_HEADERS = {
    # A UA helps reduce 429s on some networks
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def _sleep_between_calls(i: int):
    # Calls per minute; slower = fewer 429s
    rpm = max(1, int(getattr(settings, "yf_rpm", 30)))
    base = 60.0 / float(rpm)
    if i > 0:
        time.sleep(base)

def _fetch_calendar(sym: str) -> Optional[dict]:
    url = _YF_BASE.format(sym=sym)
    try:
        r = requests.get(url, params={**_YF_PARAMS, "symbol": sym}, headers=_YF_HEADERS, timeout=12)
        if r.status_code == 429:
            time.sleep(2.5)
            r = requests.get(url, params={**_YF_PARAMS, "symbol": sym}, headers=_YF_HEADERS, timeout=12)
        r.raise_for_status()
        data = r.json()
        result = (data or {}).get("quoteSummary", {}).get("result", [])
        return result[0] if result else None
    except Exception:
        return None

def _parse_when(cal_json: dict) -> str:
    try:
        when = cal_json["calendarEvents"]["earnings"].get("earningsCallTime") or ""
        s = str(when).lower()
        if "before" in s:  return "BMO"
        if "after" in s:   return "AMC"
        return "TBD"
    except Exception:
        return "TBD"

def _parse_date(cal_json: dict) -> Optional[date]:
    try:
        arr = cal_json["calendarEvents"]["earnings"]["earningsDate"]
        if isinstance(arr, list) and arr:
            iso = arr[0].get("fmt")
            if iso:
                return date.fromisoformat(iso)
    except Exception:
        pass
    return None

def upcoming_earnings_next_ndays(tickers: List[str], asof: date, days: int = 14) -> List[Dict[str, str]]:
    """Return [{ticker, date, when}], restricted to [asof, asof+days]."""
    end = asof + timedelta(days=days)
    out: List[Dict[str, str]] = []
    for i, t in enumerate(sorted(set(tickers))):
        if not t or t.upper() == "SPY":
            continue
        _sleep_between_calls(i)
        cal = _fetch_calendar(t)
        if not cal:
            continue
        d = _parse_date(cal)
        if not d:
            continue
        if asof <= d <= end:
            out.append({"ticker": t, "date": d.isoformat(), "when": _parse_when(cal)})
    out.sort(key=lambda x: (x["date"], x["ticker"]))
    return out

def upcoming_earnings_next_14d(tickers: List[str], asof: date) -> List[Dict[str, str]]:
    """Primary entry. If nothing in 14d, expand to 21d and tag those with window=21d."""
    items = upcoming_earnings_next_ndays(tickers, asof, days=14)
    if items:
        return items
    items21 = upcoming_earnings_next_ndays(tickers, asof, days=21)
    for it in items21:
        it["window"] = "21d"
    return items21
