# app/services/providers/tiingo.py
from __future__ import annotations

import os
import time
import random
import requests
from datetime import datetime, timezone

from ..rate_limit import RateLimiter, Limits, RateLimitExceeded
from .base import RateLimitProviderError

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"
Z = timezone.utc


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


class TiingoProvider:
    """Primary price provider using Tiingo daily/adjClose endpoints."""
    name = "tiingo"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        self.debug = os.getenv("TIINGO_DEBUG", "0") == "1"

        # Free-tier defaults; overridable via .env
        hourly = _env_int("TIINGO_HOURLY", 50)
        daily = _env_int("TIINGO_DAILY", 1000)
        uniq = _env_int("TIINGO_UNIQUE_MONTH", 500)
        mode = os.getenv("TIINGO_RL_MODE", "sleep")  # 'sleep' (recommended) or 'raise'
        max_sleep = _env_int("TIINGO_MAX_SLEEP", 120)
        self.limiter = RateLimiter(self.name, Limits(hourly, daily, uniq), mode=mode, max_sleep_secs=max_sleep)

    # ---------- internal helpers ----------

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError("TIINGO_API_KEY missing")
        return {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "agentic-portfolio/1.0",
        }

    def _request(self, url: str, params: dict, symbol: str) -> requests.Response:
        # local preflight limiter
        try:
            self.limiter.acquire(symbol)
        except RateLimitExceeded as e:
            if self.limiter.mode == "sleep":
                time.sleep(min(2, self.limiter.max_sleep_secs))
            else:
                raise RateLimitProviderError(str(e))

        backoff = 2.0
        total_sleep = 0.0
        max_total_sleep = _env_int("TIINGO_MAX_TOTAL_SLEEP", 900)  # 15 min
        last_exc: Exception | None = None

        for attempt in range(10):
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=30)
            except Exception as e:
                last_exc = e
                time.sleep(backoff)
                total_sleep += backoff
                print(f"[tiingo] network error for {symbol}: {e}")
                backoff = min(backoff * 2, self.limiter.max_sleep_secs)
                if total_sleep >= max_total_sleep:
                    raise RateLimitProviderError(f"Tiingo network retry budget exceeded: {e}")
                continue

            # 429 Too Many Requests
            if r.status_code == 429:
                if self.debug:
                    print(f"[tiingo] 429 for {symbol}")
                try:
                    retry_after = int(r.headers.get("Retry-After", "0"))
                except Exception:
                    retry_after = 0

                if self.limiter.mode == "sleep":
                    wait = retry_after if retry_after > 0 else backoff
                    wait = min(wait, self.limiter.max_sleep_secs)
                    time.sleep(wait)
                    total_sleep += wait
                    backoff = min(backoff * 2, self.limiter.max_sleep_secs)
                    if total_sleep >= max_total_sleep:
                        raise RateLimitProviderError("Tiingo 429 throttle exceeded sleep budget")
                    continue
                else:
                    if attempt >= 2:
                        raise RateLimitProviderError("Tiingo 429 Too Many Requests")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.limiter.max_sleep_secs)
                    continue

            # transient 5xx
            if 500 <= r.status_code < 600:
                time.sleep(backoff)
                total_sleep += backoff
                backoff = min(backoff * 2, self.limiter.max_sleep_secs)
                if total_sleep >= max_total_sleep:
                    raise RateLimitProviderError(f"Tiingo server retry budget exceeded: {r.status_code}")
                continue

            # auth issues: bubble immediately
            if r.status_code in (401, 403):
                if self.debug:
                    print(f"[tiingo] auth failed {symbol}: {r.status_code} {r.text[:160]}")
                r.raise_for_status()

            r.raise_for_status()
            self.limiter.record(endpoint="prices", symbol=symbol)
            return r

        if last_exc:
            raise RateLimitProviderError(f"Tiingo network error: {last_exc}")
        raise RateLimitProviderError("Tiingo: exhausted retries without success")

    # ---------- public API expected by ingestion ----------

    def fetch_history(self, ticker: str, start: str, end: str) -> list[dict]:
        """Return daily rows between start/end (inclusive) with adj_close."""
        url = f"{TIINGO_BASE}/{ticker}/prices"
        params = {
            "startDate": start,
            "endDate": end,
            "columns": "close,adjClose,volume,divCash,splitFactor",
        }
        r = self._request(url, params, ticker.upper())
        data = r.json() or []
        asof = datetime.now(Z).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows: list[dict] = []
        for d in data:
            dt = (d.get("date") or "")[:10]
            if not dt:
                continue
            rows.append({
                "ticker": ticker.upper(),
                "date": dt,
                "close": float(d.get("close")) if d.get("close") is not None else None,
                "adj_close": float(d.get("adjClose")) if d.get("adjClose") is not None else None,
                "volume": int(d.get("volume")) if d.get("volume") is not None else None,
                "dividend": float(d.get("divCash")) if d.get("divCash") is not None else None,
                "split": float(d.get("splitFactor")) if d.get("splitFactor") is not None else None,
                "source": self.name,
                "asof_ts": asof,
            })
        return rows

    def fetch_eod(self, ticker: str, date_str: str) -> list[dict]:
        """Convenience: same as history for a single date."""
        return self.fetch_history(ticker, date_str, date_str)
