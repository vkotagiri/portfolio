# app/services/providers/yf.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict
import contextlib

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # will raise at call time


class YFProvider:
    """
    Lightweight yfinance provider used as a fallback/secondary source.
    - No settings access at module import time (to avoid AttributeError)
    - Graceful defaults if config fields are missing
    - Returns rows: [{"symbol","date","adj_close","source","asof_ts"}, ...]
    """

    def __init__(
        self,
        rpm: int | None = None,
        burst: int | None = None,
        parallel: int | None = None,
        backoff_base: float | None = None,
        debug: bool | None = None,
    ):
        # Pull config lazily and safely
        try:
            from app.config import settings  # local import to avoid early AttributeError
        except Exception:
            settings = None

        self.rpm = rpm if rpm is not None else (getattr(settings, "yf_rpm", 240) if settings else 240)
        self.burst = burst if burst is not None else (getattr(settings, "yf_burst", 10) if settings else 10)
        self.parallel = parallel if parallel is not None else (getattr(settings, "yf_parallel", 5) if settings else 5)
        self.backoff_base = backoff_base if backoff_base is not None else (getattr(settings, "yf_backoff_base", 1.5) if settings else 1.5)
        self.debug = debug if debug is not None else (getattr(settings, "yf_debug", False) if settings else False)

    # --- public API expected by ingestion ---
    def fetch_history(self, ticker: str, start: str, end: str) -> List[Dict]:
        """
        Fetch daily adjusted-close (auto_adjust=True => 'Close' is adjusted).
        start/end are ISO strings inclusive; yfinance end is exclusive, so +1 day.
        """
        if yf is None:
            raise RuntimeError("yfinance not installed")

        try:
            # yfinance requires end to be exclusive; add 1 day
            end_exc = (pd.Timestamp(end) + timedelta(days=1)).date().isoformat()
            asof = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Single-ticker, 1d interval, adjusted
            df = yf.Ticker(ticker).history(
                start=start,
                end=end_exc,
                interval="1d",
                auto_adjust=True,
                actions=False,
                prepost=False,
            )

            if df is None or df.empty:
                return []

            # yfinance with auto_adjust=True gives adjusted prices in "Close"
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if close_col not in df.columns:
                return []

            rows: List[Dict] = []
            for idx, row in df.iterrows():
                try:
                    px = float(row[close_col])
                except Exception:
                    continue
                d = pd.Timestamp(idx).date().isoformat()
                rows.append(
                    {
                        "symbol": ticker,
                        "date": d,
                        "adj_close": px,
                        "source": "yfinance",
                        "asof_ts": asof,
                    }
                )
            return rows

        except Exception as e:
            if self.debug:
                print(f"[yfinance] fetch_history error for {ticker} {start}->{end}: {e}")
            return []
