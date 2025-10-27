# app/services/news_providers/alpha_vantage.py
from __future__ import annotations
import time
from typing import List, Dict
from datetime import datetime, timezone
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .base_news import BaseNewsProvider, NewsItem
from ...config import settings
from ...utils.rate_limit import RateLimiter

ALPHA_NEWS_URL = "https://www.alphavantage.co/query"

# Module-level limiter (conservative defaults; override via .env)
_AV_LIMITER = RateLimiter(
    rpm=settings.alphavantage_rpm,
    burst=settings.alphavantage_burst,
    rpd=settings.alphavantage_rpd,
)


class AlphaVantageNews(BaseNewsProvider):
    """
    Alpha Vantage NEWS_SENTIMENT
    - Free tier: ~5 req/min; ~25 req/day
    - We batch up to 10 tickers per call and throttle.
    """
    def __init__(self, api_key: str | None = None, session: httpx.Client | None = None):
        self.api_key = api_key or settings.alphavantage_api_key
        self.client = session or httpx.Client(timeout=20)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(1, 4))
    def _call(self, params: Dict[str, str]) -> dict:
        _AV_LIMITER.wait()
        r = self.client.get(ALPHA_NEWS_URL, params=params)
        if r.status_code == 429:
            # back off politely; retry via tenacity wrapper
            _AV_LIMITER.backoff(15)
        r.raise_for_status()
        return r.json()

    def fetch(self, tickers: list[str], time_from_iso: str, limit: int = 50) -> List[NewsItem]:
        if not self.api_key or not tickers:
            return []
        out: List[NewsItem] = []
        batch_size = 10

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ",".join(batch),
                # Alpha Vantage expects YYYYMMDDThhmmss (UTC)
                "time_from": time_from_iso.replace("-", "").replace(":", "").replace("Z", ""),
                "limit": str(limit),
                "apikey": self.api_key,
            }
            data = self._call(params)
            feed = data.get("feed", []) or []

            for item in feed:
                title = item.get("title")
                url = item.get("url")
                src = item.get("source")
                ts = item.get("time_published")  # e.g., "20251025T154500"
                try:
                    dt = datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                    published_iso = dt.isoformat().replace("+00:00", "Z")
                except Exception:
                    published_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

                overall = item.get("overall_sentiment_score")
                summary_raw = item.get("summary")
                tlist = item.get("ticker_sentiment", []) or []

                for t in tlist:
                    tk = t.get("ticker")
                    if not tk or tk not in batch:
                        continue
                    rel = float(t.get("relevance_score") or 0.0)
                    if rel < 0.2:
                        continue
                    out.append({
                        "ticker": tk,
                        "headline": title,
                        "url": url,
                        "published_at": published_iso,
                        "source": src,
                        "summary_raw": summary_raw,
                        "sentiment": overall,
                        "relevance": rel,
                    })

            # Light pacing (in addition to token bucket) to be gentle
            time.sleep(1.0)

        return out
