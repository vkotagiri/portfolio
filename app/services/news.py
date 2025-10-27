# app/services/news.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from .news_providers.alpha_vantage import AlphaVantageNews
from .llm_summarize import summarize_news_for_ticker


def _iso_days_ago_utc(days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    # Alpha Vantage expects YYYYMMDDThhmmss (no 'Z' needed here)
    return dt.strftime("%Y%m%dT000000")


def get_top_news_last_7d(tickers: List[str]) -> Dict[str, List[dict]]:
    """
    Returns {ticker: [ {headline, url, published_at, source, bullet?, ...}, ... ] }
    Keeps ~top 5 per ticker by relevance/recency; attaches up to 3 LLM bullets.
    """
    if not tickers:
        return {}

    provider = AlphaVantageNews()
    time_from = _iso_days_ago_utc(7)
    raw = provider.fetch(tickers, time_from_iso=time_from, limit=50)

    by_t: Dict[str, List[dict]] = {t: [] for t in tickers}
    for it in raw:
        tk = it.get("ticker")
        if tk in by_t:
            by_t[tk].append(it)

    # Keep 5, then attach bullets
    for t in list(by_t.keys()):
        items = sorted(
            by_t[t],
            key=lambda x: (float(x.get("relevance") or 0.0), str(x.get("published_at") or "")),
            reverse=True,
        )[:5]
        by_t[t] = items
        bullets = summarize_news_for_ticker(t, items, max_items=3)
        for i in range(min(len(items), len(bullets))):
            items[i]["bullet"] = bullets[i]

    return by_t
