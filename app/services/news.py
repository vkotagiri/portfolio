# app/services/news.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

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


def get_market_and_holdings_news(top_tickers: List[str], days: int = 7) -> Dict[str, List[dict]]:
    """
    Fetch news for market indices (SPY, QQQ) and top portfolio holdings.
    Returns {ticker: [{headline, url, published_at, source, sentiment, ...}, ...]}
    Limited to reduce API calls.
    """
    # Market-wide tickers + top holdings (limit to 5 to conserve API calls)
    market_tickers = ["SPY", "QQQ"]
    tickers_to_fetch = market_tickers + top_tickers[:5]
    # Remove duplicates
    tickers_to_fetch = list(dict.fromkeys(tickers_to_fetch))
    
    if not tickers_to_fetch:
        return {}
    
    provider = AlphaVantageNews()
    time_from = _iso_days_ago_utc(days)
    
    try:
        raw = provider.fetch(tickers_to_fetch, time_from_iso=time_from, limit=30)
    except Exception:
        return {}
    
    by_t: Dict[str, List[dict]] = {t: [] for t in tickers_to_fetch}
    for it in raw:
        tk = it.get("ticker")
        if tk in by_t:
            by_t[tk].append(it)
    
    # Keep top 3 per ticker by relevance
    for t in list(by_t.keys()):
        items = sorted(
            by_t[t],
            key=lambda x: (float(x.get("relevance") or 0.0), str(x.get("published_at") or "")),
            reverse=True,
        )[:3]
        by_t[t] = items
    
    return by_t


def format_news_for_llm(news_by_ticker: Dict[str, List[dict]]) -> str:
    """
    Format news data into a string for LLM consumption.
    """
    if not news_by_ticker:
        return "No recent news available."
    
    lines = []
    for ticker, items in news_by_ticker.items():
        if not items:
            continue
        lines.append(f"\n{ticker}:")
        for it in items[:3]:
            headline = it.get("headline", "")[:100]
            sentiment = it.get("sentiment", 0)
            sent_label = "positive" if sentiment and sentiment > 0.1 else ("negative" if sentiment and sentiment < -0.1 else "neutral")
            date = str(it.get("published_at", ""))[:10]
            lines.append(f"  - [{date}] {headline} (sentiment: {sent_label})")
    
    return "\n".join(lines) if lines else "No recent news available."
