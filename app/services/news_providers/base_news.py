# app/services/news_providers/base_news.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TypedDict


class NewsItem(TypedDict, total=False):
    """Standard news item structure."""
    ticker: str
    headline: str
    url: str
    published_at: str  # ISO format
    source: str
    summary_raw: str
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    bullet: str  # LLM-generated summary


class BaseNewsProvider(ABC):
    """Abstract base class for news providers."""
    
    @abstractmethod
    def fetch(self, tickers: List[str], time_from_iso: str, limit: int = 50) -> List[NewsItem]:
        """
        Fetch news items for given tickers.
        
        Args:
            tickers: List of ticker symbols to fetch news for
            time_from_iso: Start time in ISO format
            limit: Maximum number of items to return
            
        Returns:
            List of NewsItem dictionaries
        """
        pass
