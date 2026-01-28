# app/services/news_providers/__init__.py
from .base_news import BaseNewsProvider, NewsItem
from .alpha_vantage import AlphaVantageNews

__all__ = ["BaseNewsProvider", "NewsItem", "AlphaVantageNews"]
