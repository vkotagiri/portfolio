# app/exceptions.py
"""
Custom exceptions for the portfolio application.
Provides specific error types for different failure modes.
"""
from __future__ import annotations


class PortfolioError(Exception):
    """Base exception for portfolio application errors."""
    pass


class DataError(PortfolioError):
    """Raised when there's an issue with data quality or availability."""
    pass


class PriceNotFoundError(DataError):
    """Raised when price data is not available for a ticker/date."""
    def __init__(self, ticker: str, date: str):
        self.ticker = ticker
        self.date = date
        super().__init__(f"No price found for {ticker} on {date}")


class InsufficientDataError(DataError):
    """Raised when there's not enough data for a calculation."""
    def __init__(self, metric: str, required: int, available: int):
        self.metric = metric
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {metric}: need {required} samples, have {available}"
        )


class ValidationError(PortfolioError):
    """Raised when validation fails."""
    pass


class TradeError(PortfolioError):
    """Raised when a trade operation fails."""
    pass


class InsufficientSharesError(TradeError):
    """Raised when trying to sell more shares than owned."""
    def __init__(self, ticker: str, requested: float, available: float):
        self.ticker = ticker
        self.requested = requested
        self.available = available
        super().__init__(
            f"Cannot sell {requested} shares of {ticker}, only {available} available"
        )


class ProviderError(PortfolioError):
    """Raised when a data provider fails."""
    pass


class RateLimitError(ProviderError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, provider: str, retry_after: int | None = None):
        self.provider = provider
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {provider}"
        if retry_after:
            msg += f", retry after {retry_after}s"
        super().__init__(msg)


class ConfigurationError(PortfolioError):
    """Raised when configuration is invalid or missing."""
    pass


class APIKeyMissingError(ConfigurationError):
    """Raised when a required API key is not configured."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"API key not configured for {provider}")
