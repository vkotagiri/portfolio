# app/constants.py
"""
Centralized constants for the portfolio application.
Eliminates magic numbers and makes configuration explicit.
"""
from __future__ import annotations

# Trading calendar constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_WEEK = 5

# Default lookback periods (in trading days)
LOOKBACK_1W = 5
LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126
LOOKBACK_1Y = 252

# Risk metrics defaults
DEFAULT_LOOKBACK_DAYS = 260  # Slightly more than 1Y for buffer
MIN_SAMPLES_SHORT = 30      # Minimum for 60-day metrics
MIN_SAMPLES_LONG = 100      # Minimum for 252-day metrics

# Technical indicators
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Position limits
MAX_POSITION_WEIGHT = 0.25   # 25% max per position
CONCENTRATION_WARNING = 0.07 # Warn if >7%

# API rate limits
DEFAULT_API_TIMEOUT = 30     # seconds
MAX_RETRIES = 3

# Report generation
RECENT_CROSSOVER_DAYS = 3    # Days to consider "recent" for MACD
