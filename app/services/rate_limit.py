from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from sqlalchemy import select, func
from . import reporting  # for consistent timestamp style if needed
from ..db import get_session
from ..models import ApiCall

Z = timezone.utc

def _now_iso() -> str:
    return datetime.now(Z).strftime("%Y-%m-%dT%H:%M:%SZ")

def _iso(dt: datetime) -> str:
    return dt.astimezone(Z).strftime("%Y-%m-%dT%H:%M:%SZ")

def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=Z)

@dataclass
class Limits:
    hourly: int = 50
    daily: int = 1000
    unique_month: int = 500

class RateLimitExceeded(Exception):
    pass

class RateLimiter:
    """
    DB-backed sliding-window limiter + unique-symbol tracker.
    mode='raise' → raise RateLimitExceeded when exceeding limits.
    mode='sleep' → sleep until a safe boundary (conservative).
    """
    def __init__(self, provider: str, limits: Limits, mode: str = "raise", max_sleep_secs: int = 120):
        self.provider = provider
        self.limits = limits
        self.mode = mode
        self.max_sleep_secs = max_sleep_secs

    def _counts(self, sess, now: datetime, symbol: str | None):
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        month_start = _month_start(now)

        # Hourly count
        qh = select(func.count()).select_from(ApiCall).where(
            ApiCall.provider == self.provider,
            ApiCall.ts_utc >= _iso(hour_ago)
        )
        hourly = sess.scalar(qh) or 0

        # Daily count
        qd = select(func.count()).select_from(ApiCall).where(
            ApiCall.provider == self.provider,
            ApiCall.ts_utc >= _iso(day_ago)
        )
        daily = sess.scalar(qd) or 0

        # Unique symbols this month
        qm = select(func.count(func.distinct(ApiCall.symbol))).where(
            ApiCall.provider == self.provider,
            ApiCall.ts_utc >= _iso(month_start)
        )
        unique_month = sess.scalar(qm) or 0

        # already counted this symbol this month?
        already_symbol = False
        if symbol:
            qsm = select(func.count()).select_from(ApiCall).where(
                ApiCall.provider == self.provider,
                ApiCall.ts_utc >= _iso(month_start),
                ApiCall.symbol == symbol
            )
            already_symbol = (sess.scalar(qsm) or 0) > 0

        return hourly, daily, unique_month, already_symbol

    def acquire(self, symbol: str | None = None):
        now = datetime.now(Z)
        with get_session() as sess:
            hourly, daily, uniq_m, already_symbol = self._counts(sess, now, symbol)

            # Predict the counts *after* this call
            next_hourly = hourly + 1
            next_daily = daily + 1
            next_uniq = uniq_m if (already_symbol or not symbol) else (uniq_m + 1)

            # Enforce
            if next_hourly > self.limits.hourly:
                self._handle_block("hourly", now)
            if next_daily > self.limits.daily:
                self._handle_block("daily", now)
            if next_uniq > self.limits.unique_month:
                raise RateLimitExceeded(f"{self.provider}: unique monthly symbol limit exceeded ({next_uniq} > {self.limits.unique_month})")

    def _handle_block(self, kind: str, now: datetime):
        if self.mode == "sleep":
            # conservative: wait up to max_sleep_secs towards the boundary
            if kind == "hourly":
                next_window = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            else:
                # daily -> next midnight UTC
                next_window = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            wait = max(1, int((next_window - now).total_seconds()))
            time.sleep(min(wait, self.max_sleep_secs))
        else:
            raise RateLimitExceeded(f"{self.provider}: {kind} limit would be exceeded")

    def record(self, endpoint: str, symbol: str | None):
        with get_session() as sess:
            sess.add(ApiCall(provider=self.provider, endpoint=endpoint, symbol=symbol, ts_utc=_now_iso()))
            sess.commit()
