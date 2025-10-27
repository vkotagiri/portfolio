# app/utils/rate_limit.py
from __future__ import annotations
import time
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class _DayWindow:
    day_epoch: int
    count: int


class RateLimiter:
    """
    Token-bucket limiter with optional daily-quota guard.

    - rpm: tokens added per minute (rate)
    - burst: bucket capacity (max tokens)
    - rpd: optional daily cap (UTC day)

    Usage:
        limiter = RateLimiter(rpm=5, burst=5, rpd=25)
        limiter.wait()  # blocks until a token is available and daily cap allows
    """
    def __init__(self, rpm: int, burst: int, rpd: Optional[int] = None):
        assert rpm > 0 and burst > 0
        self._rps = float(rpm) / 60.0
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self._rpd = rpd
        self._day = _DayWindow(day_epoch=self._day_epoch(), count=0)

    def _day_epoch(self) -> int:
        return int(time.time() // 86400)

    def _maybe_reset_day(self):
        d = self._day_epoch()
        if d != self._day.day_epoch:
            self._day = _DayWindow(day_epoch=d, count=0)

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rps)

    def wait(self):
        """Block until a token is available and daily quota (if any) allows."""
        with self._lock:
            self._maybe_reset_day()
            if self._rpd is not None and self._day.count >= self._rpd:
                # Sleep until UTC midnight
                secs_to_midnight = 86400 - (time.time() % 86400)
                time.sleep(secs_to_midnight)

            self._refill()
            if self._tokens < 1.0:
                need = 1.0 - self._tokens
                sleep_s = need / self._rps if self._rps > 0 else 1.0
                time.sleep(max(0.0, sleep_s))
                self._refill()

            self._tokens -= 1.0
            self._day.count += 1

    def backoff(self, seconds: float = 10.0):
        """Helper for 429 or network pressure."""
        time.sleep(max(0.0, seconds))
