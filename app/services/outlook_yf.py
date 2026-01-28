# app/services/outlook_yf.py
from __future__ import annotations
import time
import logging
import signal
from typing import Dict, List, Tuple

try:
    import yfinance as yf
except Exception:
    yf = None

logger = logging.getLogger(__name__)

DEFAULT_RPM = 60   # faster rate - yfinance can handle it
DEFAULT_MAX = 10   # reduced cap per run to speed up
TICKER_TIMEOUT = 3  # seconds per ticker lookup

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Ticker fetch timed out")

def fetch_next_earnings(
    tickers: List[str],
    *,
    rpm: int = DEFAULT_RPM,
    max_calls: int = DEFAULT_MAX
) -> Dict[str, Tuple[str | None, str]]:
    """
    Returns {ticker: (iso_date_or_None, source)} for the next earnings date.
    Throttled to ~rpm and capped by max_calls to avoid 429s.
    No exceptions leak out: failures become (None, '...-error').
    """
    if yf is None:
        return {t: (None, "Data not available") for t in tickers}

    out: Dict[str, Tuple[str | None, str]] = {}
    # seconds between calls (rpm ~ requests/min)
    interval = max(60.0 / max(rpm, 1), 0.3)
    calls = 0
    
    # Limit tickers to process
    tickers_to_process = tickers[:max_calls]
    
    # Store original signal handler
    original_handler = signal.getsignal(signal.SIGALRM)

    for t in tickers_to_process:
        if calls >= max_calls:
            out[t] = (None, "throttle-cap")
            continue
        
        try:
            # Set up timeout using signal (only works on Unix)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TICKER_TIMEOUT)
            
            try:
                edf = yf.Ticker(t).get_earnings_dates(limit=1)
                signal.alarm(0)  # Cancel alarm
                calls += 1
                if edf is not None and not edf.empty:
                    d = edf.index[0].date().isoformat()
                    out[t] = (d, "yfinance")
                else:
                    out[t] = (None, "yfinance")
            except TimeoutError:
                logger.debug(f"Timeout fetching earnings for {t}")
                out[t] = (None, "timeout")
            except Exception as e:
                logger.debug(f"Error fetching earnings for {t}: {e}")
                out[t] = (None, "yfinance-error")
            finally:
                signal.alarm(0)  # Ensure alarm is cancelled
        except Exception:
            out[t] = (None, "yfinance-error")
        
        time.sleep(interval)

    # Restore original signal handler
    signal.signal(signal.SIGALRM, original_handler)

    # Fill in remaining tickers that weren't processed
    for t in tickers:
        if t not in out:
            out[t] = (None, "throttle-cap")

    return out
