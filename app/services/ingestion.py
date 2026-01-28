# app/services/ingestion.py
import os
from datetime import date
from typing import List, Sequence
import logging

from sqlalchemy.sql import text

from ..db import get_session
from ..repositories.holdings import all_holdings
from ..repositories.prices import upsert_prices, upsert_benchmark

# Providers & errors
from .providers.base import RateLimitProviderError
from .providers.mock_provider import MockProvider
from .providers.yf import YFProvider
import time
TIINGO_THROTTLE = float(os.getenv("TIINGO_THROTTLE_SEC", "0.4"))

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO


# Optional Tiingo (primary, rate-limited). Try/except lets you run without it.
try:
    from .providers.tiingo import TiingoProvider
    HAVE_TIINGO = True
except Exception:
    HAVE_TIINGO = False


def _provider_chain() -> Sequence[object]:
    """
    Build the provider chain based on env flags & available keys.
    OFFLINE=1  -> Mock only (deterministic synthetic prices)
    OFFLINE!=1 -> Tiingo (if key present) -> yfinance -> (optional Mock)
    """
    if os.getenv("OFFLINE", "0") == "1":
        return [MockProvider()]

    chain: List[object] = []
    if HAVE_TIINGO and os.getenv("TIINGO_API_KEY"):
        chain.append(TiingoProvider())        # requires TIINGO_API_KEY
    chain.append(YFProvider())                # free fallback
    # chain.append(MockProvider())            # uncomment if you want last-ditch demo data
    return chain


def _chain_history(ticker: str, start: str, end: str) -> List[dict]:
    """
    Try providers in order and return the first set of rows that includes adj_close values.
    """
    for provider in _provider_chain():
        try:
            rows = provider.fetch_history(ticker, start, end)
            if rows and all(r.get("adj_close") is not None for r in rows):
                return rows
        except RateLimitProviderError:
            # Provider is over its limit; try the next one.
            continue
        except Exception:
            # Any other provider-specific error; try the next one.
            continue
    return []


def ingest_eod_for_date(target_date: date):
    """
    Ingest one trading day for all holdings (+ SPY benchmark).
    Idempotent via upserts; safe to re-run for the same date.
    """
    ds = target_date.strftime("%Y-%m-%d")

    with get_session() as sess:
        tickers = [h.ticker for h in all_holdings(sess)]
        if not tickers:
            return {"status": "no-holdings", "date": ds}

        # Holdings prices
        all_rows: List[dict] = []
        for t in tickers:
            rows = _chain_history(t, ds, ds)
            all_rows.extend(rows)
            time.sleep(TIINGO_THROTTLE)

        # Benchmark (SPY)
        spy_rows = _chain_history("SPY", ds, ds)
        bench_rows = [
            {
                "symbol": "SPY",
                "date": r["date"],
                "adj_close": r["adj_close"],
                "source": r["source"],
                "asof_ts": r["asof_ts"],
            }
            for r in spy_rows
        ]

        # Upserts (idempotent)
        upsert_prices(sess, all_rows)
        upsert_benchmark(sess, bench_rows)
        sess.commit()

        return {
            "status": "ok",
            "date": ds,
            "count": len(all_rows) + len(bench_rows),
            "providers": sorted(
                {r.get("source") for r in (all_rows + bench_rows) if r.get("source")}
            ),
        }


def backfill_history(start: date, end: date):
    """
    Backfill a date range (inclusive) for all holdings + SPY.
    Useful after first setup so 50D/200D/RSI/MACD can compute.
    """
    s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    with get_session() as sess:
        tickers = [h.ticker for h in all_holdings(sess)]
        if not tickers:
            logger.info("No holdings found. Skipping backfill.")
            return {"status": "no-holdings", "start": s, "end": e}

        total = 0

        # Holdings
        for t in tickers:
            # Check if data already exists for the ticker and date range
            existing_data = sess.execute(
                text(
                    """
                    SELECT COUNT(*) FROM prices
                    WHERE ticker = :ticker AND date BETWEEN :start AND :end
                    """
                ),
                {"ticker": t, "start": s, "end": e}
            ).scalar()

            if existing_data > 0:
                logger.info(f"Skipping {t}: Data already exists for {s} to {e}.")
                continue  # Skip if data already exists

            try:
                rows = _chain_history(t, s, e)
                if rows:
                    upsert_prices(sess, rows)
                    total += len(rows)
                    logger.info(f"Fetched and inserted {len(rows)} rows for {t}.")
                    sess.commit()

                else:
                    logger.warning(f"No data fetched for {t} from {s} to {e}.")
            except Exception as ex:
                logger.error(f"Error fetching data for {t}: {ex}")

            time.sleep(TIINGO_THROTTLE)

        # Benchmark (SPY)
        spy_existing_data = sess.execute(
            text(
                """
                SELECT COUNT(*) FROM prices
                WHERE ticker = 'SPY' AND date BETWEEN :start AND :end
                """
            ),
            {"start": s, "end": e}
        ).scalar()

        if spy_existing_data > 0:
            logger.info(f"Skipping SPY: Data already exists for {s} to {e}.")
        else:
            try:
                spy_rows = _chain_history("SPY", s, e)
                if spy_rows:
                    upsert_benchmark(
                        sess,
                        [
                            {
                                "symbol": "SPY",
                                "date": r["date"],
                                "adj_close": r["adj_close"],
                                "source": r["source"],
                                "asof_ts": r["asof_ts"],
                            }
                            for r in spy_rows
                        ],
                    )
                    total += len(spy_rows)
                    logger.info(f"Fetched and inserted {len(spy_rows)} rows for SPY.")
                else:
                    logger.warning(f"No data fetched for SPY from {s} to {e}.")
            except Exception as ex:
                logger.error(f"Error fetching data for SPY: {ex}")

        sess.commit()
        logger.info(f"Backfill completed: {total} rows inserted from {s} to {e}.")
        return {"status": "ok", "start": s, "end": e, "rows": total}
