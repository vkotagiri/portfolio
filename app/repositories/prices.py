from ..models import Price, Security, BenchmarkPrice
from sqlalchemy.orm import Session
from sqlalchemy import select

def upsert_prices(sess: Session, rows: list[dict]):
    for r in rows:
        t = r["ticker"].upper()
        if not sess.get(Security, t):
            sess.add(Security(ticker=t))
        key = {"ticker": t, "date": r["date"]}
        p = sess.get(Price, key)
        if p:
            p.close = r["close"]
            p.adj_close = r.get("adj_close")
            p.volume = r.get("volume")
            p.dividend = r.get("dividend")
            p.split = r.get("split")
            p.source = r.get("source","unknown")
            p.asof_ts = r.get("asof_ts","")
        else:
            sess.add(Price(**{**r, "ticker": t}))

def get_week_series(sess: Session, ticker: str, dates: list[str]):
    q = select(Price).where(Price.ticker==ticker.upper(), Price.date.in_(dates))
    return list(sess.scalars(q).all())

def upsert_benchmark(sess: Session, rows: list[dict]):
    for r in rows:
        key = {"symbol": r["symbol"], "date": r["date"]}
        b = sess.get(BenchmarkPrice, key)
        if b:
            b.adj_close = r.get("adj_close")
            b.source = r.get("source","unknown")
            b.asof_ts = r.get("asof_ts","")
        else:
            sess.add(BenchmarkPrice(**r))
