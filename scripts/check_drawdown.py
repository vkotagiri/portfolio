#!/usr/bin/env python
"""Check portfolio drawdown."""
from app.db import get_session
from app.models import Price, Holding
from datetime import date, timedelta
import pandas as pd

with get_session() as sess:
    holdings = sess.query(Holding).all()
    tickers = [h.ticker for h in holdings]
    shares_map = {h.ticker: h.shares for h in holdings}
    
    today = date(2026, 1, 28)
    start = today - timedelta(days=365)  # Full year
    
    prices = sess.query(Price).filter(
        Price.ticker.in_(tickers),
        Price.date >= start.isoformat()
    ).all()
    
    print(f"Loaded {len(prices)} price records")
    
    data = {}
    for p in prices:
        if p.ticker not in data:
            data[p.ticker] = {}
        data[p.ticker][p.date] = p.close
    
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Trading days: {len(df)}")
    
    pv = pd.Series(dtype=float)
    for dt in df.index:
        total = sum(shares_map.get(t, 0) * df.loc[dt, t] for t in tickers if t in df.columns and pd.notna(df.loc[dt, t]))
        if total > 0:
            pv[dt] = total
    
    print(f"\nPortfolio value range: ${pv.min():,.2f} - ${pv.max():,.2f}")
    
    print("\nRecent Portfolio Values:")
    for dt, val in pv.tail(10).items():
        print(f"{dt.date()} | ${val:,.2f}")
    
    peak = pv.cummax()
    dd = (pv / peak - 1) * 100
    
    # Find when max drawdown occurred
    min_dd_date = dd.idxmin()
    peak_before_dd = pv[:min_dd_date].idxmax()
    
    print(f"\nCurrent drawdown: {dd.iloc[-1]:.2f}%")
    print(f"Max drawdown (full period): {dd.min():.2f}%")
    print(f"Max DD occurred on: {min_dd_date.date()}")
    print(f"Peak before max DD: {peak_before_dd.date()} (${pv[peak_before_dd]:,.2f})")
    print(f"Current peak value: ${peak.max():,.2f}")
