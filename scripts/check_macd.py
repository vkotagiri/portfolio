#!/usr/bin/env python
"""Check MACD crossover status for all holdings."""
from app.db import get_session
from app.models import Holding
from app.services.reporting import _series_for
from app.services.technicals import macd_12_26_9
from datetime import date

with get_session() as sess:
    holdings = sess.query(Holding).all()
    today = date(2026, 1, 27)
    
    print('MACD Crossover Check (last 3 days):')
    print('-' * 70)
    crossovers = []
    for h in sorted(holdings, key=lambda x: x.ticker):
        s = _series_for(sess, h.ticker, today, 60)
        if not s.empty:
            macd = macd_12_26_9(s, recent_sessions=3)
            if macd['recent_crossover']:
                crossovers.append((h.ticker, macd['recent_crossover_type'], macd['last_crossover']))
            print(f"{h.ticker:6} | Direction: {macd['direction']:8} | Last Cross: {macd['last_crossover']} | Recent: {macd['recent_crossover']}")
    
    print('-' * 70)
    if crossovers:
        print(f'\nStocks with crossovers in last 3 days:')
        for t, typ, dt in crossovers:
            print(f'  {t}: {typ} crossover on {dt}')
    else:
        print('\nNo crossovers in last 3 days for any stock.')
