#!/usr/bin/env python
"""Debug Alpha and Sharpe Ratio calculations."""
import numpy as np
import pandas as pd
from datetime import date, timedelta

from app.db import get_session
from app.services.reporting import (
    _series_for, _spy_series, _build_portfolio_value_series, 
    _beta_alpha_ols
)
from app.repositories.holdings import all_holdings
from app.services.risk_free import latest_1w_tbill

week_end = date(2026, 1, 27)

with get_session() as sess:
    holds = all_holdings(sess)
    tickers = [h.ticker for h in holds]
    shares_map = {h.ticker: float(h.shares) for h in holds}
    
    # Get SPY
    spy = _spy_series(sess, week_end, 260)
    print(f"SPY data points: {len(spy)}")
    
    # Get portfolio value series
    pv_60 = _build_portfolio_value_series(sess, shares_map, tickers, week_end, 80)
    print(f"Portfolio value (60d) data points: {len(pv_60)}")
    
    # Calculate returns
    pr60 = pv_60.pct_change(fill_method=None).dropna()
    sr60 = spy.reindex(pr60.index).pct_change(fill_method=None).dropna()
    idx = pr60.index.intersection(sr60.index)
    pr60 = pr60.loc[idx]
    sr60 = sr60.loc[idx]
    
    print(f"\nAligned return series: {len(pr60)} days")
    print(f"Portfolio daily return mean: {pr60.mean()*100:.4f}%")
    print(f"Portfolio daily return std:  {pr60.std()*100:.4f}%")
    print(f"SPY daily return mean:       {sr60.mean()*100:.4f}%")
    print(f"SPY daily return std:        {sr60.std()*100:.4f}%")
    
    # Annualized volatility
    sigma60 = float(pr60.std() * np.sqrt(252) * 100.0)
    print(f"\nPortfolio annualized volatility (sigma60): {sigma60:.2f}%")
    
    # Beta and Alpha (CAPM)
    ba = _beta_alpha_ols(pr60, sr60)
    if ba:
        beta, alpha_daily, r2 = ba
        alpha_annual = alpha_daily * 252.0 * 100.0  # Convert to annual %
        print(f"\n=== CAPM Regression (60 days) ===")
        print(f"Beta:              {beta:.4f}")
        print(f"Alpha (daily):     {alpha_daily*100:.6f}%")
        print(f"Alpha (annual):    {alpha_annual:.2f}%")
        print(f"R-squared:         {r2:.4f}")
        
        # Verify manually
        print(f"\n--- Manual verification ---")
        excess_port = pr60 - sr60 * beta  # Portfolio return - beta * market return
        manual_alpha_daily = excess_port.mean()
        print(f"Manual alpha (daily): {manual_alpha_daily*100:.6f}%")
        print(f"Manual alpha (annual): {manual_alpha_daily*252*100:.2f}%")
    
    # Sharpe Ratio
    rf_dict = latest_1w_tbill(week_end)
    rf_val = rf_dict.get("value")
    print(f"\n=== Sharpe Ratio Calculation ===")
    print(f"Risk-free rate (annual %): {rf_val}")
    
    if rf_val is not None:
        # rf_val is already annualized (e.g., 5.25 means 5.25% per year)
        rf_annual = float(rf_val) / 100.0  # Convert from percent to decimal
        rf_daily = rf_annual / 252.0  # Convert annual to daily
        print(f"Risk-free rate (daily):    {rf_daily*100:.6f}%")
        print(f"Risk-free rate (annual):   {rf_annual*100:.2f}%")
        
        ex = pr60 - rf_daily  # Excess returns
        print(f"\nExcess return mean (daily): {ex.mean()*100:.6f}%")
        print(f"Excess return mean (annual): {ex.mean()*252*100:.2f}%")
        
        sharpe = (ex.mean() / pr60.std()) * np.sqrt(252.0)
        print(f"\nSharpe Ratio: {sharpe:.2f}")
        
        # Alternative calculation
        annual_excess = ex.mean() * 252
        annual_vol = pr60.std() * np.sqrt(252)
        sharpe_alt = annual_excess / annual_vol
        print(f"Sharpe (alt calc): {sharpe_alt:.2f}")
    
    # Check for contradictions
    print("\n" + "="*50)
    print("ANALYSIS:")
    print("="*50)
    
    if ba and rf_val:
        # Alpha measures excess return vs CAPM prediction (benchmark-adjusted)
        # Sharpe measures excess return vs risk-free (not benchmark-adjusted)
        
        # If portfolio outperforms the CAPM prediction, alpha is positive
        # If portfolio return > risk-free rate, Sharpe is positive
        
        port_annual_return = pr60.mean() * 252 * 100
        spy_annual_return = sr60.mean() * 252 * 100
        rf_annual_pct = rf_annual * 100
        
        print(f"Portfolio annual return:  {port_annual_return:.2f}%")
        print(f"SPY annual return:        {spy_annual_return:.2f}%")
        print(f"Risk-free annual:         {rf_annual_pct:.2f}%")
        print(f"")
        print(f"Portfolio excess vs Rf:   {port_annual_return - rf_annual_pct:.2f}%")
        print(f"CAPM expected return:     {rf_annual_pct + beta * (spy_annual_return - rf_annual_pct):.2f}%")
        print(f"Alpha (outperformance):   {alpha_annual:.2f}%")
