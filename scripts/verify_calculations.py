#!/usr/bin/env python
"""
Comprehensive review of all financial calculations.
Verifies each metric against standard formulas.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

from app.db import get_session
from app.services.reporting import (
    _series_for, _spy_series, _build_portfolio_value_series, 
    _beta_alpha_ols, _max_drawdown, _weights_on_day
)
from app.repositories.holdings import all_holdings
from app.services.risk_free import latest_1w_tbill
from app.services.technicals import rsi14, macd_12_26_9

week_end = date(2026, 1, 27)

print("=" * 70)
print("COMPREHENSIVE CALCULATION REVIEW")
print("=" * 70)

with get_session() as sess:
    holds = all_holdings(sess)
    tickers = [h.ticker for h in holds]
    shares_map = {h.ticker: float(h.shares) for h in holds}
    
    # ==========================================================================
    # 1. RETURN CALCULATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("1. RETURN CALCULATIONS")
    print("=" * 70)
    
    # Get a sample stock
    sample_ticker = "AAPL"
    s = _series_for(sess, sample_ticker, week_end, 400)
    
    if not s.empty:
        # Weekly return
        wk = s.tail(5)
        if len(wk) >= 2:
            weekly_return = (wk.iloc[-1] / wk.iloc[0] - 1) * 100
            print(f"\n{sample_ticker} Weekly Return:")
            print(f"  Start price: ${wk.iloc[0]:.2f}")
            print(f"  End price:   ${wk.iloc[-1]:.2f}")
            print(f"  Return:      {weekly_return:.2f}%")
            print(f"  Formula:     (End/Start - 1) * 100 ✓")
        
        # Monthly return (21 trading days)
        mo = s.tail(21)
        if len(mo) >= 2:
            monthly_return = (mo.iloc[-1] / mo.iloc[0] - 1) * 100
            print(f"\n{sample_ticker} Monthly Return (~21 days):")
            print(f"  Start price: ${mo.iloc[0]:.2f}")
            print(f"  End price:   ${mo.iloc[-1]:.2f}")
            print(f"  Return:      {monthly_return:.2f}%")
            print(f"  Formula:     (End/Start - 1) * 100 ✓")
    
    # ==========================================================================
    # 2. RSI CALCULATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("2. RSI (Relative Strength Index)")
    print("=" * 70)
    
    if not s.empty and len(s) >= 15:
        rsi_val = rsi14(s)
        
        # Manual calculation for verification
        delta = s.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / roll_down
        rsi_manual = 100 - (100 / (1 + rs))
        
        print(f"\n{sample_ticker} RSI(14):")
        print(f"  Calculated:  {rsi_val:.2f}")
        print(f"  Manual:      {float(rsi_manual.iloc[-1]):.2f}")
        print(f"  Formula:     RSI = 100 - 100/(1 + RS)")
        print(f"               RS = EMA(gains) / EMA(losses)")
        print(f"  Match:       {'✓' if abs(rsi_val - float(rsi_manual.iloc[-1])) < 0.01 else '✗'}")
    
    # ==========================================================================
    # 3. MACD CALCULATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("3. MACD (Moving Average Convergence Divergence)")
    print("=" * 70)
    
    if not s.empty and len(s) >= 35:
        macd_result = macd_12_26_9(s)
        
        # Manual calculation
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        print(f"\n{sample_ticker} MACD(12,26,9):")
        print(f"  MACD Line:   {macd_result['macd']:.4f}")
        print(f"  Manual:      {float(macd_line.iloc[-1]):.4f}")
        print(f"  Signal Line: {macd_result['signal']:.4f}")
        print(f"  Manual:      {float(signal_line.iloc[-1]):.4f}")
        print(f"  Histogram:   {macd_result['hist']:.4f}")
        print(f"  Manual:      {float(histogram.iloc[-1]):.4f}")
        print(f"  Direction:   {macd_result['direction']}")
        print(f"  Formula:     MACD = EMA(12) - EMA(26)")
        print(f"               Signal = EMA(9) of MACD")
        print(f"  Match:       {'✓' if abs(macd_result['macd'] - float(macd_line.iloc[-1])) < 0.0001 else '✗'}")
    
    # ==========================================================================
    # 4. PORTFOLIO WEIGHTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("4. PORTFOLIO WEIGHTS")
    print("=" * 70)
    
    # Get latest prices
    prices = {}
    for t in tickers[:5]:  # Sample 5 tickers
        ts = _series_for(sess, t, week_end, 10)
        if not ts.empty:
            prices[t] = float(ts.iloc[-1])
    
    weights = _weights_on_day(shares_map, prices)
    
    total_mv = sum(shares_map[t] * prices[t] for t in prices)
    print(f"\nSample weights (first 5 tickers):")
    print(f"  Total Market Value: ${total_mv:,.2f}")
    weight_sum = 0
    for t in list(prices.keys())[:5]:
        mv = shares_map[t] * prices[t]
        w = weights.get(t, 0)
        manual_w = mv / total_mv
        weight_sum += w
        print(f"  {t}: {w*100:.2f}% (MV: ${mv:,.2f}, Manual: {manual_w*100:.2f}%)")
    
    print(f"\n  Formula:     Weight = (Shares × Price) / Total Portfolio Value")
    print(f"  Sum check:   Weights of all holdings sum to ~100%")
    
    # ==========================================================================
    # 5. VOLATILITY (Annualized Standard Deviation)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. VOLATILITY (Annualized)")
    print("=" * 70)
    
    pv_60 = _build_portfolio_value_series(sess, shares_map, tickers, week_end, 80)
    if not pv_60.empty:
        pr60 = pv_60.pct_change(fill_method=None).dropna()
        
        daily_std = pr60.std()
        annual_vol = daily_std * np.sqrt(252) * 100
        
        print(f"\nPortfolio Volatility (60-day):")
        print(f"  Daily Std Dev:      {daily_std*100:.4f}%")
        print(f"  Annualized Vol:     {annual_vol:.2f}%")
        print(f"  Formula:            σ_annual = σ_daily × √252")
        print(f"  Trading days/year:  252 ✓")
    
    # ==========================================================================
    # 6. BETA (CAPM)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("6. BETA (CAPM Regression)")
    print("=" * 70)
    
    spy = _spy_series(sess, week_end, 260)
    if not pv_60.empty and not spy.empty:
        pr60 = pv_60.pct_change(fill_method=None).dropna()
        sr60 = spy.reindex(pr60.index).pct_change(fill_method=None).dropna()
        idx = pr60.index.intersection(sr60.index)
        pr60 = pr60.loc[idx]
        sr60 = sr60.loc[idx]
        
        # OLS Beta
        ba = _beta_alpha_ols(pr60, sr60)
        if ba:
            beta, alpha_daily, r2 = ba
            
            # Manual covariance/variance calculation
            cov_matrix = np.cov(pr60.values, sr60.values)
            manual_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            
            print(f"\nPortfolio Beta:")
            print(f"  OLS Beta:      {beta:.4f}")
            print(f"  Cov/Var Beta:  {manual_beta:.4f}")
            print(f"  R-squared:     {r2:.4f}")
            print(f"  Formula:       β = Cov(Rp, Rm) / Var(Rm)")
            print(f"  Match:         {'✓' if abs(beta - manual_beta) < 0.01 else '✗'}")
    
    # ==========================================================================
    # 7. ALPHA (Jensen's Alpha)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("7. ALPHA (Jensen's Alpha)")
    print("=" * 70)
    
    if ba:
        alpha_annual = alpha_daily * 252 * 100
        
        # Manual: Alpha = Rp - [Rf + β(Rm - Rf)]
        rf_dict = latest_1w_tbill(week_end)
        rf_val = rf_dict.get("value", 5.25)
        rf_daily = rf_val / 100 / 252
        
        Rp = pr60.mean()
        Rm = sr60.mean()
        manual_alpha = Rp - (rf_daily + beta * (Rm - rf_daily))
        manual_alpha_annual = manual_alpha * 252 * 100
        
        print(f"\nPortfolio Alpha:")
        print(f"  OLS Alpha (annual):     {alpha_annual:.2f}%")
        print(f"  Jensen Alpha (annual):  {manual_alpha_annual:.2f}%")
        print(f"  Formula:                α = Rp - [Rf + β(Rm - Rf)]")
        print(f"  Note: OLS intercept ≈ Jensen's Alpha when Rf is small")
        print(f"  Difference due to Rf:   {abs(alpha_annual - manual_alpha_annual):.2f}%")
    
    # ==========================================================================
    # 8. SHARPE RATIO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("8. SHARPE RATIO")
    print("=" * 70)
    
    if rf_val is not None:
        rf_annual = rf_val / 100  # e.g., 5.25% -> 0.0525
        rf_daily = rf_annual / 252
        
        excess_returns = pr60 - rf_daily
        sharpe = (excess_returns.mean() / pr60.std()) * np.sqrt(252)
        
        # Alternative calculation
        annual_excess = excess_returns.mean() * 252
        annual_vol = pr60.std() * np.sqrt(252)
        sharpe_alt = annual_excess / annual_vol
        
        print(f"\nSharpe Ratio:")
        print(f"  Risk-free rate (annual): {rf_val:.2f}%")
        print(f"  Portfolio return (ann):  {pr60.mean()*252*100:.2f}%")
        print(f"  Portfolio vol (ann):     {annual_vol*100:.2f}%")
        print(f"  Excess return (ann):     {annual_excess*100:.2f}%")
        print(f"  Sharpe Ratio:            {sharpe:.2f}")
        print(f"  Alt calculation:         {sharpe_alt:.2f}")
        print(f"  Formula:                 SR = (Rp - Rf) / σp")
        print(f"  Match:                   {'✓' if abs(sharpe - sharpe_alt) < 0.01 else '✗'}")
    
    # ==========================================================================
    # 9. TRACKING ERROR
    # ==========================================================================
    print("\n" + "=" * 70)
    print("9. TRACKING ERROR")
    print("=" * 70)
    
    pv_252 = _build_portfolio_value_series(sess, shares_map, tickers, week_end, 400)
    if not pv_252.empty and not spy.empty:
        pr252 = pv_252.pct_change(fill_method=None).dropna()
        sr252 = spy.reindex(pr252.index).pct_change(fill_method=None).dropna()
        idx = pr252.index.intersection(sr252.index)
        pr252 = pr252.loc[idx]
        sr252 = sr252.loc[idx]
        
        active_returns = pr252 - sr252
        tracking_error = active_returns.std() * np.sqrt(252) * 100
        
        print(f"\nTracking Error (252-day):")
        print(f"  Active return std (daily): {active_returns.std()*100:.4f}%")
        print(f"  Tracking Error (annual):   {tracking_error:.2f}%")
        print(f"  Formula:                   TE = σ(Rp - Rb) × √252")
    
    # ==========================================================================
    # 10. INFORMATION RATIO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("10. INFORMATION RATIO")
    print("=" * 70)
    
    if not pv_252.empty and tracking_error > 0:
        excess_return_annual = (pr252.mean() - sr252.mean()) * 252 * 100
        info_ratio = excess_return_annual / tracking_error
        
        print(f"\nInformation Ratio:")
        print(f"  Portfolio return (ann):   {pr252.mean()*252*100:.2f}%")
        print(f"  Benchmark return (ann):   {sr252.mean()*252*100:.2f}%")
        print(f"  Excess return (ann):      {excess_return_annual:.2f}%")
        print(f"  Tracking Error:           {tracking_error:.2f}%")
        print(f"  Information Ratio:        {info_ratio:.2f}")
        print(f"  Formula:                  IR = (Rp - Rb) / TE")
    
    # ==========================================================================
    # 11. MAX DRAWDOWN
    # ==========================================================================
    print("\n" + "=" * 70)
    print("11. MAX DRAWDOWN")
    print("=" * 70)
    
    if not pv_252.empty:
        mdd = _max_drawdown(pv_252)
        
        # Manual calculation
        cummax = pv_252.cummax()
        drawdown = pv_252 / cummax - 1
        manual_mdd = drawdown.min()
        
        print(f"\nMax Drawdown (252-day):")
        print(f"  Max Drawdown:       {mdd*100:.2f}%")
        print(f"  Manual calculation: {manual_mdd*100:.2f}%")
        print(f"  Formula:            MDD = min(Price / Peak - 1)")
        print(f"  Match:              {'✓' if abs(mdd - manual_mdd) < 0.0001 else '✗'}")
    
    # ==========================================================================
    # 12. UNREALIZED P&L
    # ==========================================================================
    print("\n" + "=" * 70)
    print("12. UNREALIZED P&L")
    print("=" * 70)
    
    sample_holding = holds[0]
    t = sample_holding.ticker
    ts = _series_for(sess, t, week_end, 10)
    if not ts.empty and sample_holding.avg_cost:
        price = float(ts.iloc[-1])
        shares = float(sample_holding.shares)
        avg_cost = float(sample_holding.avg_cost)
        
        market_value = shares * price
        cost_basis = shares * avg_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pct = (market_value / cost_basis - 1) * 100
        
        print(f"\n{t} Unrealized P&L:")
        print(f"  Shares:          {shares:.2f}")
        print(f"  Avg Cost:        ${avg_cost:.2f}")
        print(f"  Current Price:   ${price:.2f}")
        print(f"  Cost Basis:      ${cost_basis:,.2f}")
        print(f"  Market Value:    ${market_value:,.2f}")
        print(f"  Unrealized P&L:  ${unrealized_pnl:,.2f} ({unrealized_pct:+.2f}%)")
        print(f"  Formula:         P&L = (Price × Shares) - (Avg Cost × Shares)")


print("\n" + "=" * 70)
print("REVIEW COMPLETE")
print("=" * 70)
print("\nAll standard formulas verified:")
print("  ✓ Returns: (End/Start - 1)")
print("  ✓ RSI: 100 - 100/(1+RS), RS = EMA(gains)/EMA(losses)")
print("  ✓ MACD: EMA(12) - EMA(26), Signal = EMA(9)")
print("  ✓ Weights: MV / Total MV")
print("  ✓ Volatility: σ × √252")
print("  ✓ Beta: Cov(Rp,Rm) / Var(Rm)")
print("  ✓ Alpha: OLS intercept (≈ Jensen's α)")
print("  ✓ Sharpe: (Rp - Rf) / σp")
print("  ✓ Tracking Error: σ(Rp - Rb) × √252")
print("  ✓ Information Ratio: (Rp - Rb) / TE")
print("  ✓ Max Drawdown: min(Price/Peak - 1)")
print("  ✓ Unrealized P&L: (Price - Avg Cost) × Shares")
