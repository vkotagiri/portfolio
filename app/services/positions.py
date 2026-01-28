# app/services/positions.py
"""
Professional-grade position and return tracking.
Implements Time-Weighted Return (TWR) calculation with daily snapshots.
Supports backdated trade adjustments.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
from sqlalchemy import and_, delete
from sqlalchemy.orm import Session

from ..db import get_session
from ..models import (
    DailyPosition, DailyPortfolioReturn, Trade, Holding, 
    Price, BenchmarkPrice, Security
)

logger = logging.getLogger(__name__)


def get_price_on_date(sess: Session, ticker: str, as_of: date) -> Optional[float]:
    """Get the price for a ticker on a specific date, or most recent prior."""
    # Try exact date first
    price = sess.query(Price).filter(
        Price.ticker == ticker,
        Price.date == as_of.isoformat()
    ).first()
    if price:
        return float(price.adj_close or price.close)
    
    # Fall back to most recent price before this date
    price = sess.query(Price).filter(
        Price.ticker == ticker,
        Price.date <= as_of.isoformat()
    ).order_by(Price.date.desc()).first()
    
    return float(price.adj_close or price.close) if price else None


def get_benchmark_price_on_date(sess: Session, symbol: str, as_of: date) -> Optional[float]:
    """Get benchmark price for a specific date."""
    price = sess.query(BenchmarkPrice).filter(
        BenchmarkPrice.symbol == symbol,
        BenchmarkPrice.date == as_of.isoformat()
    ).first()
    if price:
        return float(price.adj_close) if price.adj_close else None
    
    # Fall back to most recent
    price = sess.query(BenchmarkPrice).filter(
        BenchmarkPrice.symbol == symbol,
        BenchmarkPrice.date <= as_of.isoformat()
    ).order_by(BenchmarkPrice.date.desc()).first()
    
    return float(price.adj_close) if price and price.adj_close else None


def get_trading_days(sess: Session, start: date, end: date) -> List[date]:
    """Get list of trading days from prices table."""
    result = sess.query(Price.date).filter(
        Price.date >= start.isoformat(),
        Price.date <= end.isoformat()
    ).distinct().order_by(Price.date).all()
    
    return [date.fromisoformat(r[0]) for r in result]


def reconstruct_positions_on_date(
    sess: Session, 
    as_of: date,
    trades_before: List[Trade]
) -> Dict[str, Dict]:
    """
    Reconstruct what positions looked like on a specific date
    by replaying all trades up to that date.
    
    Returns: {ticker: {shares, avg_cost, cost_basis}}
    """
    positions: Dict[str, Dict] = {}
    
    # Sort trades by date, then by id (order within same day)
    sorted_trades = sorted(trades_before, key=lambda t: (t.trade_date, t.id))
    
    for trade in sorted_trades:
        if trade.trade_date > as_of.isoformat():
            continue
            
        ticker = trade.ticker
        if ticker not in positions:
            positions[ticker] = {"shares": 0, "avg_cost": 0, "cost_basis": 0}
        
        pos = positions[ticker]
        
        if trade.trade_type == "BUY":
            # Update average cost
            old_value = pos["shares"] * pos["avg_cost"]
            new_value = trade.shares * trade.price
            new_shares = pos["shares"] + trade.shares
            if new_shares > 0:
                pos["avg_cost"] = (old_value + new_value) / new_shares
            pos["shares"] = new_shares
            pos["cost_basis"] = pos["shares"] * pos["avg_cost"]
            
        elif trade.trade_type == "SELL":
            pos["shares"] = max(0, pos["shares"] - trade.shares)
            pos["cost_basis"] = pos["shares"] * pos["avg_cost"]
    
    # Remove zero positions
    return {k: v for k, v in positions.items() if v["shares"] > 0}


def calculate_daily_snapshot(
    sess: Session,
    as_of: date,
    positions: Dict[str, Dict]
) -> Tuple[List[DailyPosition], float, float]:
    """
    Calculate daily position snapshots and portfolio totals.
    
    Returns: (position_records, total_market_value, total_cost_basis)
    """
    records = []
    total_mv = 0.0
    total_cost = 0.0
    
    # First pass: calculate market values
    position_values = {}
    for ticker, pos in positions.items():
        price = get_price_on_date(sess, ticker, as_of)
        if price is None:
            logger.warning(f"No price for {ticker} on {as_of}")
            continue
            
        mv = pos["shares"] * price
        position_values[ticker] = {
            "price": price,
            "market_value": mv,
            "shares": pos["shares"],
            "avg_cost": pos["avg_cost"],
            "cost_basis": pos["cost_basis"]
        }
        total_mv += mv
        total_cost += pos["cost_basis"]
    
    # Second pass: calculate weights and create records
    for ticker, pv in position_values.items():
        weight = pv["market_value"] / total_mv if total_mv > 0 else 0
        unrealized = pv["market_value"] - pv["cost_basis"]
        unrealized_pct = (unrealized / pv["cost_basis"] * 100) if pv["cost_basis"] > 0 else None
        
        # Get previous day price for daily return
        prev_date = as_of - timedelta(days=1)
        prev_price = get_price_on_date(sess, ticker, prev_date)
        day_return = None
        if prev_price and prev_price > 0:
            day_return = (pv["price"] / prev_price - 1) * 100
        
        records.append(DailyPosition(
            date=as_of.isoformat(),
            ticker=ticker,
            shares=pv["shares"],
            price=pv["price"],
            market_value=round(pv["market_value"], 2),
            cost_basis=round(pv["cost_basis"], 2),
            avg_cost=round(pv["avg_cost"], 2),
            weight=round(weight, 6),
            day_return_pct=round(day_return, 4) if day_return is not None else None,
            unrealized_pnl=round(unrealized, 2),
            unrealized_pct=round(unrealized_pct, 2) if unrealized_pct is not None else None
        ))
    
    return records, total_mv, total_cost


def calculate_twr_return(
    prev_value: float,
    curr_value: float,
    cash_flow: float
) -> Optional[float]:
    """
    Calculate Time-Weighted Return for a single period.
    
    TWR formula: (End Value - Cash Flow) / Start Value - 1
    This removes the impact of cash flows (buys/sells) on the return.
    """
    if prev_value <= 0:
        return None
    
    # Adjust end value by removing cash flow impact
    adjusted_end = curr_value - cash_flow
    return (adjusted_end / prev_value - 1) * 100


def rebuild_returns_from_date(
    start_date: date,
    end_date: Optional[date] = None
) -> Dict:
    """
    Rebuild all position snapshots and portfolio returns from start_date to end_date.
    This is the core function called when backdated trades are entered.
    
    Returns: {status, start, end, days_processed, positions_created}
    """
    if end_date is None:
        end_date = date.today()
    
    logger.info(f"Rebuilding returns from {start_date} to {end_date}")
    
    with get_session() as sess:
        # Get all trades (we'll filter as we go)
        all_trades = sess.query(Trade).order_by(Trade.trade_date, Trade.id).all()
        
        # Get trading days in range
        trading_days = get_trading_days(sess, start_date, end_date)
        
        if not trading_days:
            return {"status": "no_trading_days", "start": start_date.isoformat(), "end": end_date.isoformat()}
        
        # Delete existing records in this date range
        sess.execute(
            delete(DailyPosition).where(
                and_(
                    DailyPosition.date >= start_date.isoformat(),
                    DailyPosition.date <= end_date.isoformat()
                )
            )
        )
        sess.execute(
            delete(DailyPortfolioReturn).where(
                and_(
                    DailyPortfolioReturn.date >= start_date.isoformat(),
                    DailyPortfolioReturn.date <= end_date.isoformat()
                )
            )
        )
        
        # Get previous day's portfolio value for first day TWR calculation
        prev_return = sess.query(DailyPortfolioReturn).filter(
            DailyPortfolioReturn.date < start_date.isoformat()
        ).order_by(DailyPortfolioReturn.date.desc()).first()
        
        prev_value = prev_return.portfolio_value if prev_return else 0
        cumulative_return = prev_return.cumulative_return_pct if prev_return else 0
        
        # Get benchmark previous
        prev_benchmark = get_benchmark_price_on_date(sess, "SPY", start_date - timedelta(days=1))
        benchmark_cumulative = prev_return.benchmark_cumulative_pct if prev_return else 0
        
        positions_created = 0
        days_processed = 0
        
        for trading_day in trading_days:
            # Get trades that happened on or before this day
            trades_up_to_day = [t for t in all_trades if t.trade_date <= trading_day.isoformat()]
            
            # Reconstruct positions
            positions = reconstruct_positions_on_date(sess, trading_day, trades_up_to_day)
            
            # Calculate daily snapshot
            pos_records, total_mv, total_cost = calculate_daily_snapshot(sess, trading_day, positions)
            
            # Add position records
            for rec in pos_records:
                sess.add(rec)
            positions_created += len(pos_records)
            
            # Calculate cash flow for the day (buys are positive inflow, sells negative)
            day_trades = [t for t in all_trades if t.trade_date == trading_day.isoformat()]
            cash_flow = sum(
                t.total_value if t.trade_type == "BUY" else -t.total_value
                for t in day_trades
            )
            
            # Calculate realized P&L for the day
            realized_pnl_day = sum(
                t.realized_gain or 0 for t in day_trades if t.trade_type == "SELL"
            )
            
            # Calculate TWR
            daily_return = calculate_twr_return(prev_value, total_mv, cash_flow) if prev_value > 0 else None
            
            # Update cumulative return
            if daily_return is not None:
                if cumulative_return is None or cumulative_return == 0:
                    cumulative_return = daily_return
                else:
                    # Compound returns: (1 + cum)(1 + daily) - 1
                    cumulative_return = (1 + cumulative_return/100) * (1 + daily_return/100) * 100 - 100
            
            # Benchmark return
            benchmark_price = get_benchmark_price_on_date(sess, "SPY", trading_day)
            benchmark_return = None
            if benchmark_price and prev_benchmark and prev_benchmark > 0:
                benchmark_return = (benchmark_price / prev_benchmark - 1) * 100
                if benchmark_cumulative is None or benchmark_cumulative == 0:
                    benchmark_cumulative = benchmark_return
                else:
                    benchmark_cumulative = (1 + benchmark_cumulative/100) * (1 + benchmark_return/100) * 100 - 100
            
            # Unrealized P&L
            unrealized_pnl = total_mv - total_cost
            
            # Create portfolio return record
            port_return = DailyPortfolioReturn(
                date=trading_day.isoformat(),
                portfolio_value=round(total_mv, 2),
                cost_basis=round(total_cost, 2),
                cash_flow=round(cash_flow, 2),
                daily_return_pct=round(daily_return, 4) if daily_return is not None else None,
                cumulative_return_pct=round(cumulative_return, 4) if cumulative_return is not None else None,
                benchmark_return_pct=round(benchmark_return, 4) if benchmark_return is not None else None,
                benchmark_cumulative_pct=round(benchmark_cumulative, 4) if benchmark_cumulative is not None else None,
                active_return_pct=round(daily_return - benchmark_return, 4) if daily_return and benchmark_return else None,
                realized_pnl_day=round(realized_pnl_day, 2) if realized_pnl_day else None,
                unrealized_pnl=round(unrealized_pnl, 2),
                total_pnl=round((realized_pnl_day or 0) + unrealized_pnl, 2)
            )
            sess.add(port_return)
            
            # Update for next iteration
            prev_value = total_mv
            if benchmark_price:
                prev_benchmark = benchmark_price
            days_processed += 1
        
        sess.commit()
        
        logger.info(f"Rebuilt {days_processed} days, {positions_created} position records")
        
        return {
            "status": "ok",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "days_processed": days_processed,
            "positions_created": positions_created
        }


def rebuild_all_returns() -> Dict:
    """Rebuild all returns from the earliest trade date."""
    with get_session() as sess:
        first_trade = sess.query(Trade).order_by(Trade.trade_date).first()
        if not first_trade:
            return {"status": "no_trades", "message": "No trades found to rebuild from"}
        
        start_date = date.fromisoformat(first_trade.trade_date)
        
    return rebuild_returns_from_date(start_date)


def get_performance_summary(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> Dict:
    """Get performance summary for a date range."""
    with get_session() as sess:
        query = sess.query(DailyPortfolioReturn)
        
        if start_date:
            query = query.filter(DailyPortfolioReturn.date >= start_date.isoformat())
        if end_date:
            query = query.filter(DailyPortfolioReturn.date <= end_date.isoformat())
        
        returns = query.order_by(DailyPortfolioReturn.date).all()
        
        if not returns:
            return {"status": "no_data"}
        
        first = returns[0]
        last = returns[-1]
        
        # Calculate total realized gains in period
        total_realized = sum(r.realized_pnl_day or 0 for r in returns)
        
        return {
            "start_date": first.date,
            "end_date": last.date,
            "start_value": first.portfolio_value,
            "end_value": last.portfolio_value,
            "twr_return_pct": last.cumulative_return_pct,
            "benchmark_return_pct": last.benchmark_cumulative_pct,
            "active_return_pct": (last.cumulative_return_pct or 0) - (last.benchmark_cumulative_pct or 0),
            "total_realized_pnl": round(total_realized, 2),
            "unrealized_pnl": last.unrealized_pnl,
            "total_pnl": last.total_pnl,
            "trading_days": len(returns)
        }


def record_daily_snapshot(as_of: date = None) -> Dict:
    """
    Record today's position snapshot. Called by daily refresh.
    Only adds/updates the single day's record.
    """
    if as_of is None:
        as_of = date.today()
    
    return rebuild_returns_from_date(as_of, as_of)
