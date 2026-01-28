from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer, Float, ForeignKey, Text
from .db import Base
# --- API quota logging ---
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer

class Security(Base):
    __tablename__ = "securities"
    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    exchange: Mapped[str | None] = mapped_column(String, nullable=True)
    currency: Mapped[str | None] = mapped_column(String, nullable=True)
    sector: Mapped[str | None] = mapped_column(String, nullable=True)
    industry: Mapped[str | None] = mapped_column(String, nullable=True)
    sector_updated: Mapped[str | None] = mapped_column(String, nullable=True)
    first_seen: Mapped[str | None] = mapped_column(String, nullable=True)
    last_seen: Mapped[str | None] = mapped_column(String, nullable=True)

class Price(Base):
    __tablename__ = "prices"
    ticker: Mapped[str] = mapped_column(String, ForeignKey("securities.ticker"), primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)  # YYYY-MM-DD
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dividend: Mapped[float | None] = mapped_column(Float, nullable=True)
    split: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String)
    asof_ts: Mapped[str] = mapped_column(String)

class Holding(Base):
    __tablename__ = "holdings"
    ticker: Mapped[str] = mapped_column(String, ForeignKey("securities.ticker"), primary_key=True)
    shares: Mapped[float] = mapped_column(Float)
    avg_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_last_updated: Mapped[str | None] = mapped_column(String, nullable=True)

class BenchmarkPrice(Base):
    __tablename__ = "benchmark_prices"
    symbol: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String)
    asof_ts: Mapped[str] = mapped_column(String)

class Factor(Base):
    __tablename__ = "factors"
    date: Mapped[str] = mapped_column(String, primary_key=True)
    mkt: Mapped[float | None] = mapped_column(Float, nullable=True)
    smb: Mapped[float | None] = mapped_column(Float, nullable=True)
    hml: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmw: Mapped[float | None] = mapped_column(Float, nullable=True)
    cma: Mapped[float | None] = mapped_column(Float, nullable=True)
    mom: Mapped[float | None] = mapped_column(Float, nullable=True)
    rf: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String)
    asof_ts: Mapped[str] = mapped_column(String)

class OptionQuote(Base):
    __tablename__ = "options"
    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    asof_date: Mapped[str] = mapped_column(String, primary_key=True)
    expiry: Mapped[str] = mapped_column(String, primary_key=True)
    strike: Mapped[float] = mapped_column(Float, primary_key=True)
    right: Mapped[str] = mapped_column(String, primary_key=True)  # 'C' or 'P'
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String)

class Run(Base):
    __tablename__ = "runs"
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    started_ts: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_ts: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

class Validation(Base):
    __tablename__ = "validation"
    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    rule: Mapped[str] = mapped_column(String, primary_key=True)
    passed: Mapped[int] = mapped_column(Integer)  # 0/1
    details: Mapped[str | None] = mapped_column(Text, nullable=True)



class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String, ForeignKey("securities.ticker"), index=True)
    trade_date: Mapped[str] = mapped_column(String, index=True)  # YYYY-MM-DD
    trade_type: Mapped[str] = mapped_column(String)  # 'BUY' or 'SELL'
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    total_value: Mapped[float] = mapped_column(Float)  # shares * price
    realized_gain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # for sells: (price - avg_cost) * shares
    avg_cost_at_trade: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # snapshot of avg cost
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_ts: Mapped[str] = mapped_column(String)  # when trade was recorded


class DailyPosition(Base):
    """Daily snapshot of each position - stores EOD state for accurate historical tracking."""
    __tablename__ = "daily_positions"
    date: Mapped[str] = mapped_column(String, primary_key=True)  # YYYY-MM-DD
    ticker: Mapped[str] = mapped_column(String, ForeignKey("securities.ticker"), primary_key=True)
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)  # EOD price
    market_value: Mapped[float] = mapped_column(Float)  # shares * price
    cost_basis: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # shares * avg_cost
    avg_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # position weight in portfolio
    day_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # single stock return
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unrealized_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class DailyPortfolioReturn(Base):
    """Portfolio-level daily returns - the source of truth for performance."""
    __tablename__ = "daily_portfolio_returns"
    date: Mapped[str] = mapped_column(String, primary_key=True)  # YYYY-MM-DD
    portfolio_value: Mapped[float] = mapped_column(Float)  # total market value EOD
    cost_basis: Mapped[float] = mapped_column(Float)  # total cost basis
    cash_flow: Mapped[float] = mapped_column(Float, default=0)  # net buys - sells for the day
    daily_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # TWR daily return
    cumulative_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # TWR from inception
    benchmark_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # SPY daily return
    benchmark_cumulative_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # SPY cumulative
    active_return_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # portfolio - benchmark
    realized_pnl_day: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # realized gains for the day
    unrealized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # total unrealized
    total_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # realized + unrealized


class ApiCall(Base):
    __tablename__ = "api_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String, index=True)
    endpoint: Mapped[str] = mapped_column(String)
    symbol: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ts_utc: Mapped[str] = mapped_column(String)  # ISO8601, e.g., 2025-10-25T21:59:30Z

