# app/server/cli.py
import typer
from pathlib import Path
from datetime import date, datetime

from ..db import engine, Base, get_session
from ..models import Security, Holding, Trade
from ..services.ingestion import ingest_eod_for_date, backfill_history
from ..services.reporting import build_weekly_report, build_daily_report
from ..services.positions import rebuild_returns_from_date, rebuild_all_returns, get_performance_summary
from dotenv import load_dotenv
load_dotenv()   # loads .env into os.environ

app = typer.Typer(add_completion=False)

def parse_date(s: str) -> date:
    s = s.strip().lower()
    if s == "today":
        return date.today()
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise typer.BadParameter("Use 'today' or YYYY-MM-DD")

@app.command()
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    typer.echo("DB initialized.")

@app.command()
def load_holdings(csv_path: Path):
    """Load/Upsert holdings from a CSV with columns: ticker,shares,avg_cost."""
    import pandas as pd
    if not csv_path.exists():
        raise typer.BadParameter(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    rows = []
    for _, r in df.iterrows():
        t = str(r["ticker"]).strip().upper()
        shares = float(r["shares"])
        avg_cost = None
        if "avg_cost" in r and not pd.isna(r["avg_cost"]):
            avg_cost = float(r["avg_cost"])
        rows.append({"ticker": t, "shares": shares, "avg_cost": avg_cost})

    with get_session() as sess:
        for r in rows:
            if not sess.get(Security, r["ticker"]):
                sess.add(Security(ticker=r["ticker"]))
        sess.flush()

        for r in rows:
            h = sess.get(Holding, r["ticker"])
            if h:
                h.shares = r["shares"]
                h.avg_cost = r["avg_cost"]
            else:
                sess.add(Holding(ticker=r["ticker"], shares=r["shares"], avg_cost=r["avg_cost"]))
        sess.commit()

    typer.echo(f"Loaded {len(rows)} holdings from {csv_path}.")

@app.command("import-initial-trades")
def import_initial_trades_cmd(
    as_of: str = typer.Argument(..., help="Date when holdings were acquired (YYYY-MM-DD)"),
):
    """
    Import current holdings as initial BUY trades.
    Use this to bootstrap the position tracking system with existing holdings.
    This creates synthetic BUY trades for all holdings as of the specified date.
    """
    d = parse_date(as_of)
    
    with get_session() as sess:
        holdings = sess.query(Holding).all()
        
        if not holdings:
            typer.echo("No holdings found to import.")
            return
        
        # Check for existing trades on this date to avoid duplicates
        existing = sess.query(Trade).filter(Trade.trade_date == d.isoformat()).count()
        if existing > 0:
            typer.echo(f"⚠ Found {existing} trades already on {d}. Skipping to avoid duplicates.")
            typer.echo("  Delete existing trades first if you want to re-import.")
            return
        
        count = 0
        for h in holdings:
            # Skip if this holding was created by a trade we recorded today
            recent_trade = sess.query(Trade).filter(
                Trade.ticker == h.ticker,
                Trade.trade_date == date.today().isoformat()
            ).first()
            if recent_trade:
                continue
                
            trade = Trade(
                ticker=h.ticker,
                trade_date=d.isoformat(),
                trade_type="BUY",
                shares=h.shares,
                price=h.avg_cost or 0,
                total_value=h.shares * (h.avg_cost or 0),
                realized_gain=None,
                avg_cost_at_trade=None,
                notes="Initial position import",
                created_ts=datetime.now().isoformat()
            )
            sess.add(trade)
            count += 1
        
        sess.commit()
        typer.echo(f"✓ Imported {count} holdings as initial BUY trades dated {d}")
    
    # Rebuild returns from that date
    typer.echo("Rebuilding returns from import date...")
    result = rebuild_returns_from_date(d)
    typer.echo(f"✓ Processed {result.get('days_processed', 0)} trading days")
    typer.echo(f"✓ Created {result.get('positions_created', 0)} position records")

@app.command("update-holding")
def update_holding_cmd(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    shares: float = typer.Argument(..., help="New total shares (use 0 to remove)"),
    avg_cost: float = typer.Option(None, "--cost", "-c", help="Average cost per share"),
):
    """Update or add a single holding. Use shares=0 to remove."""
    ticker = ticker.strip().upper()
    with get_session() as sess:
        # Ensure security exists
        if not sess.get(Security, ticker):
            sess.add(Security(ticker=ticker))
            sess.flush()
        
        h = sess.get(Holding, ticker)
        if shares == 0:
            # Remove holding
            if h:
                sess.delete(h)
                sess.commit()
                typer.echo(f"✓ Removed {ticker} from portfolio")
            else:
                typer.echo(f"⚠ {ticker} not in portfolio")
        else:
            if h:
                h.shares = shares
                if avg_cost is not None:
                    h.avg_cost = avg_cost
                sess.commit()
                typer.echo(f"✓ Updated {ticker}: {shares} shares @ ${h.avg_cost or 'N/A'}")
            else:
                sess.add(Holding(ticker=ticker, shares=shares, avg_cost=avg_cost))
                sess.commit()
                typer.echo(f"✓ Added {ticker}: {shares} shares @ ${avg_cost or 'N/A'}")

@app.command("sell")
def sell_cmd(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    shares: float = typer.Argument(..., help="Number of shares to sell"),
    price: float = typer.Argument(..., help="Sell price per share"),
    trade_date: str = typer.Option("today", "--date", "-d", help="Trade date (YYYY-MM-DD or 'today')"),
    notes: str = typer.Option(None, "--notes", "-n", help="Optional notes"),
):
    """Sell shares from an existing holding with price tracking."""
    ticker = ticker.strip().upper()
    d = parse_date(trade_date)
    
    with get_session() as sess:
        h = sess.get(Holding, ticker)
        if not h:
            typer.echo(f"✗ {ticker} not in portfolio")
            raise typer.Exit(1)
        
        sell_shares = min(shares, h.shares)  # Can't sell more than owned
        avg_cost = h.avg_cost or 0
        total_value = sell_shares * price
        realized_gain = (price - avg_cost) * sell_shares
        
        # Log the trade
        trade = Trade(
            ticker=ticker,
            trade_date=d.isoformat(),
            trade_type="SELL",
            shares=sell_shares,
            price=price,
            total_value=total_value,
            realized_gain=realized_gain,
            avg_cost_at_trade=avg_cost,
            notes=notes,
            created_ts=datetime.now().isoformat()
        )
        sess.add(trade)
        
        # Update holding
        if sell_shares >= h.shares:
            sess.delete(h)
            typer.echo(f"✓ Sold all {sell_shares:.2f} shares of {ticker} @ ${price:.2f}")
        else:
            h.shares -= sell_shares
            typer.echo(f"✓ Sold {sell_shares:.2f} shares of {ticker} @ ${price:.2f}, remaining: {h.shares:.2f}")
        
        sess.commit()
        
        gain_str = f"+${realized_gain:.2f}" if realized_gain >= 0 else f"-${abs(realized_gain):.2f}"
        typer.echo(f"  Trade date: {d}, Realized P&L: {gain_str}")
    
    # Recalculate returns from trade date
    typer.echo("  Recalculating portfolio returns...")
    result = rebuild_returns_from_date(d)
    typer.echo(f"  ✓ Updated {result.get('days_processed', 0)} days of returns")

@app.command("buy")
def buy_cmd(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    shares: float = typer.Argument(..., help="Number of shares to buy"),
    price: float = typer.Argument(..., help="Purchase price per share"),
    trade_date: str = typer.Option("today", "--date", "-d", help="Trade date (YYYY-MM-DD or 'today')"),
    notes: str = typer.Option(None, "--notes", "-n", help="Optional notes"),
):
    """Buy shares (adds to existing or creates new holding with updated avg cost)."""
    ticker = ticker.strip().upper()
    d = parse_date(trade_date)
    
    with get_session() as sess:
        # Ensure security exists
        if not sess.get(Security, ticker):
            sess.add(Security(ticker=ticker))
            sess.flush()
        
        h = sess.get(Holding, ticker)
        old_avg_cost = h.avg_cost if h else None
        
        if h:
            # Calculate new average cost
            old_value = h.shares * (h.avg_cost or 0)
            new_value = shares * price
            new_shares = h.shares + shares
            new_avg = (old_value + new_value) / new_shares if new_shares > 0 else price
            h.shares = new_shares
            h.avg_cost = round(new_avg, 2)
            h.cost_last_updated = d.isoformat()
        else:
            h = Holding(ticker=ticker, shares=shares, avg_cost=price, cost_last_updated=d.isoformat())
            sess.add(h)
        
        # Log the trade
        trade = Trade(
            ticker=ticker,
            trade_date=d.isoformat(),
            trade_type="BUY",
            shares=shares,
            price=price,
            total_value=shares * price,
            realized_gain=None,  # No realized gain on buys
            avg_cost_at_trade=old_avg_cost,
            notes=notes,
            created_ts=datetime.now().isoformat()
        )
        sess.add(trade)
        sess.commit()
        
        typer.echo(f"✓ Bought {shares} {ticker} @ ${price:.2f} on {d}")
        typer.echo(f"  New position: {h.shares:.2f} shares @ ${h.avg_cost:.2f} avg cost")
    
    # Recalculate returns from trade date
    typer.echo("  Recalculating portfolio returns...")
    result = rebuild_returns_from_date(d)
    typer.echo(f"  ✓ Updated {result.get('days_processed', 0)} days of returns")

@app.command("refresh-sectors")
def refresh_sectors_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Force refresh even if sector data exists"),
):
    """Refresh sector/industry classification for all holdings."""
    from ..services.sectors import refresh_all_sectors
    
    typer.echo("🏭 Refreshing sector data from Yahoo Finance...")
    typer.echo("   (This may take a minute due to rate limiting)")
    
    results = refresh_all_sectors(force=force)
    
    # Group by sector for summary
    sector_counts: dict = {}
    for ticker, sector in results.items():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    typer.echo("\n📊 Sector Summary:")
    for sector in sorted(sector_counts.keys()):
        typer.echo(f"   {sector}: {sector_counts[sector]} holdings")
    
    typer.echo(f"\n✓ Updated {len(results)} securities")

@app.command("list-holdings")
def list_holdings_cmd():
    """List all current holdings."""
    with get_session() as sess:
        holdings = sess.query(Holding).order_by(Holding.ticker).all()
        if not holdings:
            typer.echo("No holdings found.")
            return
        typer.echo(f"{'Ticker':<8} {'Shares':>10} {'Avg Cost':>12}")
        typer.echo("-" * 32)
        for h in holdings:
            cost_str = f"${h.avg_cost:.2f}" if h.avg_cost else "N/A"
            typer.echo(f"{h.ticker:<8} {h.shares:>10.2f} {cost_str:>12}")

@app.command("trades")
def trades_cmd(
    ticker: str = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of trades to show"),
):
    """Show trade history."""
    with get_session() as sess:
        query = sess.query(Trade).order_by(Trade.trade_date.desc(), Trade.id.desc())
        if ticker:
            query = query.filter(Trade.ticker == ticker.upper())
        trades = query.limit(limit).all()
        
        if not trades:
            typer.echo("No trades found.")
            return
        
        typer.echo(f"{'Date':<12} {'Type':<5} {'Ticker':<8} {'Shares':>8} {'Price':>10} {'P&L':>12}")
        typer.echo("-" * 60)
        total_realized = 0
        for t in trades:
            pnl_str = ""
            if t.realized_gain is not None:
                total_realized += t.realized_gain
                pnl_str = f"+${t.realized_gain:.2f}" if t.realized_gain >= 0 else f"-${abs(t.realized_gain):.2f}"
            typer.echo(f"{t.trade_date:<12} {t.trade_type:<5} {t.ticker:<8} {t.shares:>8.2f} ${t.price:>9.2f} {pnl_str:>12}")
        
        typer.echo("-" * 60)
        total_str = f"+${total_realized:.2f}" if total_realized >= 0 else f"-${abs(total_realized):.2f}"
        typer.echo(f"Total Realized P&L: {total_str}")

@app.command("realized-gains")
def realized_gains_cmd(
    year: int = typer.Option(None, "--year", "-y", help="Filter by year"),
):
    """Show realized gains/losses summary."""
    with get_session() as sess:
        query = sess.query(Trade).filter(Trade.trade_type == "SELL")
        if year:
            query = query.filter(Trade.trade_date.like(f"{year}%"))
        trades = query.order_by(Trade.trade_date).all()
        
        if not trades:
            typer.echo("No sell trades found.")
            return
        
        total_gain = 0
        total_proceeds = 0
        typer.echo(f"{'Date':<12} {'Ticker':<8} {'Shares':>8} {'Proceeds':>12} {'Gain/Loss':>12}")
        typer.echo("-" * 56)
        for t in trades:
            gain = t.realized_gain or 0
            total_gain += gain
            total_proceeds += t.total_value
            gain_str = f"+${gain:.2f}" if gain >= 0 else f"-${abs(gain):.2f}"
            typer.echo(f"{t.trade_date:<12} {t.ticker:<8} {t.shares:>8.2f} ${t.total_value:>11.2f} {gain_str:>12}")
        
        typer.echo("-" * 56)
        total_str = f"+${total_gain:.2f}" if total_gain >= 0 else f"-${abs(total_gain):.2f}"
        typer.echo(f"Total Proceeds: ${total_proceeds:,.2f}")
        typer.echo(f"Total Realized Gain/Loss: {total_str}")

@app.command("ingest-eod")
def ingest_eod_cmd(date_value: str = typer.Argument("today", metavar="DATE", help="YYYY-MM-DD or 'today'")):
    """Ingest adjusted closes for DATE (default: today)."""
    d = parse_date(date_value)
    out = ingest_eod_for_date(d)
    typer.echo(out)

@app.command("backfill")
def backfill_cmd(
    start: str = typer.Argument(..., metavar="START", help="YYYY-MM-DD"),
    end: str = typer.Argument(..., metavar="END", help="YYYY-MM-DD or 'today'"),
):
    """Backfill history from START to END (inclusive) for all holdings + SPY."""
    s = parse_date(start)
    e = parse_date(end)
    if s > e:
        raise typer.BadParameter("START must be <= END")
    out = backfill_history(s, e)
    typer.echo(out)
  
@app.command("report")
def report_cmd(
    week_ending_value: str = typer.Argument("today", metavar="WEEK_ENDING", help="YYYY-MM-DD or 'today'"),
    outlook: bool = typer.Option(False, "--outlook", help="(Optional) fetch news module if available"),
    ai_summary: bool = typer.Option(True, "--ai-summary/--no-ai-summary", help="Include LLM summary (default: enabled)"),
    send_email: bool = typer.Option(False, "--email", "-e", help="Send email notification after report"),
):
    """Generate Weekly Deep Dive HTML for WEEK_ENDING (default: today)."""
    from ..config import settings
    from ..services.email_notify import send_portfolio_email, send_error_notification
    
    d = parse_date(week_ending_value)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "weekly.html"
    
    try:
        # Build report, optionally returning payload for email
        out = build_weekly_report(
            d, 
            str(out_html), 
            fetch_outlook=outlook, 
            ai_summary=ai_summary,
            return_payload=send_email  # Only get payload if we need it for email
        )
        typer.echo(out)
        
        # Send email if requested
        if send_email:
            if not settings.smtp_user or not settings.smtp_password or not settings.email_to:
                typer.echo("⚠ Email not configured. Set SMTP_USER, SMTP_PASSWORD, and EMAIL_TO in .env")
            else:
                # Get report data for email (from payload if available)
                report_data = out.get("payload", {"as_of": d.isoformat(), "status": "ok"})
                
                result = send_portfolio_email(
                    report_data=report_data,
                    report_path=str(out_html.absolute()),
                    smtp_host=settings.smtp_host,
                    smtp_port=settings.smtp_port,
                    smtp_user=settings.smtp_user,
                    smtp_password=settings.smtp_password,
                    to_email=settings.email_to,
                    from_email=settings.email_from,
                    openai_api_key=settings.openai_api_key,
                    use_tls=settings.smtp_use_tls,
                    report_base_url=settings.report_base_url
                )
                if result["status"] == "ok":
                    typer.echo(f"📧 Email sent to {result['to']}")
                else:
                    typer.echo(f"⚠ Email failed: {result.get('error')}")
                    
    except Exception as e:
        error_msg = str(e)
        typer.echo(f"✗ Report generation failed: {error_msg}")
        
        # Send error notification if email is configured
        if send_email and settings.smtp_user and settings.smtp_password and settings.email_to:
            result = send_error_notification(
                error_message=error_msg,
                report_date=d,
                smtp_host=settings.smtp_host,
                smtp_port=settings.smtp_port,
                smtp_user=settings.smtp_user,
                smtp_password=settings.smtp_password,
                to_email=settings.email_to,
                from_email=settings.email_from,
                use_tls=settings.smtp_use_tls
            )
            if result["status"] == "ok":
                typer.echo(f"📧 Error notification sent to {settings.email_to}")
        
        raise typer.Exit(1)

@app.command("report-daily")
def report_daily_cmd(
    as_of_value: str = typer.Argument("today", metavar="AS_OF", help="YYYY-MM-DD or 'today'"),
    ai_summary: bool = typer.Option(True, "--ai-summary/--no-ai-summary", help="Include LLM summary (default: enabled)"),
):
    """Generate Daily Portfolio Brief HTML for AS_OF (default: today)."""
    d = parse_date(as_of_value)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "daily.html"
    out = build_daily_report(d, str(out_html), ai_summary=ai_summary)
    typer.echo(out)

def generate_report_once(week_ending: str):
    d = parse_date(week_ending)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "weekly.html"
    return build_weekly_report(d, str(out_html))

# ==================== Position & Return Commands ====================

@app.command("rebuild-returns")
def rebuild_returns_cmd(
    start: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD), defaults to first trade"),
    end: str = typer.Option("today", "--end", "-e", help="End date (YYYY-MM-DD or 'today')"),
):
    """Rebuild daily position snapshots and portfolio returns from scratch."""
    from ..services.positions import rebuild_returns_from_date, rebuild_all_returns
    
    if start is None:
        typer.echo("Rebuilding all returns from first trade...")
        result = rebuild_all_returns()
    else:
        s = parse_date(start)
        e = parse_date(end)
        typer.echo(f"Rebuilding returns from {s} to {e}...")
        result = rebuild_returns_from_date(s, e)
    
    if result.get("status") == "ok":
        typer.echo(f"✓ Processed {result['days_processed']} trading days")
        typer.echo(f"✓ Created {result['positions_created']} position records")
    else:
        typer.echo(f"⚠ {result}")

@app.command("performance")
def performance_cmd(
    start: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("today", "--end", "-e", help="End date (YYYY-MM-DD or 'today')"),
):
    """Show portfolio performance summary (TWR)."""
    from ..services.positions import get_performance_summary
    
    s = parse_date(start) if start else None
    e = parse_date(end)
    
    summary = get_performance_summary(s, e)
    
    if summary.get("status") == "no_data":
        typer.echo("No performance data found. Run 'rebuild-returns' first.")
        return
    
    typer.echo("\n📊 Portfolio Performance Summary")
    typer.echo("=" * 50)
    typer.echo(f"Period: {summary['start_date']} to {summary['end_date']}")
    typer.echo(f"Trading Days: {summary['trading_days']}")
    typer.echo("-" * 50)
    typer.echo(f"Starting Value:     ${summary['start_value']:>15,.2f}")
    typer.echo(f"Ending Value:       ${summary['end_value']:>15,.2f}")
    typer.echo("-" * 50)
    
    # Returns
    twr = summary.get('twr_return_pct')
    bench = summary.get('benchmark_return_pct')
    active = summary.get('active_return_pct')
    
    twr_str = f"{twr:+.2f}%" if twr is not None else "N/A"
    bench_str = f"{bench:+.2f}%" if bench is not None else "N/A"
    active_str = f"{active:+.2f}%" if active is not None else "N/A"
    
    typer.echo(f"Portfolio Return (TWR): {twr_str:>12}")
    typer.echo(f"Benchmark (SPY):        {bench_str:>12}")
    typer.echo(f"Active Return:          {active_str:>12}")
    typer.echo("-" * 50)
    
    # P&L
    realized = summary.get('total_realized_pnl', 0)
    unrealized = summary.get('unrealized_pnl', 0)
    total = summary.get('total_pnl', 0)
    
    def fmt_pnl(val):
        return f"+${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"
    
    typer.echo(f"Realized P&L:       {fmt_pnl(realized):>16}")
    typer.echo(f"Unrealized P&L:     {fmt_pnl(unrealized):>16}")
    typer.echo(f"Total P&L:          {fmt_pnl(total):>16}")
    typer.echo("=" * 50)

@app.command("daily-returns")
def daily_returns_cmd(
    days: int = typer.Option(10, "--days", "-d", help="Number of days to show"),
    ticker: str = typer.Option(None, "--ticker", "-t", help="Show returns for specific ticker"),
):
    """Show daily portfolio or stock returns."""
    from ..models import DailyPortfolioReturn, DailyPosition
    
    with get_session() as sess:
        if ticker:
            # Show specific stock returns
            positions = sess.query(DailyPosition).filter(
                DailyPosition.ticker == ticker.upper()
            ).order_by(DailyPosition.date.desc()).limit(days).all()
            
            if not positions:
                typer.echo(f"No position data for {ticker}. Run 'rebuild-returns' first.")
                return
            
            typer.echo(f"\n📈 {ticker.upper()} Daily Returns (last {days} days)")
            typer.echo(f"{'Date':<12} {'Price':>10} {'Shares':>10} {'Value':>12} {'Return':>10}")
            typer.echo("-" * 58)
            for p in reversed(positions):
                ret_str = f"{p.day_return_pct:+.2f}%" if p.day_return_pct else "N/A"
                typer.echo(f"{p.date:<12} ${p.price:>9.2f} {p.shares:>10.2f} ${p.market_value:>11,.2f} {ret_str:>10}")
        else:
            # Show portfolio returns
            returns = sess.query(DailyPortfolioReturn).order_by(
                DailyPortfolioReturn.date.desc()
            ).limit(days).all()
            
            if not returns:
                typer.echo("No return data found. Run 'rebuild-returns' first.")
                return
            
            typer.echo(f"\n📊 Portfolio Daily Returns (last {days} days)")
            typer.echo(f"{'Date':<12} {'Value':>14} {'Daily':>10} {'Cumulative':>12} {'vs SPY':>10}")
            typer.echo("-" * 62)
            for r in reversed(returns):
                daily_str = f"{r.daily_return_pct:+.2f}%" if r.daily_return_pct else "N/A"
                cum_str = f"{r.cumulative_return_pct:+.2f}%" if r.cumulative_return_pct else "N/A"
                active_str = f"{r.active_return_pct:+.2f}%" if r.active_return_pct else "N/A"
                typer.echo(f"{r.date:<12} ${r.portfolio_value:>13,.2f} {daily_str:>10} {cum_str:>12} {active_str:>10}")

if __name__ == "__main__":
    app()