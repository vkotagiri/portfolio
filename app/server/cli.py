# app/server/cli.py
from __future__ import annotations

from pathlib import Path
from datetime import date
import typer

from ..db import engine, Base, get_session
from ..models import Security, Holding
from ..services.ingestion import ingest_eod_for_date, backfill_history
from ..services.reporting import build_weekly_report

app = typer.Typer(no_args_is_help=True, add_completion=False)


def parse_date(s: str) -> date:
    s = (s or "").strip().lower()
    if s == "today" or s == "":
        return date.today()
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise typer.BadParameter("Use 'today' or YYYY-MM-DD") from exc


@app.command("init-db")
def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    typer.echo("DB initialized.")


@app.command("load-holdings")
def load_holdings(csv_path: Path = typer.Argument(..., help="CSV with columns: ticker,shares[,avg_cost]")):
    """Load/Upsert holdings from a CSV."""
    import pandas as pd

    if not csv_path.exists():
        raise typer.BadParameter(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    need_cols = {"ticker", "shares"}
    if not need_cols.issubset(set(df.columns)):
        raise typer.BadParameter("CSV must include at least: ticker, shares")

    rows = []
    has_avg = "avg_cost" in df.columns
    for _, r in df.iterrows():
        t = str(r["ticker"]).strip().upper()
        shares = float(r["shares"])
        avg_cost = None
        if has_avg and r["avg_cost"] is not None and str(r["avg_cost"]) != "nan":
            avg_cost = float(r["avg_cost"])
        rows.append({"ticker": t, "shares": shares, "avg_cost": avg_cost})

    with get_session() as sess:
        # Ensure securities exist first (FK safety)
        for r in rows:
            if not sess.get(Security, r["ticker"]):
                sess.add(Security(ticker=r["ticker"]))
        sess.flush()

        # Upsert holdings
        for r in rows:
            h = sess.get(Holding, r["ticker"])
            if h:
                h.shares = r["shares"]
                h.avg_cost = r["avg_cost"]
            else:
                sess.add(Holding(ticker=r["ticker"], shares=r["shares"], avg_cost=r["avg_cost"]))
        sess.commit()

    typer.echo(f"Loaded {len(rows)} holdings from {csv_path}.")


@app.command("ingest-eod")
def ingest_eod_cmd(
    date_value: str = typer.Argument("today", metavar="DATE", help="YYYY-MM-DD or 'today'")
):
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
    outlook: bool = typer.Option(
        False,
        "--outlook/--no-outlook",
        help="Include throttled online earnings outlook (uses yfinance).",
    ),
):
    """Generate Weekly Deep Dive HTML for WEEK_ENDING (default: today)."""
    d = parse_date(week_ending_value)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "weekly.html"
    out = build_weekly_report(d, out_html, fetch_outlook=outlook)
    typer.echo(out)


def generate_report_once(week_ending: str):
    d = parse_date(week_ending)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "weekly.html"
    return build_weekly_report(d, out_html)


if __name__ == "__main__":
    app()
