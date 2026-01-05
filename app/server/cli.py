# app/server/cli.py
import typer
from pathlib import Path
from datetime import date

from ..db import engine, Base, get_session
from ..models import Security, Holding
from ..services.ingestion import ingest_eod_for_date, backfill_history
from ..services.reporting import build_weekly_report, build_daily_report
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
    ai_summary: bool = typer.Option(False, "--ai-summary", help="Append LLM summary at bottom"),
):
    """Generate Weekly Deep Dive HTML for WEEK_ENDING (default: today)."""
    d = parse_date(week_ending_value)
    out_dir = Path("reports") / d.strftime("%Y-%m-%d")
    out_html = out_dir / "weekly.html"
    out = build_weekly_report(d, str(out_html), fetch_outlook=outlook, ai_summary=ai_summary)
    typer.echo(out)

@app.command("report-daily")
def report_daily_cmd(
    as_of_value: str = typer.Argument("today", metavar="AS_OF", help="YYYY-MM-DD or 'today'"),
    ai_summary: bool = typer.Option(False, "--ai-summary", help="Append LLM summary to daily"),
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

if __name__ == "__main__":
    app()