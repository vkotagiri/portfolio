# Agentic Portfolio (Local, Free-First)

This scaffolds a **local agentic system** to ingest **evening adjusted closes**, compute analytics, validate strictly (print **"Data not available"** where required), and render a **single-page HTML** report plus an **infographic**. Start simple (SQLite), scale later (Postgres) without rewrites.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
python -m app.server.cli init-db
python -m app.server.cli load-holdings data/holdings_sample.csv
python -m app.server.cli ingest-eod --date today
python -m app.server.cli report --week-ending today

uvicorn app.server.api:app --reload --port 8000  # http://127.0.0.1:8000/report
```

Switch to Postgres later by setting `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/agentic`.

## Layout
- `app/config.py` — settings
- `app/db.py` — engine/session
- `app/models.py` — ORM models
- `app/services/*` — ingestion/metrics/validation/reporting/providers
- `app/server/api.py` — FastAPI server
- `app/server/cli.py` — Typer CLI
- `app/server/templates/report.html.j2` — HTML template
- `data/holdings_sample.csv` — sample portfolio
