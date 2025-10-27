#!/usr/bin/env bash
# scripts/full_refresh.sh
# Full pipeline: clean -> (optional) install -> init DB -> load holdings -> backfill -> ingest -> report
# Calendar-aware: maps dates to NYSE sessions to avoid weekend/holiday noise.

set -Eeuo pipefail
trap 'code=$?; echo -e "\n[ERROR] Exit $code at line $LINENO: ${BASH_COMMAND}" >&2' ERR
DEBUG="${DEBUG:-0}"; [[ "$DEBUG" == "1" ]] && set -x

# --- Paths ---
SCRIPT_FILE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_FILE")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# --- Logger (portable) ---
log() { printf "\n\033[1;34m[%s]\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { echo "Error: $*" >&2; exit 1; }

# --- Defaults (overridable via flags/env) ---
VENV="${VENV:-env_portfolio}"
CSV="${CSV:-data/holdings_sample.csv}"
DB="${DB:-portfolio.db}"
START="${START:-2024-01-01}"
END="${END:-today}"
WEEK="${WEEK:-today}"
OFFLINE_OVERRIDE="${OFFLINE_OVERRIDE:-}"
INSTALL="${INSTALL:-0}"
FRESH=0

usage() {
  cat <<USAGE
Usage: $0 [options]
  --fresh                 Delete $DB before initializing
  --install               pip install -r requirements.txt (and upgrade pip)
  --start YYYY-MM-DD      Backfill start (default: $START) [calendar-aware → first session ≥ date]
  --end YYYY-MM-DD|today  Backfill end (default: $END)     [calendar-aware → last session ≤ date]
  --week YYYY-MM-DD|today Week ending (default: $WEEK)     [calendar-aware → last session ≤ date]
  --csv <path>            Holdings CSV (default: $CSV)
  --venv <path>           Virtualenv folder (default: $VENV)
  --offline 0|1           Override OFFLINE for this run (overrides .env)
  --debug                 Enable shell tracing (set -x)
  -h, --help              This help
USAGE
}

# --- Args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift ;;
    --install) INSTALL=1; shift ;;
    --start) START="${2:?}"; shift 2 ;;
    --end) END="${2:?}"; shift 2 ;;
    --week) WEEK="${2:?}"; shift 2 ;;
    --csv) CSV="${2:?}"; shift 2 ;;
    --venv) VENV="${2:?}"; shift 2 ;;
    --offline) OFFLINE_OVERRIDE="${2:?}"; shift 2 ;;
    --debug) set -x; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

log "Starting refresh in $ROOT_DIR"

# --- Load .env ---
if [[ -f .env ]]; then
  log "Loading .env"
  set -a; source .env; set +a
fi
# One-run OFFLINE override
if [[ -n "${OFFLINE_OVERRIDE}" ]]; then
  export OFFLINE="${OFFLINE_OVERRIDE}"
  log "OFFLINE overridden: ${OFFLINE}"
fi

# --- Ensure venv ---
if [[ ! -d "$VENV" ]]; then
  log "Creating venv: $VENV"
  command -v python3.11 >/dev/null 2>&1 || die "python3.11 not found in PATH"
  python3.11 -m venv "$VENV"
  INSTALL=1
fi
log "Activating venv: $VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# --- Python version check ---
python - <<'PY' || exit 1
import sys
major, minor = sys.version_info[:2]
assert major==3 and minor>=10, f"Need Python >=3.10, found {major}.{minor}"
print(f"Using Python {major}.{minor}")
PY

# --- Install requirements if requested/new venv ---
if [[ "$INSTALL" == "1" ]]; then
  log "Installing requirements"
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt --upgrade
fi

# --- Sanity ---
[[ -f "$CSV" ]] || die "Holdings CSV not found: $CSV"
if [[ "${OFFLINE:-0}" != "1" ]]; then
  if [[ -n "${TIINGO_API_KEY:-}" ]]; then
    log "Tiingo key detected → Tiingo PRIMARY (rate-limit aware), yfinance fallback."
  else
    log "No TIINGO_API_KEY set → using yfinance fallback (and/or Mock if enabled)."
  fi
else
  log "OFFLINE=1 → Mock provider only."
fi

# --- Calendar helpers (NYSE via pandas-market-calendars) ---
resolve_last_trading_day() {
  python - "$1" <<'PY'
import sys
from datetime import date, timedelta, datetime
try:
    import zoneinfo
    TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    TZ = None
import pandas_market_calendars as pmc

def parse(s: str) -> date:
    s = s.strip().lower()
    if s == "today":
        return (datetime.now(TZ).date() if TZ else date.today())
    return date.fromisoformat(s)

d = parse(sys.argv[1])
nyse = pmc.get_calendar("NYSE")
start = d - timedelta(days=21)      # look back ~3 weeks
sched = nyse.schedule(start_date=start, end_date=d)
print((sched.index[-1].date() if not sched.empty else d).isoformat())
PY
}

resolve_first_trading_day_on_or_after() {
  python - "$1" <<'PY'
import sys
from datetime import date, timedelta, datetime
try:
    import zoneinfo
    TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    TZ = None
import pandas_market_calendars as pmc

def parse(s: str) -> date:
    s = s.strip().lower()
    if s == "today":
        return (datetime.now(TZ).date() if TZ else date.today())
    return date.fromisoformat(s)

d = parse(sys.argv[1])
nyse = pmc.get_calendar("NYSE")
end = d + timedelta(days=21)        # look forward ~3 weeks
sched = nyse.schedule(start_date=d, end_date=end)
print((sched.index[0].date() if not sched.empty else d).isoformat())
PY
}

# --- Resolve calendar-aware dates ---
RESOLVED_START="$(resolve_first_trading_day_on_or_after "$START")"
RESOLVED_END="$(resolve_last_trading_day "$END")"
RESOLVED_WEEK="$(resolve_last_trading_day "$WEEK")"

log "Calendar-adjusted START: $RESOLVED_START  (from: $START)"
log "Calendar-adjusted END:   $RESOLVED_END    (from: $END)"
log "Calendar-adjusted WEEK:  $RESOLVED_WEEK   (from: $WEEK)"

# --- Clean caches ---
log "Cleaning __pycache__ and *.pyc"
find app -name "__pycache__" -type d -exec rm -rf {} + || true
find . -name "*.pyc" -delete || true

# --- Fresh DB if requested ---
if [[ "$FRESH" -eq 1 ]]; then
  log "Deleting DB: $DB"
  rm -f "$DB"
fi

# --- Pipeline ---
log "Init DB"
python -m app.server.cli init-db

log "Load holdings: $CSV"
python -m app.server.cli load-holdings "$CSV"

log "Backfill: $RESOLVED_START → $RESOLVED_END"
python -m app.server.cli backfill "$RESOLVED_START" "$RESOLVED_END"

log "Ingest EOD (END=$RESOLVED_END)"
python -m app.server.cli ingest-eod "$RESOLVED_END"

log "Generate report: $RESOLVED_WEEK"
python -m app.server.cli report "$RESOLVED_WEEK"

OUT="reports/${RESOLVED_WEEK}/weekly.html"
log "Done. Report at: ${OUT}"
if command -v open >/dev/null 2>&1 && [[ -f "$OUT" ]]; then open "$OUT" || true; fi
