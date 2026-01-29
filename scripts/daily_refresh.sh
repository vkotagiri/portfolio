#!/bin/bash
# =============================================================================
# Daily Portfolio Refresh Script
# =============================================================================
# This script automates the daily portfolio update workflow:
# 1. Finds the latest date with price data in the database
# 2. Backfills prices from the next day up to yesterday (or today if market closed)
# 3. Generates a weekly report as of the latest data date
# 4. Sends email notification (success or failure)
# 5. Optionally starts an HTTP server to view reports
# =============================================================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_DIR/portfolio.db"
VENV_PATH="$PROJECT_DIR/env_portfolio/bin/activate"
REPORTS_DIR="$PROJECT_DIR/reports"
HTTP_PORT="${HTTP_PORT:-8080}"
SEND_EMAIL="${SEND_EMAIL:-false}"
START_HTTP_SERVER="${START_HTTP_SERVER:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track state for error reporting
CURRENT_STEP=""
NEW_LAST_DATE=""

# Error handler - sends email notification on failure
send_error_email() {
    local error_msg="$1"
    local report_date="${NEW_LAST_DATE:-$(date '+%Y-%m-%d')}"
    
    echo -e "${RED}✗ Error in ${CURRENT_STEP}: ${error_msg}${NC}"
    
    if [ "$SEND_EMAIL" = "true" ]; then
        echo -e "${YELLOW}Sending error notification email...${NC}"
        cd "$PROJECT_DIR"
        source "$VENV_PATH" 2>/dev/null || true
        
        python3 << PYEOF
from app.config import settings
from app.services.email_notify import send_error_notification
from datetime import date

error_text = """Step: ${CURRENT_STEP}
Error: ${error_msg}
Time: $(date '+%Y-%m-%d %H:%M:%S')
Host: $(hostname)"""

result = send_error_notification(
    error_message=error_text,
    report_date=date.fromisoformat("${report_date}"),
    smtp_host=settings.smtp_host,
    smtp_port=settings.smtp_port,
    smtp_user=settings.smtp_user,
    smtp_password=settings.smtp_password,
    to_email=settings.email_to,
    from_email=settings.email_from,
    use_tls=settings.smtp_use_tls
)
print(f"Error notification: {result}")
PYEOF
    fi
    exit 1
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Portfolio Daily Refresh Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Find the latest date with prices
CURRENT_STEP="Step 1 - Find latest price date"
echo -e "${YELLOW}[Step 1]${NC} Finding latest price date in database..."
LAST_DATE=$(sqlite3 "$DB_PATH" "SELECT MAX(date) FROM prices;" 2>&1)

if [ $? -ne 0 ] || [ -z "$LAST_DATE" ]; then
    send_error_email "Failed to query database or no prices found: $LAST_DATE"
fi

echo -e "${GREEN}✓ Latest price date: $LAST_DATE${NC}"

# Step 2: Calculate next date and today's date
CURRENT_STEP="Step 2 - Calculate dates"
echo -e "${YELLOW}[Step 2]${NC} Calculating date range for backfill..."

# Get next day after last date (macOS compatible)
if [[ "$OSTYPE" == "darwin"* ]]; then
    NEXT_DATE=$(date -j -v+1d -f "%Y-%m-%d" "$LAST_DATE" "+%Y-%m-%d")
    TODAY=$(date "+%Y-%m-%d")
    YESTERDAY=$(date -j -v-1d "+%Y-%m-%d")
else
    NEXT_DATE=$(date -d "$LAST_DATE + 1 day" "+%Y-%m-%d")
    TODAY=$(date "+%Y-%m-%d")
    YESTERDAY=$(date -d "yesterday" "+%Y-%m-%d")
fi

# Use yesterday as end date (market may not have closed today yet)
END_DATE="$YESTERDAY"

echo -e "  Next date to fetch: $NEXT_DATE"
echo -e "  End date (yesterday): $END_DATE"

# Step 3: Run backfill if needed
CURRENT_STEP="Step 3 - Backfill prices"
if [[ "$NEXT_DATE" > "$END_DATE" ]]; then
    echo -e "${GREEN}✓ Prices are already up to date. No backfill needed.${NC}"
else
    echo -e "${YELLOW}[Step 3]${NC} Running backfill from $NEXT_DATE to $END_DATE..."
    
    cd "$PROJECT_DIR"
    source "$VENV_PATH"
    if ! python -m app.server.cli backfill "$NEXT_DATE" "$END_DATE"; then
        send_error_email "Backfill command failed for $NEXT_DATE to $END_DATE"
    fi
    
    echo -e "${GREEN}✓ Backfill completed.${NC}"
fi

# Step 4: Verify new latest date
CURRENT_STEP="Step 4 - Verify price data"
echo -e "${YELLOW}[Step 4]${NC} Verifying latest price date..."
NEW_LAST_DATE=$(sqlite3 "$DB_PATH" "SELECT MAX(date) FROM prices;" 2>&1)
if [ $? -ne 0 ] || [ -z "$NEW_LAST_DATE" ]; then
    send_error_email "Failed to verify latest price date: $NEW_LAST_DATE"
fi
echo -e "${GREEN}✓ New latest price date: $NEW_LAST_DATE${NC}"

# Step 5: Record daily position snapshot and returns
CURRENT_STEP="Step 5 - Record position snapshot"
echo -e "${YELLOW}[Step 5]${NC} Recording daily position snapshot..."
cd "$PROJECT_DIR"
source "$VENV_PATH"
if ! python -m app.server.cli rebuild-returns --start "$NEW_LAST_DATE" --end "$NEW_LAST_DATE"; then
    send_error_email "Failed to rebuild returns for $NEW_LAST_DATE"
fi
echo -e "${GREEN}✓ Position snapshot recorded for $NEW_LAST_DATE${NC}"

# Step 6: Generate report (with AI summary and optional email)
CURRENT_STEP="Step 6 - Generate report"
echo -e "${YELLOW}[Step 6]${NC} Generating weekly report for $NEW_LAST_DATE with AI summary..."
cd "$PROJECT_DIR"
source "$VENV_PATH"

if [ "$SEND_EMAIL" = "true" ]; then
    echo -e "  Email notifications: ${GREEN}enabled${NC}"
    # The --email flag handles both success email AND error notification if it fails
    if ! python -m app.server.cli report "$NEW_LAST_DATE" --ai-summary --email; then
        # Error email already sent by the CLI command
        exit 1
    fi
else
    if ! python -m app.server.cli report "$NEW_LAST_DATE" --ai-summary; then
        send_error_email "Report generation failed for $NEW_LAST_DATE"
    fi
fi
echo -e "${GREEN}✓ Report generated at: $REPORTS_DIR/$NEW_LAST_DATE/weekly.html${NC}"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Daily refresh completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 7: Start HTTP server (optional - disabled by default for scheduled runs)
if [ "$START_HTTP_SERVER" = "true" ]; then
    echo -e "${YELLOW}[Step 7]${NC} Starting HTTP server on port $HTTP_PORT..."
    echo ""
    echo -e "${GREEN}Reports available at:${NC}"
    echo -e "  All reports:    http://localhost:$HTTP_PORT/"
    echo -e "  Latest report:  http://localhost:$HTTP_PORT/$NEW_LAST_DATE/weekly.html"
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    
    cd "$REPORTS_DIR"
    python -m http.server "$HTTP_PORT"
fi
