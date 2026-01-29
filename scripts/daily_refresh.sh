#!/bin/bash
# =============================================================================
# Daily Portfolio Refresh Script
# =============================================================================
# This script automates the daily portfolio update workflow:
# 1. Finds the latest date with price data in the database
# 2. Backfills prices from the next day up to yesterday (or today if market closed)
# 3. Generates a weekly report as of the latest data date
# 4. Starts an HTTP server to view reports
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_DIR/portfolio.db"
VENV_PATH="$PROJECT_DIR/env_portfolio/bin/activate"
REPORTS_DIR="$PROJECT_DIR/reports"
HTTP_PORT="${HTTP_PORT:-8080}"
SEND_EMAIL="${SEND_EMAIL:-false}"  # Set to 'true' to send email notifications

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Portfolio Daily Refresh Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Find the latest date with prices
echo -e "${YELLOW}[Step 1]${NC} Finding latest price date in database..."
LAST_DATE=$(sqlite3 "$DB_PATH" "SELECT MAX(date) FROM prices;")

if [ -z "$LAST_DATE" ]; then
    echo -e "${RED}Error: No prices found in database. Run initial backfill first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Latest price date: $LAST_DATE${NC}"

# Step 2: Calculate next date and today's date
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
if [[ "$NEXT_DATE" > "$END_DATE" ]]; then
    echo -e "${GREEN}✓ Prices are already up to date. No backfill needed.${NC}"
else
    echo -e "${YELLOW}[Step 3]${NC} Running backfill from $NEXT_DATE to $END_DATE..."
    
    # Activate virtual environment and run backfill
    cd "$PROJECT_DIR"
    source "$VENV_PATH"
    python -m app.server.cli backfill "$NEXT_DATE" "$END_DATE"
    
    echo -e "${GREEN}✓ Backfill completed.${NC}"
fi

# Step 4: Verify new latest date
echo -e "${YELLOW}[Step 4]${NC} Verifying latest price date..."
NEW_LAST_DATE=$(sqlite3 "$DB_PATH" "SELECT MAX(date) FROM prices;")
echo -e "${GREEN}✓ New latest price date: $NEW_LAST_DATE${NC}"

# Step 5: Record daily position snapshot and returns
echo -e "${YELLOW}[Step 5]${NC} Recording daily position snapshot..."
cd "$PROJECT_DIR"
source "$VENV_PATH"
python -m app.server.cli rebuild-returns --start "$NEW_LAST_DATE" --end "$NEW_LAST_DATE"
echo -e "${GREEN}✓ Position snapshot recorded for $NEW_LAST_DATE${NC}"

# Step 6: Generate report (with AI summary and optional email)
echo -e "${YELLOW}[Step 6]${NC} Generating weekly report for $NEW_LAST_DATE with AI summary..."
cd "$PROJECT_DIR"
source "$VENV_PATH"

if [ "$SEND_EMAIL" = "true" ]; then
    echo -e "  Email notifications: ${GREEN}enabled${NC}"
    python -m app.server.cli report "$NEW_LAST_DATE" --ai-summary --email
else
    python -m app.server.cli report "$NEW_LAST_DATE" --ai-summary
fi
echo -e "${GREEN}✓ Report generated at: $REPORTS_DIR/$NEW_LAST_DATE/weekly.html${NC}"

# Step 7: Start HTTP server
echo -e "${YELLOW}[Step 7]${NC} Starting HTTP server on port $HTTP_PORT..."
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Reports available at:${NC}"
echo -e "  All reports:    http://localhost:$HTTP_PORT/"
echo -e "  Latest report:  http://localhost:$HTTP_PORT/$NEW_LAST_DATE/weekly.html"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

cd "$REPORTS_DIR"
python -m http.server "$HTTP_PORT"
