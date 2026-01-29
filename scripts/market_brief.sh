#!/bin/bash
# Market Brief script - generates AI news summary and sends email
# Run at 12 PM and 4:30 PM on weekdays

set -e

# Determine brief type based on time
HOUR=$(date +%H)
if [ "$HOUR" -lt 14 ]; then
    BRIEF_TYPE="midday"
else
    BRIEF_TYPE="afternoon"
fi

# Allow override via argument
if [ -n "$1" ]; then
    BRIEF_TYPE="$1"
fi

cd /Users/venkat/work/agentic-portfolio
source env_portfolio/bin/activate

# Set up logging
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/market_brief_$(date +%Y%m%d_%H%M).log"

echo "Starting $BRIEF_TYPE market brief at $(date)" | tee -a "$LOG_FILE"

# Run the market brief command
python -m app.server.cli market-brief "$BRIEF_TYPE" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Market brief completed successfully at $(date)" | tee -a "$LOG_FILE"
else
    echo "Market brief failed with exit code $EXIT_CODE at $(date)" | tee -a "$LOG_FILE"
    
    # Send error notification email
    python -c "
from app.services.market_brief import send_market_brief_email
send_market_brief_email(
    'Market brief generation failed. Check logs at $LOG_FILE',
    brief_type='$BRIEF_TYPE'
)
" 2>/dev/null || true
fi

exit $EXIT_CODE
