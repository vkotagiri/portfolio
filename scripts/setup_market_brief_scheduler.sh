#!/bin/bash
# Setup market brief schedulers for 12 PM and 4:30 PM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "📰 Setting up Market Brief Schedulers"
echo "======================================"

# Make script executable
chmod +x "$SCRIPT_DIR/market_brief.sh"

# Create LaunchAgents directory if needed
mkdir -p "$LAUNCH_AGENTS_DIR"
mkdir -p "$PROJECT_DIR/logs"

# Unload existing jobs if present
for plist in "com.portfolio.market-brief-midday" "com.portfolio.market-brief-afternoon"; do
    if launchctl list | grep -q "$plist"; then
        echo "Unloading existing $plist..."
        launchctl unload "$LAUNCH_AGENTS_DIR/$plist.plist" 2>/dev/null || true
    fi
done

# Copy plist files
echo "Installing scheduler files..."
cp "$SCRIPT_DIR/com.portfolio.market-brief-midday.plist" "$LAUNCH_AGENTS_DIR/"
cp "$SCRIPT_DIR/com.portfolio.market-brief-afternoon.plist" "$LAUNCH_AGENTS_DIR/"

# Load the jobs
echo "Loading schedulers..."
launchctl load "$LAUNCH_AGENTS_DIR/com.portfolio.market-brief-midday.plist"
launchctl load "$LAUNCH_AGENTS_DIR/com.portfolio.market-brief-afternoon.plist"

echo ""
echo "✅ Market Brief Schedulers Installed!"
echo ""
echo "Schedule:"
echo "  📌 Midday Brief:    12:00 PM daily"
echo "  📌 Afternoon Brief: 4:30 PM daily"
echo ""
echo "Commands:"
echo "  Test midday:    python -m app.server.cli market-brief midday"
echo "  Test afternoon: python -m app.server.cli market-brief afternoon"
echo "  Preview only:   python -m app.server.cli market-brief midday --no-email"
echo ""
echo "Logs: $PROJECT_DIR/logs/market_brief_*.log"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.portfolio.market-brief-midday.plist"
echo "  launchctl unload ~/Library/LaunchAgents/com.portfolio.market-brief-afternoon.plist"
