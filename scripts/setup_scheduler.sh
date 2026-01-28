#!/bin/bash
# Setup script for scheduling daily portfolio refresh at 7 PM
# Usage: ./setup_scheduler.sh [install|uninstall|status]

PLIST_NAME="com.portfolio.daily-refresh.plist"
PLIST_SRC="$(dirname "$0")/$PLIST_NAME"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

case "$1" in
    install)
        echo "📅 Installing daily portfolio refresh scheduler..."
        
        # Create LaunchAgents directory if it doesn't exist
        mkdir -p "$HOME/Library/LaunchAgents"
        
        # Copy plist to LaunchAgents
        cp "$PLIST_SRC" "$PLIST_DEST"
        
        # Load the job
        launchctl load "$PLIST_DEST"
        
        echo "✅ Scheduler installed! Portfolio refresh will run daily at 7:00 PM."
        echo ""
        echo "📋 To check status:  ./setup_scheduler.sh status"
        echo "🗑️  To uninstall:    ./setup_scheduler.sh uninstall"
        echo "▶️  To run now:       launchctl start com.portfolio.daily-refresh"
        ;;
        
    uninstall)
        echo "🗑️  Uninstalling daily portfolio refresh scheduler..."
        
        # Unload the job
        launchctl unload "$PLIST_DEST" 2>/dev/null
        
        # Remove the plist
        rm -f "$PLIST_DEST"
        
        echo "✅ Scheduler uninstalled."
        ;;
        
    status)
        echo "📊 Scheduler Status:"
        echo "===================="
        
        if [ -f "$PLIST_DEST" ]; then
            echo "✅ Plist installed at: $PLIST_DEST"
            
            # Check if loaded
            if launchctl list | grep -q "com.portfolio.daily-refresh"; then
                echo "✅ Job is loaded and active"
                launchctl list | grep "com.portfolio.daily-refresh"
            else
                echo "⚠️  Job is installed but not loaded"
                echo "   Run: launchctl load $PLIST_DEST"
            fi
        else
            echo "❌ Scheduler not installed"
            echo "   Run: ./setup_scheduler.sh install"
        fi
        
        echo ""
        echo "📁 Log files:"
        echo "   Output: ~/work/agentic-portfolio/logs/daily_refresh.log"
        echo "   Errors: ~/work/agentic-portfolio/logs/daily_refresh_error.log"
        ;;
        
    run)
        echo "▶️  Running portfolio refresh now..."
        launchctl start com.portfolio.daily-refresh
        echo "✅ Job triggered. Check logs for output."
        ;;
        
    *)
        echo "Portfolio Daily Refresh Scheduler"
        echo "=================================="
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  install    Install and enable the daily 7 PM scheduler"
        echo "  uninstall  Remove the scheduler"
        echo "  status     Check if scheduler is installed and running"
        echo "  run        Trigger the refresh job immediately"
        echo ""
        echo "Schedule: Daily at 7:00 PM local time"
        ;;
esac
