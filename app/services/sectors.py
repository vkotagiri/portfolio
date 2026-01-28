# app/services/sectors.py
"""
Sector and industry classification service.
Fetches and caches sector data from Yahoo Finance.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..db import get_session
from ..models import Security

logger = logging.getLogger(__name__)

# Static sector mapping for common tickers (fallback when API fails)
STATIC_SECTORS = {
    # Technology
    "AAPL": ("Technology", "Consumer Electronics"),
    "MSFT": ("Technology", "Software—Infrastructure"),
    "GOOG": ("Communication Services", "Internet Content & Information"),
    "GOOGL": ("Communication Services", "Internet Content & Information"),
    "AMZN": ("Consumer Cyclical", "Internet Retail"),
    "META": ("Communication Services", "Internet Content & Information"),
    "NVDA": ("Technology", "Semiconductors"),
    "AMD": ("Technology", "Semiconductors"),
    "AVGO": ("Technology", "Semiconductors"),
    "MU": ("Technology", "Semiconductors"),
    "LRCX": ("Technology", "Semiconductor Equipment"),
    "MCHP": ("Technology", "Semiconductors"),
    "INTC": ("Technology", "Semiconductors"),
    "TSM": ("Technology", "Semiconductors"),
    "TSLA": ("Consumer Cyclical", "Auto Manufacturers"),
    "DELL": ("Technology", "Computer Hardware"),
    "CRWD": ("Technology", "Software—Infrastructure"),
    "PANW": ("Technology", "Software—Infrastructure"),
    "SNOW": ("Technology", "Software—Application"),
    "APP": ("Technology", "Software—Application"),
    
    # Financials
    "JPM": ("Financial Services", "Banks—Diversified"),
    "V": ("Financial Services", "Credit Services"),
    "MA": ("Financial Services", "Credit Services"),
    "COF": ("Financial Services", "Credit Services"),
    "HOOD": ("Financial Services", "Capital Markets"),
    
    # Healthcare
    "ISRG": ("Healthcare", "Medical Instruments & Supplies"),
    "GEHC": ("Healthcare", "Medical Devices"),
    
    # Consumer
    "HD": ("Consumer Cyclical", "Home Improvement Retail"),
    "WMT": ("Consumer Defensive", "Discount Stores"),
    "COST": ("Consumer Defensive", "Discount Stores"),
    "NKE": ("Consumer Cyclical", "Footwear & Accessories"),
    "CMG": ("Consumer Cyclical", "Restaurants"),
    "CCL": ("Consumer Cyclical", "Travel Services"),
    "RACE": ("Consumer Cyclical", "Auto Manufacturers"),
    
    # Real Estate
    "SBRA": ("Real Estate", "REIT—Healthcare Facilities"),
    
    # Energy / Industrials
    "APLD": ("Technology", "Information Technology Services"),
    "WULF": ("Financial Services", "Capital Markets"),
    "XLI": ("ETF", "Industrials Sector"),
    "QS": ("Consumer Cyclical", "Auto Parts"),
    "CRCL": ("Healthcare", "Medical Devices"),
    
    # ETFs & Fixed Income
    "BIL": ("Fixed Income", "Treasury Bills"),
    "GLD": ("Commodities", "Gold"),
    "SIVR": ("Commodities", "Silver"),
    "SLV": ("Commodities", "Silver"),
    "COPX": ("ETF", "Copper Miners"),
    "SPY": ("ETF", "S&P 500 Index"),
    "QQQ": ("ETF", "Nasdaq 100 Index"),
    "VTI": ("ETF", "Total Stock Market"),
    "VOO": ("ETF", "S&P 500 Index"),
    "VUG": ("ETF", "Growth"),
    "VEA": ("ETF", "Developed Markets ex-US"),
    "VXUS": ("ETF", "International"),
    "INDA": ("ETF", "India"),
    "SCHA": ("ETF", "Small Cap"),
    "SQQQ": ("ETF", "Inverse Nasdaq"),
    "ITDB": ("ETF", "Target Date"),
}

# GICS Sector color mapping for visualization
SECTOR_COLORS = {
    "Technology": "#3b82f6",        # Blue
    "Information Technology": "#3b82f6",
    "Healthcare": "#10b981",        # Green
    "Health Care": "#10b981",
    "Financials": "#8b5cf6",        # Purple
    "Financial Services": "#8b5cf6",
    "Consumer Cyclical": "#f59e0b", # Yellow/Orange
    "Consumer Discretionary": "#f59e0b",
    "Communication Services": "#06b6d4",  # Cyan
    "Industrials": "#6b7280",       # Gray
    "Consumer Defensive": "#84cc16", # Lime
    "Consumer Staples": "#84cc16",
    "Energy": "#ef4444",            # Red
    "Utilities": "#a855f7",         # Light Purple
    "Real Estate": "#ec4899",       # Pink
    "Basic Materials": "#78716c",   # Stone
    "Materials": "#78716c",
    "Commodities": "#eab308",       # Yellow
    "ETF": "#64748b",               # Slate
    "Other": "#9ca3af",             # Default gray
    "Cash": "#22c55e",              # Green for cash/bonds
    "Fixed Income": "#22c55e",
    "Unknown": "#9ca3af",
}


def get_sector_color(sector: str) -> str:
    """Get color for a sector."""
    return SECTOR_COLORS.get(sector, SECTOR_COLORS["Other"])


def fetch_sector_for_ticker(ticker: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch sector and industry for a single ticker.
    Uses static mapping first, then Yahoo Finance as fallback.
    
    Returns: (sector, industry) or (None, None) if not available
    """
    # Check static mapping first (faster, no API call)
    if ticker in STATIC_SECTORS:
        return STATIC_SECTORS[ticker]
    
    # Handle special cases for known ETF patterns
    if ticker in ("BIL", "SHY", "TLT", "IEF", "SGOV"):
        return "Fixed Income", "Treasury Bonds"
    if ticker in ("GLD", "IAU", "SLV", "SIVR"):
        return "Commodities", "Precious Metals"
    if ticker.startswith("V") and len(ticker) <= 4:  # Vanguard ETFs
        return "ETF", "Index Fund"
    
    # Try Yahoo Finance (may fail due to rate limits)
    try:
        import yfinance as yf
        
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        industry = info.get("industry")
        
        if sector:
            return sector, industry
        
        # Check if it's an ETF
        quote_type = info.get("quoteType", "")
        if quote_type == "ETF":
            return "ETF", info.get("category", "Index Fund")
            
    except Exception as e:
        logger.debug(f"YFinance lookup failed for {ticker}: {e}")
    
    return None, None


def update_security_sector(sess: Session, ticker: str, sector: str, industry: str) -> None:
    """Update sector and industry for a security in the database."""
    sec = sess.get(Security, ticker)
    if sec:
        sec.sector = sector
        sec.industry = industry
        sec.sector_updated = datetime.now().isoformat()


def refresh_all_sectors(force: bool = False) -> Dict[str, str]:
    """
    Refresh sector data for all securities.
    
    Args:
        force: If True, refresh even if sector data exists
        
    Returns: Dict mapping ticker to sector
    """
    import time
    
    results = {}
    
    with get_session() as sess:
        securities = sess.query(Security).all()
        
        for sec in securities:
            # Skip if already has sector and not forcing
            if sec.sector and not force:
                results[sec.ticker] = sec.sector
                continue
            
            sector, industry = fetch_sector_for_ticker(sec.ticker)
            
            if sector:
                sec.sector = sector
                sec.industry = industry
                sec.sector_updated = datetime.now().isoformat()
                results[sec.ticker] = sector
                logger.info(f"Updated {sec.ticker}: {sector} / {industry}")
            else:
                results[sec.ticker] = "Unknown"
            
            # Rate limit
            time.sleep(0.3)
        
        sess.commit()
    
    return results


def get_sector_breakdown(
    holdings_with_weights: List[Dict],
    sess: Session
) -> Dict[str, Dict]:
    """
    Calculate sector breakdown from holdings.
    
    Args:
        holdings_with_weights: List of dicts with 'ticker' and 'weight' keys
        sess: Database session
        
    Returns: Dict with sector breakdown data
    """
    sector_weights: Dict[str, float] = {}
    sector_tickers: Dict[str, List[str]] = {}
    sector_count: Dict[str, int] = {}
    unknown_tickers = []
    
    for h in holdings_with_weights:
        ticker = h.get("ticker")
        weight = h.get("weight", 0)
        
        if not ticker:
            continue
            
        # Get sector from database
        sec = sess.get(Security, ticker)
        sector = sec.sector if sec and sec.sector else None
        
        # Fetch if missing
        if not sector:
            sector, industry = fetch_sector_for_ticker(ticker)
            if sector and sec:
                sec.sector = sector
                sec.industry = industry
                sec.sector_updated = datetime.now().isoformat()
        
        if not sector:
            sector = "Unknown"
            unknown_tickers.append(ticker)
        
        # Aggregate
        sector_weights[sector] = sector_weights.get(sector, 0) + weight
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append(ticker)
        sector_count[sector] = sector_count.get(sector, 0) + 1
    
    # Sort by weight descending
    sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
    
    # Build result
    breakdown = []
    for sector, weight in sorted_sectors:
        breakdown.append({
            "sector": sector,
            "weight": weight,
            "weight_pct": f"{weight * 100:.1f}%",
            "count": sector_count[sector],
            "tickers": sorted(sector_tickers[sector]),
            "color": get_sector_color(sector),
        })
    
    # Commit any sector updates
    sess.commit()
    
    return {
        "breakdown": breakdown,
        "total_sectors": len(breakdown),
        "unknown_tickers": unknown_tickers,
        "largest_sector": breakdown[0] if breakdown else None,
    }
