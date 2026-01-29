# app/services/market_brief.py
"""
Daily AI Market Brief Service
Fetches news from multiple sources, generates AI summary, and sends via email.
Covers: market events, portfolio holdings, geopolitical/macro events.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json

import httpx
from openai import OpenAI

from .news_providers.alpha_vantage import AlphaVantageNews
from .email_notify import send_portfolio_email
from ..config import settings
from ..db import get_session
from ..models import Holding

logger = logging.getLogger(__name__)


# Market-wide tickers for general market news
MARKET_TICKERS = ["SPY", "QQQ", "DIA", "IWM", "VIX"]

# Macro/geopolitical topics to search
MACRO_TOPICS = ["FED", "FOMC", "inflation", "tariff", "china", "oil", "treasury", "rate", "GDP", "jobs", "unemployment", "CPI", "PPI"]


def _iso_hours_ago_utc(hours: int) -> str:
    """Generate ISO timestamp for X hours ago."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y%m%dT%H%M%S")


def fetch_market_snapshot() -> Dict[str, Any]:
    """
    Fetch current market data from local database first, fallback to yfinance.
    Returns quick snapshot for context.
    """
    # Try to get from local DB first (from daily ingestion)
    try:
        from ..models import Price, BenchmarkPrice
        from ..db import get_session
        from datetime import date
        
        snapshot = {}
        today = date.today()
        
        market_tickers = {
            "SPY": "S&P 500",
            "QQQ": "Nasdaq 100",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "GLD": "Gold",
            "TLT": "20Y Treasury",
        }
        
        with get_session() as sess:
            for ticker, name in market_tickers.items():
                # Try regular prices first
                prices = sess.query(Price).filter(
                    Price.ticker == ticker
                ).order_by(Price.date.desc()).limit(2).all()
                
                if len(prices) >= 2:
                    current = float(prices[0].close)
                    prev = float(prices[1].close)
                    change_pct = ((current - prev) / prev) * 100
                    snapshot[name] = {
                        "price": round(current, 2),
                        "change_pct": round(change_pct, 2),
                        "as_of": str(prices[0].date),
                    }
                else:
                    # Try benchmark prices (SPY is stored there)
                    bench = sess.query(BenchmarkPrice).filter(
                        BenchmarkPrice.symbol == ticker
                    ).order_by(BenchmarkPrice.date.desc()).limit(2).all()
                    
                    if len(bench) >= 2 and bench[0].adj_close and bench[1].adj_close:
                        current = float(bench[0].adj_close)
                        prev = float(bench[1].adj_close)
                        change_pct = ((current - prev) / prev) * 100
                        snapshot[name] = {
                            "price": round(current, 2),
                            "change_pct": round(change_pct, 2),
                            "as_of": str(bench[0].date),
                        }
        
        if snapshot:
            logger.info(f"Got market snapshot from local DB: {len(snapshot)} indices")
            return snapshot
            
    except Exception as e:
        logger.debug(f"Local DB snapshot failed: {e}")
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        import time
        
        tickers = {
            "SPY": "S&P 500",
            "QQQ": "Nasdaq 100", 
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "^VIX": "VIX",
            "GLD": "Gold",
            "TLT": "20Y Treasury",
        }
        
        snapshot = {}
        
        # Fetch one at a time with small delay to avoid rate limits
        for ticker, name in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="2d")
                
                if hist is not None and len(hist) >= 2:
                    current = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2])
                    change_pct = ((current - prev) / prev) * 100
                    snapshot[name] = {
                        "price": round(current, 2),
                        "change_pct": round(change_pct, 2),
                    }
                time.sleep(0.2)  # Small delay between requests
            except Exception as e:
                logger.debug(f"Snapshot {ticker} failed: {e}")
                continue
        
        return snapshot
    except Exception as e:
        logger.debug(f"Market snapshot failed: {e}")
        return {}


def fetch_earnings_calendar(tickers: List[str], days_ahead: int = 5) -> List[Dict[str, Any]]:
    """
    Check upcoming earnings for portfolio holdings.
    Uses existing earnings service with fallback to yfinance.
    """
    upcoming = []
    today = datetime.now().date()
    cutoff = today + timedelta(days=days_ahead)
    
    # Try to use the existing earnings service first
    try:
        from .earnings import get_upcoming_earnings
        
        earnings_data = get_upcoming_earnings(tickers[:15], days_ahead=days_ahead)
        
        for item in earnings_data:
            ticker = item.get("ticker", "")
            ed_str = item.get("date", "")
            if ed_str:
                try:
                    ed = datetime.strptime(ed_str, "%Y-%m-%d").date()
                    if today <= ed <= cutoff:
                        upcoming.append({
                            "ticker": ticker,
                            "date": ed_str,
                            "days_until": (ed - today).days,
                            "when": item.get("when", ""),  # BMO/AMC
                        })
                except ValueError:
                    continue
        
        if upcoming:
            return sorted(upcoming, key=lambda x: x.get("days_until", 999))
    except Exception as e:
        logger.debug(f"Earnings service failed: {e}")
    
    # Fallback to yfinance
    try:
        import yfinance as yf
        import time
        
        # Only check top 10 holdings to save time and avoid rate limits
        for ticker in tickers[:10]:
            try:
                stock = yf.Ticker(ticker)
                cal = stock.calendar
                
                if cal is not None and not cal.empty:
                    # Calendar might have earnings date
                    if "Earnings Date" in cal.index:
                        earnings_dates = cal.loc["Earnings Date"]
                        if isinstance(earnings_dates, (list, tuple)) and len(earnings_dates) > 0:
                            ed = earnings_dates[0]
                            if hasattr(ed, 'date'):
                                ed = ed.date()
                            if today <= ed <= cutoff:
                                upcoming.append({
                                    "ticker": ticker,
                                    "date": str(ed),
                                    "days_until": (ed - today).days,
                                })
                time.sleep(0.3)  # Rate limit protection
            except Exception as e:
                logger.debug(f"Earnings check {ticker} failed: {e}")
                continue
        
        return sorted(upcoming, key=lambda x: x.get("days_until", 999))
    except Exception as e:
        logger.debug(f"Earnings calendar failed: {e}")
        return []


def fetch_economic_calendar() -> List[Dict[str, Any]]:
    """
    Simple economic calendar - key events to watch.
    """
    # Static list of recurring important events (could be enhanced with API)
    today = datetime.now()
    day_of_week = today.weekday()
    
    events = []
    
    # FOMC meetings are typically every 6 weeks
    # Jobs report first Friday of month
    # CPI mid-month
    
    if today.day <= 7 and day_of_week == 4:  # First Friday
        events.append({"event": "Jobs Report (likely)", "importance": "high"})
    
    if 10 <= today.day <= 15:
        events.append({"event": "CPI Release (likely)", "importance": "high"})
    
    # Add based on current market context
    return events


def fetch_portfolio_tickers() -> List[str]:
    """Get current portfolio tickers from database."""
    try:
        with get_session() as sess:
            holdings = sess.query(Holding).filter(Holding.shares > 0).all()
            return [h.ticker for h in holdings]
    except Exception as e:
        logger.error(f"Failed to fetch portfolio tickers: {e}")
        return []


def fetch_news_alpha_vantage(
    tickers: List[str],
    hours_back: int = 8,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Fetch news from Alpha Vantage for given tickers."""
    if not tickers:
        return []
    
    provider = AlphaVantageNews()
    time_from = _iso_hours_ago_utc(hours_back)
    
    try:
        raw = provider.fetch(tickers, time_from_iso=time_from, limit=limit)
        return raw
    except Exception as e:
        logger.error(f"Alpha Vantage news fetch failed: {e}")
        return []


def fetch_market_news_finnhub(hours_back: int = 8) -> List[Dict[str, Any]]:
    """
    Fetch general market news from Finnhub (free tier available).
    Falls back gracefully if no API key.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        logger.debug("FINNHUB_API_KEY not set, skipping Finnhub news")
        return []
    
    try:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": api_key}
        
        with httpx.Client(timeout=20) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Filter to recent news
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        news = []
        for item in data[:30]:  # Limit items
            ts = item.get("datetime", 0)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if dt >= cutoff:
                news.append({
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published_at": dt.isoformat(),
                    "category": item.get("category", "general"),
                })
        return news
    except Exception as e:
        logger.error(f"Finnhub news fetch failed: {e}")
        return []


def fetch_rss_headlines() -> List[Dict[str, Any]]:
    """
    Fetch headlines from free financial RSS feeds.
    No API key required.
    """
    import xml.etree.ElementTree as ET
    
    feeds = [
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("CNBC Top", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
        ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
        ("Investing.com News", "https://www.investing.com/rss/news.rss"),
    ]
    
    news = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
    
    for name, url in feeds:
        try:
            with httpx.Client(timeout=15) as client:
                response = client.get(url)
                response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            for item in root.findall(".//item")[:10]:
                title = item.find("title")
                link = item.find("link")
                desc = item.find("description")
                pub_date = item.find("pubDate")
                
                if title is not None and title.text:
                    news.append({
                        "headline": title.text.strip(),
                        "summary": desc.text.strip() if desc is not None and desc.text else "",
                        "source": name,
                        "url": link.text.strip() if link is not None and link.text else "",
                        "published_at": pub_date.text if pub_date is not None else "",
                        "category": "market",
                    })
        except Exception as e:
            logger.debug(f"RSS feed {name} failed: {e}")
            continue
    
    return news


def aggregate_news(
    portfolio_tickers: List[str],
    hours_back: int = 8
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate news from all sources into categories.
    Returns: {
        "market": [...],
        "portfolio": [...],
        "macro": [...],
    }
    """
    result = {
        "market": [],
        "portfolio": [],
        "macro": [],
    }
    
    # 1. Fetch portfolio-specific news (Alpha Vantage)
    if portfolio_tickers:
        # Prioritize top holdings (by typical weight)
        top_tickers = portfolio_tickers[:10]
        av_news = fetch_news_alpha_vantage(top_tickers, hours_back=hours_back, limit=30)
        
        for item in av_news:
            result["portfolio"].append({
                "ticker": item.get("ticker", ""),
                "headline": item.get("headline", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "sentiment": item.get("sentiment", 0),
                "summary": item.get("summary_raw", ""),
            })
    
    # 2. Fetch market-wide news (Alpha Vantage for indices)
    market_news = fetch_news_alpha_vantage(MARKET_TICKERS, hours_back=hours_back, limit=20)
    for item in market_news:
        result["market"].append({
            "ticker": item.get("ticker", ""),
            "headline": item.get("headline", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
            "sentiment": item.get("sentiment", 0),
        })
    
    # 3. Fetch macro/general news (Finnhub or RSS)
    finnhub_news = fetch_market_news_finnhub(hours_back=hours_back)
    if finnhub_news:
        for item in finnhub_news:
            # Check if headline contains macro keywords
            headline_lower = item.get("headline", "").lower()
            is_macro = any(topic.lower() in headline_lower for topic in MACRO_TOPICS)
            
            if is_macro:
                result["macro"].append(item)
            else:
                result["market"].append(item)
    
    # 4. Add RSS headlines for broader coverage
    rss_news = fetch_rss_headlines()
    for item in rss_news:
        headline_lower = item.get("headline", "").lower()
        is_macro = any(topic.lower() in headline_lower for topic in MACRO_TOPICS)
        
        if is_macro:
            result["macro"].append(item)
        else:
            result["market"].append(item)
    
    # Deduplicate by headline similarity
    for category in result:
        seen_headlines = set()
        unique = []
        for item in result[category]:
            headline = item.get("headline", "").lower()[:50]
            if headline not in seen_headlines:
                seen_headlines.add(headline)
                unique.append(item)
        result[category] = unique[:15]  # Limit per category
    
    return result


def generate_ai_market_brief(
    news_data: Dict[str, List[Dict[str, Any]]],
    portfolio_tickers: List[str],
    brief_type: str = "midday",  # "midday" or "afternoon"
    market_snapshot: Optional[Dict[str, Any]] = None,
    upcoming_earnings: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Use OpenAI to generate an actionable market brief.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not configured. Cannot generate AI brief."
    
    # Build context for the LLM
    context_parts = []
    
    # Current date/time
    now = datetime.now()
    context_parts.append(f"## DATE: {now.strftime('%A, %B %d, %Y')} - {now.strftime('%I:%M %p')}")
    
    # Market snapshot
    if market_snapshot:
        context_parts.append("\n## MARKET SNAPSHOT (Live)")
        for name, data in market_snapshot.items():
            change = data.get("change_pct", 0)
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            context_parts.append(f"- {name}: {data.get('price', 'N/A')} ({arrow}{abs(change):.2f}%)")
    
    # Upcoming earnings
    if upcoming_earnings:
        context_parts.append("\n## UPCOMING EARNINGS (Portfolio)")
        for e in upcoming_earnings[:5]:
            days = e.get("days_until", 0)
            when = "TODAY" if days == 0 else f"in {days} days" if days > 0 else f"{abs(days)} days ago"
            context_parts.append(f"- {e['ticker']}: {e['date']} ({when})")
    
    # Market news
    if news_data.get("market"):
        context_parts.append("\n## MARKET NEWS")
        for item in news_data["market"][:8]:
            sentiment = item.get("sentiment", "")
            sent_str = f" (sentiment: {sentiment})" if sentiment else ""
            context_parts.append(f"- {item['headline']}{sent_str} [{item.get('source', 'N/A')}]")
    
    # Portfolio-specific news
    if news_data.get("portfolio"):
        context_parts.append("\n## PORTFOLIO HOLDINGS NEWS")
        for item in news_data["portfolio"][:8]:
            ticker = item.get("ticker", "")
            sentiment = item.get("sentiment", "")
            sent_str = f" (sentiment: {sentiment})" if sentiment else ""
            context_parts.append(f"- [{ticker}] {item['headline']}{sent_str}")
    
    # Macro/geopolitical news
    if news_data.get("macro"):
        context_parts.append("\n## MACRO & GEOPOLITICAL")
        for item in news_data["macro"][:6]:
            context_parts.append(f"- {item['headline']} [{item.get('source', 'N/A')}]")
    
    news_context = "\n".join(context_parts)
    
    time_label = "Midday (12 PM)" if brief_type == "midday" else "Market Close (4:30 PM)"
    
    top_holdings = ', '.join(portfolio_tickers[:15])
    
    system_prompt = f"""You are a senior portfolio manager's market analyst assistant.
Write a concise, actionable {time_label} market brief.

Portfolio holdings include: {top_holdings}

STRUCTURE (use these exact headers):
📊 **MARKET MOOD** - One sentence: bullish/bearish/neutral with key driver
📈 **KEY MOVERS** - 2-3 bullet points on biggest market stories TODAY
🎯 **PORTFOLIO WATCH** - Holdings with news or earnings; what to watch
⚠️ **RISKS** - Macro/geopolitical risks if any
📋 **ACTION ITEMS** - 1-2 specific things to monitor or consider

RULES:
- Use data from the MARKET SNAPSHOT for current index levels
- Reference specific tickers in **bold** 
- Include percentage moves where available
- If a holding has earnings soon, mention it
- Be specific with numbers (e.g., "VIX at 18.5" not "VIX elevated")
- NO generic sign-offs, closings, or pleasantries
- NO "Best regards" or "Stay informed" type endings
- Keep total length under 350 words
- Today's date is provided in the context - use it
"""

    user_prompt = f"""Generate the {time_label} market brief based on this real-time data:

{news_context}

Focus on actionable intelligence. End immediately after ACTION ITEMS - no closing."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI market brief generation failed: {e}")
        return f"AI brief generation failed: {e}"


def send_market_brief_email(
    brief_content: str,
    brief_type: str = "midday",
    news_data: Optional[Dict] = None
) -> bool:
    """
    Send the market brief via email.
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    email_to = os.environ.get("EMAIL_TO")
    email_from = os.environ.get("EMAIL_FROM", smtp_user)
    
    if not all([smtp_user, smtp_password, email_to]):
        logger.error("Email configuration incomplete")
        return False
    
    time_label = "Midday" if brief_type == "midday" else "Close"
    today = datetime.now().strftime("%b %d")
    subject = f"📰 {time_label} Market Brief - {today}"
    
    # Build HTML email
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #1a1a2e; font-size: 24px; border-bottom: 2px solid #4361ee; padding-bottom: 10px; }}
            h2 {{ color: #4361ee; font-size: 16px; margin-top: 24px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 8px; }}
            .ticker {{ background: #e8f4f8; padding: 2px 6px; border-radius: 4px; font-weight: 600; }}
            .bullish {{ color: #10b981; }}
            .bearish {{ color: #ef4444; }}
            .neutral {{ color: #6b7280; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #6b7280; }}
            strong {{ color: #1a1a2e; }}
        </style>
    </head>
    <body>
        <h1>📰 {time_label} Market Brief</h1>
        <div style="white-space: pre-wrap; font-size: 14px;">
{brief_content}
        </div>
        <div class="footer">
            Generated by Portfolio AI Assistant<br>
            {datetime.now().strftime("%Y-%m-%d %H:%M %Z")}
        </div>
    </body>
    </html>
    """
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to
        
        # Plain text version
        msg.attach(MIMEText(brief_content, "plain"))
        # HTML version
        msg.attach(MIMEText(html_content, "html"))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(email_from, [email_to], msg.as_string())
        
        logger.info(f"Market brief email sent to {email_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to send market brief email: {e}")
        return False


def run_market_brief(brief_type: str = "midday") -> Dict[str, Any]:
    """
    Main function to run the market brief pipeline.
    Called by CLI or scheduler.
    
    Args:
        brief_type: "midday" (12 PM) or "afternoon" (4:30 PM)
    
    Returns:
        Dict with status and details
    """
    logger.info(f"Starting {brief_type} market brief generation...")
    
    # 1. Get portfolio tickers
    portfolio_tickers = fetch_portfolio_tickers()
    logger.info(f"Portfolio has {len(portfolio_tickers)} tickers")
    
    # 2. Aggregate news (look back further in morning, less in afternoon)
    hours_back = 12 if brief_type == "midday" else 6
    news_data = aggregate_news(portfolio_tickers, hours_back=hours_back)
    
    total_news = sum(len(v) for v in news_data.values())
    logger.info(f"Aggregated {total_news} news items")
    
    # 3. Fetch market snapshot
    logger.info("Fetching market snapshot...")
    market_snapshot = fetch_market_snapshot()
    logger.info(f"Got snapshot for {len(market_snapshot)} indices")
    
    # 4. Check upcoming earnings
    logger.info("Checking upcoming earnings...")
    upcoming_earnings = fetch_earnings_calendar(portfolio_tickers, days_ahead=5)
    logger.info(f"Found {len(upcoming_earnings)} upcoming earnings")
    
    # 5. Generate AI brief
    brief_content = generate_ai_market_brief(
        news_data, 
        portfolio_tickers, 
        brief_type,
        market_snapshot=market_snapshot,
        upcoming_earnings=upcoming_earnings,
    )
    
    # 6. Send email
    email_sent = send_market_brief_email(brief_content, brief_type, news_data)
    
    return {
        "status": "ok" if email_sent else "partial",
        "brief_type": brief_type,
        "news_count": total_news,
        "news_breakdown": {k: len(v) for k, v in news_data.items()},
        "market_snapshot": bool(market_snapshot),
        "upcoming_earnings": len(upcoming_earnings),
        "portfolio_tickers": len(portfolio_tickers),
        "email_sent": email_sent,
        "brief_preview": brief_content[:500] + "..." if len(brief_content) > 500 else brief_content,
    }


if __name__ == "__main__":
    # Test run
    import sys
    logging.basicConfig(level=logging.INFO)
    
    brief_type = sys.argv[1] if len(sys.argv) > 1 else "midday"
    result = run_market_brief(brief_type)
    print(json.dumps(result, indent=2))
