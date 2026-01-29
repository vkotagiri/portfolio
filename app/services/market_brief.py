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
MACRO_TOPICS = ["FED", "FOMC", "inflation", "tariff", "china", "oil", "treasury"]


def _iso_hours_ago_utc(hours: int) -> str:
    """Generate ISO timestamp for X hours ago."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y%m%dT%H%M%S")


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
        ("Reuters Business", "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"),
        ("CNBC Top", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
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
    brief_type: str = "midday"  # "midday" or "afternoon"
) -> str:
    """
    Use OpenAI to generate an actionable market brief.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not configured. Cannot generate AI brief."
    
    # Build context for the LLM
    context_parts = []
    
    # Market news
    if news_data.get("market"):
        context_parts.append("## MARKET NEWS")
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
    
    system_prompt = f"""You are a senior portfolio manager's market analyst assistant.
Write a concise, actionable {time_label} market brief email.

Portfolio holdings: {', '.join(portfolio_tickers[:15])}

Your brief should:
1. START with a 1-sentence market mood summary (bullish/bearish/mixed)
2. Highlight 2-3 most important market-moving stories
3. Flag any news affecting portfolio holdings with actionable context
4. Note any geopolitical/macro risks to monitor
5. End with 1-2 specific watchlist items or action items

Format as a professional email - brief, scannable, no fluff.
Use bullet points and bold for key items.
Keep total length under 400 words.
"""

    user_prompt = f"""Generate the {time_label} market brief based on this news:

{news_context}

Remember: Be concise, actionable, and focus on what matters for the portfolio."""

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
    
    # 3. Generate AI brief
    brief_content = generate_ai_market_brief(news_data, portfolio_tickers, brief_type)
    
    # 4. Send email
    email_sent = send_market_brief_email(brief_content, brief_type, news_data)
    
    return {
        "status": "ok" if email_sent else "partial",
        "brief_type": brief_type,
        "news_count": total_news,
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
