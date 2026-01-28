# app/services/email_notify.py
"""
Email notification service for portfolio reports.
Sends AI-generated portfolio manager briefs via email.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _generate_pm_brief(report_data: Dict[str, Any], api_key: Optional[str], report_url: Optional[str] = None) -> str:
    """
    Generate a professional Portfolio Manager brief using AI.
    This is a concise, action-oriented summary suitable for email.
    """
    if not api_key or OpenAI is None:
        return _generate_fallback_brief(report_data, report_url)
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Build context from report data
        context_lines = []
        context_lines.append(f"Report Date: {report_data.get('as_of', 'N/A')}")
        context_lines.append(f"Portfolio Return: {report_data.get('port_ret', 'N/A')}")
        context_lines.append(f"Benchmark (SPY): {report_data.get('spy_ret', 'N/A')}")
        
        # Risk metrics
        risk = report_data.get("risk") or {}
        if risk:
            context_lines.append(f"Beta: {risk.get('beta', 'N/A')}")
            context_lines.append(f"Sharpe: {risk.get('sharpe', 'N/A')}")
            context_lines.append(f"Max Drawdown: {risk.get('max_dd', 'N/A')}")
        
        # Top/bottom contributors
        if report_data.get("top"):
            top_str = ", ".join([f"{x['ticker']} ({x['contrib']})" for x in report_data["top"][:3]])
            context_lines.append(f"Top Contributors: {top_str}")
        if report_data.get("bot"):
            bot_str = ", ".join([f"{x['ticker']} ({x['contrib']})" for x in report_data["bot"][:3]])
            context_lines.append(f"Bottom Contributors: {bot_str}")
        
        # Concentration
        conc = report_data.get("concentration") or {}
        if conc:
            context_lines.append(f"Largest Position: {conc.get('largest_ticker')} at {conc.get('largest_weight')}")
            context_lines.append(f"Top 5 Concentration: {conc.get('top5')}")
        
        # Correlation/Diversification
        corr = report_data.get("correlation") or {}
        if corr:
            context_lines.append(f"Diversification Score: {corr.get('diversification_score')}/100")
            if corr.get("redundant"):
                redundant = corr["redundant"][:3]
                red_str = ", ".join([f"{r['ticker1']}/{r['ticker2']}" for r in redundant])
                context_lines.append(f"Redundant Pairs: {red_str}")
            if corr.get("hedges"):
                hedges = corr["hedges"][:3]
                hedge_str = ", ".join([h["ticker"] for h in hedges])
                context_lines.append(f"Hedging Positions: {hedge_str}")
        
        # Breadth & Technical Signals (NEW)
        breadth = report_data.get("breadth") or {}
        if breadth:
            context_lines.append(f"Above 50d MA: {breadth.get('pct_above_50d')}")
            context_lines.append(f"Above 200d MA: {breadth.get('pct_above_200d')}")
            
            # MACD Crossovers (important actionable signals)
            macd_bull = breadth.get("macd_bull_tickers") or []
            macd_bear = breadth.get("macd_bear_tickers") or []
            if macd_bull:
                context_lines.append(f"⚡ RECENT BULLISH MACD CROSSOVERS: {', '.join(macd_bull)}")
            if macd_bear:
                context_lines.append(f"⚠️ RECENT BEARISH MACD CROSSOVERS: {', '.join(macd_bear)}")
            
            # Golden/Death crosses
            golden = breadth.get("golden_cross_tickers") or []
            death = breadth.get("death_cross_tickers") or []
            if golden:
                context_lines.append(f"Golden Crosses (bullish): {', '.join(golden)}")
            if death:
                context_lines.append(f"Death Crosses (bearish): {', '.join(death)}")
            
            # RSI extremes
            overbought = breadth.get("overbought_tickers") or []
            oversold = breadth.get("oversold_tickers") or []
            if overbought:
                context_lines.append(f"RSI Overbought (>70): {', '.join(overbought)}")
            if oversold:
                context_lines.append(f"RSI Oversold (<30): {', '.join(oversold)}")
        
        # Sector breakdown
        sectors = report_data.get("sector_breakdown") or []
        if sectors:
            top_sectors = sorted(sectors, key=lambda x: x.get("weight", 0), reverse=True)[:3]
            sector_str = ", ".join([f"{s['sector']} ({s['weight_pct']})" for s in top_sectors])
            context_lines.append(f"Top Sectors: {sector_str}")
        
        context = "\n".join(context_lines)
        
        system_prompt = """You are a senior portfolio manager sending a brief morning email to yourself or your team.

Write a concise, professional email brief (NOT a full report). Format:

SUBJECT LINE: Generate a compelling 5-8 word subject summarizing the key point

BODY:
1. Bottom Line Up Front (1-2 sentences): What's the single most important thing?

2. This Week's Numbers: 
   - Portfolio vs benchmark (outperform/underperform by how much?)
   - Key winners and losers

3. Technical Signals (IMPORTANT - include specific tickers):
   - List any MACD bullish/bearish crossovers by ticker name
   - List any Golden Cross or Death Cross signals
   - Note RSI overbought (>70) or oversold (<30) positions
   - These are actionable trading signals - be specific!

4. Action Items (bullet points, be specific):
   - What trades to consider based on technical signals
   - Positions to trim or add based on signals
   - Any positions to exit due to bearish signals

5. Risk Check:
   - Concentration concerns (top 5 weight)
   - Correlation clusters needing attention
   - Diversification score assessment

Keep it under 300 words. Be direct and actionable.
Do NOT use markdown formatting like ** or __ - use plain text only.
Start directly with the subject line, no preamble."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Portfolio data:\n{context}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating AI brief: {e}")
        return _generate_fallback_brief(report_data)


def _generate_fallback_brief(report_data: Dict[str, Any], report_url: Optional[str] = None) -> str:
    """Generate a simple brief without AI if OpenAI is unavailable."""
    lines = []
    lines.append(f"SUBJECT: Portfolio Report - {report_data.get('as_of', 'N/A')}")
    lines.append("")
    lines.append("WEEKLY SUMMARY")
    lines.append(f"Portfolio Return: {report_data.get('port_ret', 'N/A')}")
    lines.append(f"Benchmark (SPY): {report_data.get('spy_ret', 'N/A')}")
    
    if report_data.get("top"):
        lines.append("")
        lines.append("Top Contributors:")
        for x in report_data["top"][:3]:
            lines.append(f"  - {x['ticker']}: {x['contrib']}")
    
    if report_data.get("bot"):
        lines.append("")
        lines.append("Bottom Contributors:")
        for x in report_data["bot"][:3]:
            lines.append(f"  - {x['ticker']}: {x['contrib']}")
    
    # Technical Signals
    breadth = report_data.get("breadth") or {}
    if breadth:
        macd_bull = breadth.get("macd_bull_tickers") or []
        macd_bear = breadth.get("macd_bear_tickers") or []
        if macd_bull or macd_bear:
            lines.append("")
            lines.append("TECHNICAL SIGNALS:")
            if macd_bull:
                lines.append(f"  Bullish MACD Crossovers: {', '.join(macd_bull)}")
            if macd_bear:
                lines.append(f"  Bearish MACD Crossovers: {', '.join(macd_bear)}")
    
    risk = report_data.get("risk") or {}
    if risk:
        lines.append("")
        lines.append("Risk Metrics:")
        lines.append(f"  Beta: {risk.get('beta', 'N/A')}")
        lines.append(f"  Sharpe: {risk.get('sharpe', 'N/A')}")
        lines.append(f"  Max Drawdown: {risk.get('max_dd', 'N/A')}")
    
    lines.append("")
    if report_url:
        lines.append(f"View full report: {report_url}")
    else:
        lines.append("See full report for details.")
    
    return "\n".join(lines)


def send_portfolio_email(
    report_data: Dict[str, Any],
    report_path: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    from_email: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    use_tls: bool = True,
    report_base_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send portfolio brief email after report generation.
    
    Returns dict with status and any error message.
    """
    from_email = from_email or smtp_user
    
    # Build clickable report URL
    report_url = None
    if report_base_url:
        # Extract date from report_path (e.g., reports/2026-01-27/weekly.html)
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})/weekly\.html', report_path)
        if match:
            report_date = match.group(1)
            report_url = f"{report_base_url.rstrip('/')}/{report_date}/weekly.html"
    
    try:
        # Generate the AI brief
        brief = _generate_pm_brief(report_data, openai_api_key, report_url)
        
        # Parse subject from brief (first line should be SUBJECT:)
        lines = brief.split("\n")
        subject = f"Portfolio Report - {report_data.get('as_of', date.today())}"
        body = brief
        
        for i, line in enumerate(lines):
            if line.upper().startswith("SUBJECT:") or line.upper().startswith("SUBJECT LINE:"):
                subject = line.split(":", 1)[1].strip()
                body = "\n".join(lines[i+1:]).strip()
                break
        
        # Create email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Build footer with clickable link or file path
        if report_url:
            text_footer = f"View full report: {report_url}"
            html_footer = f'<a href="{report_url}" style="color: #0066cc; text-decoration: none;">📊 View Full Report</a>'
        else:
            text_footer = f"Full report saved to: {report_path}"
            html_footer = f"Full report saved to: {report_path}"
        
        # Plain text version
        text_body = f"{body}\n\n---\n{text_footer}"
        
        # HTML version (simple formatting)
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                {_text_to_html(body)}
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                <p style="font-size: 14px; margin-top: 15px;">
                    {html_footer}
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        # Send email
        context = ssl.create_default_context()
        
        if use_tls:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.sendmail(from_email, to_email, msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(from_email, to_email, msg.as_string())
        
        logger.info(f"Portfolio email sent to {to_email}")
        return {"status": "ok", "to": to_email, "subject": subject}
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {"status": "error", "error": str(e)}


def send_error_notification(
    error_message: str,
    report_date: date,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    from_email: Optional[str] = None,
    use_tls: bool = True
) -> Dict[str, Any]:
    """
    Send error notification when report generation fails.
    """
    from_email = from_email or smtp_user
    
    try:
        subject = f"⚠️ Portfolio Report FAILED - {report_date}"
        
        body = f"""Portfolio Report Generation Failed

Date: {report_date}
Error: {error_message}

Please check the logs and retry manually:
  python -m app.server.cli report {report_date} --ai-summary

---
This is an automated notification from your portfolio system.
"""
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send
        context = ssl.create_default_context()
        
        if use_tls:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_password)
                server.sendmail(from_email, to_email, msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(from_email, to_email, msg.as_string())
        
        logger.info(f"Error notification sent to {to_email}")
        return {"status": "ok", "to": to_email}
        
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")
        return {"status": "error", "error": str(e)}


def _text_to_html(text: str) -> str:
    """Convert plain text brief to simple HTML."""
    import html
    
    # Escape HTML
    text = html.escape(text)
    
    # Convert line breaks
    text = text.replace("\n\n", "</p><p>")
    text = text.replace("\n", "<br>")
    
    # Bold headings (lines ending with :)
    lines = []
    for line in text.split("<br>"):
        if line.strip().endswith(":") and len(line.strip()) < 50:
            line = f"<strong>{line}</strong>"
        lines.append(line)
    text = "<br>".join(lines)
    
    return f"<p>{text}</p>"
