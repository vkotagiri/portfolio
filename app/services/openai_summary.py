# app/services/openai_summary.py
from __future__ import annotations

import html
from datetime import date
from typing import Any, Dict, List, Tuple

from ..config import settings

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from .earnings import upcoming_earnings_next_14d
except Exception:
    def upcoming_earnings_next_14d(tickers, asof: date):
        return []

_SYSTEM = (
    "You are a seasoned sell-side equity analyst. "
    "Return clean, professional HTML only (no markdown), suitable for embedding in a report."
)

def _earnings_lines(tickers: List[str], asof: date) -> Tuple[List[str], bool]:
    items = []
    expanded = False
    try:
        rows = upcoming_earnings_next_14d(tickers, asof)
        for r in rows:
            tag = r.get("window", "14d")
            tag_str = "" if tag == "14d" else " (within next 21d)"
            items.append(f"{r['ticker']} — {r['date']} ({r['when']}){tag_str}")
            if tag != "14d":
                expanded = True
    except Exception:
        pass
    return items, expanded

def _fallback_card(summary: Dict[str, Any],
                   top_contrib: List[Dict[str, Any]],
                   bottom_contrib: List[Dict[str, Any]],
                   risk: Dict[str, Any],
                   breadth: Dict[str, Any],
                   concentration: Dict[str, Any],
                   tickers: List[str],
                   week_end: date) -> str:
    elines, expanded = _earnings_lines(tickers, week_end)
    def _li(s: str) -> str: return f"<li>{html.escape(s)}</li>"
    earnings_ul = "".join(_li(x) for x in elines) if elines else "<li>None within next 14 days.</li>"
    note = "" if elines or not expanded else "<li><i>Note:</i> 14-day window empty; showing first items within 21 days.</li>"
    return f"""
<section class="section">
  <h2>AI Summary</h2>
  <div class="ai-summary">
    <h3>Performance</h3>
    <ul>
      <li>Week ending {html.escape(summary.get('week_end',''))}: Portfolio {html.escape(str(summary.get('portfolio_weekly_return','N/A')))} vs SPY {html.escape(str(summary.get('spy_weekly_return','N/A')))}.</li>
    </ul>
    <h3>Attribution</h3>
    <ul>
      <li><b>Top</b>: {", ".join(f"{c['ticker']} ({c['ret_pct']}%, CTR {c['ctr_pct']}%)" for c in top_contrib) if top_contrib else "Data not available"}.</li>
      <li><b>Bottom</b>: {", ".join(f"{c['ticker']} ({c['ret_pct']}%, CTR {c['ctr_pct']}%)" for c in bottom_contrib) if bottom_contrib else "Data not available"}.</li>
    </ul>
    <h3>Risk</h3>
    <ul>
      <li>Vol (60D, ann.): {html.escape(str(risk.get('sigma60','N/A')))}; Beta (60D): {html.escape(str(risk.get('beta60','N/A')))}; Alpha (60D, ann.): {html.escape(str(risk.get('alpha60_annual','N/A')))}; R²: {html.escape(str(risk.get('r2_60','N/A')))}; Sharpe (60D): {html.escape(str(risk.get('sharpe60','N/A')))}.</li>
      <li>TE (252D): {html.escape(str(risk.get('te_252','N/A')))}; IR (252D): {html.escape(str(risk.get('ir_252','N/A')))}; MDD (252D): {html.escape(str(risk.get('mdd_252','N/A')))}.</li>
    </ul>
    <h3>Technicals & Breadth</h3>
    <ul>
      <li>{html.escape(str(breadth.get('pct_above_50d','N/A')))} above 50D; {html.escape(str(breadth.get('pct_above_200d','N/A')))} above 200D; MACD ≤2 sessions Bullish/Bearish: {breadth.get('macd_recent_bull',0)}/{breadth.get('macd_recent_bear',0)}.</li>
    </ul>
    <h3>Concentration</h3>
    <ul>
      <li>Largest: {html.escape(str(concentration.get('largest_ticker','-')))} @ {html.escape(str(concentration.get('largest_weight','N/A')))}. Top-5: {html.escape(str(concentration.get('top5','N/A')))}.</li>
    </ul>
    <h3>Upcoming Earnings</h3>
    <ul>
      {earnings_ul}
      {note}
    </ul>
  </div>
</section>
""".strip()

def summarize_weekly_to_html(*,
                             week_end: date,
                             tickers: List[str],
                             summary: Dict[str, Any],
                             top_contrib: List[Dict[str, Any]],
                             bottom_contrib: List[Dict[str, Any]],
                             risk: Dict[str, Any],
                             breadth: Dict[str, Any],
                             concentration: Dict[str, Any]) -> str:
    """Returns a professional HTML card. Ensures concrete earnings are included."""
    elines, expanded = _earnings_lines(tickers, week_end)

    if OpenAI is None or not getattr(settings, "openai_api_key", None):
        return _fallback_card(summary, top_contrib, bottom_contrib, risk, breadth, concentration, tickers, week_end)

    client = OpenAI(api_key=settings.openai_api_key)
    must_include = "\n".join(f"- {e}" for e in elines) if elines else "(No items in 14d; if empty, say so.)"

    user_prompt = f"""
Return a single HTML fragment for a concise analyst card (≤180 words) with sections:
<h3>Performance</h3>, <h3>Attribution</h3>, <h3>Risk</h3>, <h3>Technicals & Breadth</h3>, <h3>Concentration</h3>, <h3>Upcoming Earnings</h3>.
Use short bullet lists (<ul><li>…</li></ul>). Be factual; do not invent numbers.

Data (JSON-like):
summary={summary}
top={top_contrib}
bottom={bottom_contrib}
risk={risk}
breadth={breadth}
concentration={concentration}

MUST include this exact 'Upcoming Earnings' list verbatim as <li> items (one per line), unless it's empty:
{must_include}

If the 14-day window was empty but a 21-day expansion was used, add a short note: "14-day window empty; showing first items within 21 days."
"""

    try:
        resp = client.chat.completions.create(
            model=getattr(settings, "openai_model", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=500,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        html_out = (resp.choices[0].message.content or "").strip()
        if not html_out or html_out.startswith("- ") or html_out.startswith("* "):
            # Guard against markdown slips
            return _fallback_card(summary, top_contrib, bottom_contrib, risk, breadth, concentration, tickers, week_end)
        return f'<section class="section"><h2>AI Summary</h2><div class="ai-summary">{html_out}</div></section>'
    except Exception:
        return _fallback_card(summary, top_contrib, bottom_contrib, risk, breadth, concentration, tickers, week_end)
