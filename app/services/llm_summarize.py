# app/services/llm_summarize.py
from __future__ import annotations
from typing import List, Dict
from openai import OpenAI
from ..config import settings


def summarize_news_for_ticker(ticker: str, items: List[Dict[str, object]], max_items: int = 3) -> List[str]:
    """
    Returns up to max_items concise bullets.
    If OPENAI_API_KEY is missing, returns simple headline bullets (no LLM).
    """
    if not items:
        return []

    # Sort by relevance then recency
    items = sorted(
        items,
        key=lambda x: (float(x.get("relevance") or 0.0), str(x.get("published_at") or "")),
        reverse=True,
    )[:max_items]

    if not settings.openai_api_key:
        return [f"{str(it.get('published_at',''))[:10]} — {it.get('headline','')}" for it in items]

    client = OpenAI(api_key=settings.openai_api_key)

    lines = []
    for it in items:
        d = str(it.get("published_at", ""))[:10]
        lines.append(f"- {d} | {it.get('headline','')} | src={it.get('source','')} | url={it.get('url','')}")

    prompt = f"""You are a buy-side analyst. Summarize the most material implications for {ticker} from the headlines below in <= 18 words per bullet. Focus on revenue, margins, FCF, guidance, regulatory risk, or strategic positioning.
Headlines:
{chr(10).join(lines)}
Return 1 bullet per line. No numbering."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        bullets = [b.strip(" -•\t") for b in text.splitlines() if b.strip()]
        return bullets[:max_items]
    except Exception:
        return [f"{str(it.get('published_at',''))[:10]} — {it.get('headline','')}" for it in items]
