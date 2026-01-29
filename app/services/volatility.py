# app/services/volatility.py
"""
Volatility monitoring and risk analysis service.
- Fetches VIX from FRED
- Classifies volatility regime
- Performs stress tests
- Generates risk alerts
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# VIX Regime thresholds
VIX_REGIMES = {
    "low": (0, 15),
    "normal": (15, 20),
    "elevated": (20, 25),
    "high": (25, 30),
    "crisis": (30, float("inf")),
}

VIX_REGIME_LABELS = {
    "low": ("🟢 Low", "Markets calm, consider adding risk"),
    "normal": ("🟡 Normal", "Standard volatility environment"),
    "elevated": ("🟠 Elevated", "Increased uncertainty, review hedges"),
    "high": ("🔴 High", "Significant stress, reduce risk exposure"),
    "crisis": ("🔴 Crisis", "Extreme fear, defensive positioning recommended"),
}

# Alert thresholds (can be overridden via environment)
def get_vix_alert_threshold() -> float:
    return float(os.environ.get("VIX_ALERT_THRESHOLD", 25))

def get_vol_alert_threshold() -> float:
    return float(os.environ.get("VOL_ALERT_THRESHOLD", 20))

def get_beta_alert_threshold() -> float:
    return float(os.environ.get("BETA_ALERT_THRESHOLD", 1.3))


def fetch_vix_from_fred(
    api_key: Optional[str] = None,
    lookback_days: int = 252,
    as_of: Optional[date] = None
) -> Dict[str, Any]:
    """
    Fetch VIX (VIXCLS) from FRED API.
    
    Returns dict with:
        - current: latest VIX value
        - date: date of latest reading
        - history: DataFrame of VIX history
        - percentile: current VIX percentile vs 1-year history
        - regime: classified regime (low/normal/elevated/high/crisis)
        - regime_label: human-readable regime description
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.warning("FRED_API_KEY not set, cannot fetch VIX")
        return {"error": "FRED_API_KEY not configured"}
    
    try:
        end_date = as_of or date.today()
        start_date = end_date - timedelta(days=lookback_days * 2)  # Buffer for non-trading days
        
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "VIXCLS",
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date.isoformat(),
            "observation_end": end_date.isoformat(),
            "sort_order": "asc",
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get("observations", [])
        if not observations:
            return {"error": "No VIX data returned from FRED"}
        
        # Parse into DataFrame
        records = []
        for obs in observations:
            try:
                val = float(obs["value"])
                records.append({"date": obs["date"], "vix": val})
            except (ValueError, KeyError):
                continue  # Skip "." or invalid values
        
        if not records:
            return {"error": "No valid VIX observations"}
        
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        # Get last N trading days
        df = df.tail(lookback_days)
        
        # Current value
        current_vix = float(df["vix"].iloc[-1])
        current_date = df.index[-1].date()
        
        # Calculate percentile rank
        percentile = float((df["vix"] < current_vix).mean() * 100)
        
        # Classify regime
        regime = "normal"
        for regime_name, (low, high) in VIX_REGIMES.items():
            if low <= current_vix < high:
                regime = regime_name
                break
        
        regime_label, regime_advice = VIX_REGIME_LABELS.get(regime, ("Unknown", ""))
        
        # Calculate recent stats
        vix_20d_avg = float(df["vix"].tail(20).mean())
        vix_20d_std = float(df["vix"].tail(20).std())
        vix_1y_avg = float(df["vix"].mean())
        vix_1y_high = float(df["vix"].max())
        vix_1y_low = float(df["vix"].min())
        
        # VIX trend (rising/falling/stable)
        vix_5d_ago = float(df["vix"].iloc[-6]) if len(df) > 5 else current_vix
        vix_change = current_vix - vix_5d_ago
        if vix_change > 2:
            trend = "rising"
        elif vix_change < -2:
            trend = "falling"
        else:
            trend = "stable"
        
        logger.info(f"VIX: {current_vix:.2f} ({regime_label}) - {percentile:.0f}th percentile")
        
        return {
            "current": current_vix,
            "date": current_date.isoformat(),
            "percentile": round(percentile, 1),
            "regime": regime,
            "regime_label": regime_label,
            "regime_advice": regime_advice,
            "trend": trend,
            "change_5d": round(vix_change, 2),
            "avg_20d": round(vix_20d_avg, 2),
            "std_20d": round(vix_20d_std, 2),
            "avg_1y": round(vix_1y_avg, 2),
            "high_1y": round(vix_1y_high, 2),
            "low_1y": round(vix_1y_low, 2),
            "history": df,  # For advanced analysis
        }
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch VIX from FRED: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error processing VIX data: {e}")
        return {"error": str(e)}


def stress_test_portfolio(
    holdings: List[Dict[str, Any]],
    ticker_betas: Dict[str, float],
    spy_shock_pct: float = -10.0,
) -> Dict[str, Any]:
    """
    Estimate portfolio impact if SPY moves by spy_shock_pct.
    
    Uses individual stock betas to estimate expected moves.
    
    Args:
        holdings: List of {"ticker": str, "weight": float, "beta": float (optional)}
        ticker_betas: Dict mapping ticker -> beta (from correlation/regression)
        spy_shock_pct: % move in SPY to simulate (negative for drop)
    
    Returns:
        - portfolio_impact: estimated portfolio % change
        - ticker_impacts: sorted list of individual impacts
        - vulnerable: top 5 most negatively impacted
        - hedges: positions that benefit from the shock
    """
    results = []
    total_weight = 0
    weighted_impact = 0
    
    for h in holdings:
        ticker = h.get("ticker", "")
        weight = h.get("weight", 0)
        
        if not ticker or weight <= 0:
            continue
        
        # Get beta - from provided dict, from holding, or default to 1.0
        beta = ticker_betas.get(ticker, h.get("beta", 1.0))
        
        # Special cases for known inverse/leveraged ETFs
        if ticker == "SQQQ":
            beta = -3.0  # 3x inverse QQQ
        elif ticker in ("SH", "SDS"):
            beta = -1.0 if ticker == "SH" else -2.0
        
        # Estimate impact: stock_move ≈ beta * spy_move
        stock_impact = beta * spy_shock_pct
        contribution = weight * stock_impact
        
        results.append({
            "ticker": ticker,
            "weight": round(weight * 100, 2),
            "beta": round(beta, 2),
            "estimated_move": round(stock_impact, 2),
            "contribution": round(contribution, 4),
        })
        
        total_weight += weight
        weighted_impact += contribution
    
    # Sort by contribution (worst first)
    results.sort(key=lambda x: x["contribution"])
    
    # Identify vulnerable (most negative) and hedges (positive impact)
    vulnerable = [r for r in results if r["estimated_move"] < spy_shock_pct * 0.8][:5]
    hedges = [r for r in reversed(results) if r["estimated_move"] > 0][:5]
    
    portfolio_impact = weighted_impact / total_weight if total_weight > 0 else 0
    
    return {
        "spy_shock_pct": spy_shock_pct,
        "portfolio_impact": round(portfolio_impact, 2),
        "ticker_impacts": results,
        "vulnerable": vulnerable,
        "hedges": hedges,
        "total_positions": len(results),
    }


def generate_risk_alerts(
    vix_data: Dict[str, Any],
    portfolio_vol: Optional[float] = None,
    portfolio_beta: Optional[float] = None,
    vix_threshold: Optional[float] = None,
    vol_threshold: Optional[float] = None,
    beta_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Generate risk alerts based on VIX and portfolio metrics.
    
    Thresholds are read from environment variables if not provided:
    - VIX_ALERT_THRESHOLD (default: 25)
    - VOL_ALERT_THRESHOLD (default: 20)
    - BETA_ALERT_THRESHOLD (default: 1.3)
    
    Returns list of alerts with severity (warning/critical) and message.
    """
    # Use environment thresholds if not provided
    vix_threshold = vix_threshold or get_vix_alert_threshold()
    vol_threshold = vol_threshold or get_vol_alert_threshold()
    beta_threshold = beta_threshold or get_beta_alert_threshold()
    
    alerts = []
    
    # VIX alerts
    if "current" in vix_data:
        vix = vix_data["current"]
        
        if vix >= 30:
            alerts.append({
                "type": "vix_crisis",
                "severity": "critical",
                "title": "VIX Crisis Level",
                "message": f"VIX at {vix:.1f} indicates extreme market fear. Consider defensive positioning.",
                "value": vix,
                "threshold": 30,
            })
        elif vix >= vix_threshold:
            alerts.append({
                "type": "vix_elevated",
                "severity": "warning",
                "title": "Elevated VIX",
                "message": f"VIX at {vix:.1f} is above {vix_threshold} threshold. Monitor positions closely.",
                "value": vix,
                "threshold": vix_threshold,
            })
        
        # VIX spike alert (rapid increase)
        if vix_data.get("change_5d", 0) > 5:
            alerts.append({
                "type": "vix_spike",
                "severity": "warning",
                "title": "VIX Spike",
                "message": f"VIX jumped {vix_data['change_5d']:.1f} points in 5 days. Volatility rising rapidly.",
                "value": vix_data["change_5d"],
                "threshold": 5,
            })
    
    # Portfolio volatility alerts
    if portfolio_vol is not None:
        if portfolio_vol >= vol_threshold * 1.5:
            alerts.append({
                "type": "portfolio_vol_high",
                "severity": "critical",
                "title": "Very High Portfolio Volatility",
                "message": f"Portfolio volatility at {portfolio_vol:.1f}% is significantly above target ({vol_threshold}%).",
                "value": portfolio_vol,
                "threshold": vol_threshold,
            })
        elif portfolio_vol >= vol_threshold:
            alerts.append({
                "type": "portfolio_vol_elevated",
                "severity": "warning",
                "title": "Elevated Portfolio Volatility",
                "message": f"Portfolio volatility at {portfolio_vol:.1f}% exceeds {vol_threshold}% threshold.",
                "value": portfolio_vol,
                "threshold": vol_threshold,
            })
    
    # High beta alert
    if portfolio_beta is not None and portfolio_beta > beta_threshold:
        alerts.append({
            "type": "high_beta",
            "severity": "warning",
            "title": "High Portfolio Beta",
            "message": f"Portfolio beta of {portfolio_beta:.2f} exceeds {beta_threshold:.1f} threshold. Consider reducing exposure.",
            "value": portfolio_beta,
            "threshold": beta_threshold,
        })
    
    return alerts


def build_risk_dashboard(
    holdings: List[Dict[str, Any]],
    ticker_betas: Dict[str, float],
    portfolio_vol: Optional[float] = None,
    portfolio_beta: Optional[float] = None,
    as_of: Optional[date] = None,
    fred_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build complete risk dashboard data for report.
    
    Returns:
        - vix: VIX data and regime classification
        - stress_test: SPY -10% scenario analysis
        - alerts: list of risk alerts
        - summary: human-readable summary
    """
    # Fetch VIX
    vix_data = fetch_vix_from_fred(api_key=fred_api_key, as_of=as_of)
    
    # Remove DataFrame from serializable output
    vix_display = {k: v for k, v in vix_data.items() if k != "history"}
    
    # Run stress test
    stress_test = stress_test_portfolio(holdings, ticker_betas, spy_shock_pct=-10.0)
    
    # Generate alerts
    alerts = generate_risk_alerts(
        vix_data=vix_data,
        portfolio_vol=portfolio_vol,
        portfolio_beta=portfolio_beta,
    )
    
    # Build summary
    summary_parts = []
    
    if "current" in vix_data:
        summary_parts.append(
            f"VIX at {vix_data['current']:.1f} ({vix_data.get('regime_label', 'N/A')}, "
            f"{vix_data.get('percentile', 0):.0f}th percentile)"
        )
    
    if portfolio_vol:
        summary_parts.append(f"Portfolio volatility: {portfolio_vol:.1f}%")
    
    if portfolio_beta:
        summary_parts.append(f"Portfolio beta: {portfolio_beta:.2f}")
    
    summary_parts.append(
        f"Stress test (SPY -10%): Portfolio would fall ~{abs(stress_test['portfolio_impact']):.1f}%"
    )
    
    if alerts:
        alert_summary = f"{len(alerts)} active alert(s)"
        critical = [a for a in alerts if a["severity"] == "critical"]
        if critical:
            alert_summary = f"⚠️ {len(critical)} critical alert(s)!"
        summary_parts.append(alert_summary)
    else:
        summary_parts.append("No active alerts")
    
    return {
        "vix": vix_display,
        "stress_test": stress_test,
        "alerts": alerts,
        "summary": " | ".join(summary_parts),
        "has_alerts": len(alerts) > 0,
        "has_critical_alerts": any(a["severity"] == "critical" for a in alerts),
    }


def format_risk_dashboard_for_llm(risk_data: Dict[str, Any]) -> str:
    """Format risk dashboard data for LLM analysis."""
    lines = ["## Risk Dashboard"]
    
    # VIX section
    vix = risk_data.get("vix", {})
    if "current" in vix:
        lines.append(f"\n### VIX Analysis")
        lines.append(f"- Current VIX: {vix['current']:.2f}")
        lines.append(f"- Regime: {vix.get('regime_label', 'N/A')} - {vix.get('regime_advice', '')}")
        lines.append(f"- Percentile (1Y): {vix.get('percentile', 0):.0f}th")
        lines.append(f"- 5-day change: {vix.get('change_5d', 0):+.2f}")
        lines.append(f"- 20-day avg: {vix.get('avg_20d', 0):.2f}")
        lines.append(f"- 1Y range: {vix.get('low_1y', 0):.2f} - {vix.get('high_1y', 0):.2f}")
    
    # Stress test section
    stress = risk_data.get("stress_test", {})
    if stress:
        lines.append(f"\n### Stress Test: SPY {stress.get('spy_shock_pct', -10)}% Scenario")
        lines.append(f"- Expected portfolio impact: {stress.get('portfolio_impact', 0):+.2f}%")
        
        if stress.get("vulnerable"):
            vuln = ", ".join([f"{v['ticker']} ({v['estimated_move']:+.1f}%)" for v in stress["vulnerable"][:3]])
            lines.append(f"- Most vulnerable: {vuln}")
        
        if stress.get("hedges"):
            hedges = ", ".join([f"{h['ticker']} ({h['estimated_move']:+.1f}%)" for h in stress["hedges"][:3]])
            lines.append(f"- Natural hedges: {hedges}")
    
    # Alerts section
    alerts = risk_data.get("alerts", [])
    if alerts:
        lines.append(f"\n### Active Alerts ({len(alerts)})")
        for alert in alerts:
            severity = "🔴" if alert["severity"] == "critical" else "⚠️"
            lines.append(f"- {severity} {alert['title']}: {alert['message']}")
    else:
        lines.append("\n### Alerts: None - risk levels normal")
    
    return "\n".join(lines)
