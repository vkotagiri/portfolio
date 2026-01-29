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

def get_drawdown_alert_threshold() -> float:
    """Drawdown % that triggers warning (default -10%)"""
    return float(os.environ.get("DRAWDOWN_ALERT_THRESHOLD", -10))

def get_drawdown_critical_threshold() -> float:
    """Drawdown % that triggers critical alert (default -15%)"""
    return float(os.environ.get("DRAWDOWN_CRITICAL_THRESHOLD", -15))


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


def fetch_vix_term_structure(
    api_key: Optional[str] = None,
    as_of: Optional[date] = None
) -> Dict[str, Any]:
    """
    Fetch VIX and VIX3M (3-month VIX) to analyze term structure.
    
    Term structure interpretation:
    - Contango (VIX < VIX3M): Normal, markets calm/complacent
    - Backwardation (VIX > VIX3M): Fear, hedging demand high
    - Ratio > 1.1: Strong backwardation (significant fear)
    - Ratio < 0.9: Deep contango (excessive complacency)
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        return {"error": "FRED_API_KEY not configured"}
    
    try:
        end_date = as_of or date.today()
        start_date = end_date - timedelta(days=30)  # Just need recent data
        
        results = {}
        
        # Fetch both VIX and VIX3M
        for series_id, key in [("VIXCLS", "vix"), ("VXVCLS", "vix3m")]:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": start_date.isoformat(),
                "observation_end": end_date.isoformat(),
                "sort_order": "desc",
                "limit": 5,  # Just need latest
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for obs in data.get("observations", []):
                try:
                    results[key] = float(obs["value"])
                    results[f"{key}_date"] = obs["date"]
                    break
                except (ValueError, KeyError):
                    continue
        
        if "vix" not in results or "vix3m" not in results:
            return {"error": "Could not fetch VIX term structure data"}
        
        # Calculate term structure metrics
        vix = results["vix"]
        vix3m = results["vix3m"]
        ratio = vix / vix3m if vix3m > 0 else 1.0
        spread = vix - vix3m
        
        # Classify structure
        if ratio > 1.1:
            structure = "backwardation"
            structure_label = "🔴 Backwardation"
            interpretation = "Market fear elevated - hedging demand high"
        elif ratio > 1.0:
            structure = "mild_backwardation"
            structure_label = "🟠 Mild Backwardation"
            interpretation = "Slightly elevated near-term concern"
        elif ratio > 0.9:
            structure = "contango"
            structure_label = "🟢 Contango"
            interpretation = "Normal market conditions"
        else:
            structure = "deep_contango"
            structure_label = "🟡 Deep Contango"
            interpretation = "Markets may be overly complacent"
        
        logger.info(f"VIX Term Structure: {vix:.2f}/{vix3m:.2f} = {ratio:.3f} ({structure_label})")
        
        return {
            "vix": round(vix, 2),
            "vix3m": round(vix3m, 2),
            "ratio": round(ratio, 3),
            "spread": round(spread, 2),
            "structure": structure,
            "structure_label": structure_label,
            "interpretation": interpretation,
        }
        
    except Exception as e:
        logger.error(f"Error fetching VIX term structure: {e}")
        return {"error": str(e)}


def get_volatility_targeting_recommendation(
    portfolio_vol: float,
    vix_data: Dict[str, Any],
    current_cash_pct: float = 0,
    target_vol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate volatility targeting recommendations.
    
    When portfolio vol exceeds target, recommend increasing cash/defensive positions.
    
    Returns:
        - action: 'hold', 'reduce_risk', 'add_risk'
        - target_cash_pct: suggested cash allocation
        - reasoning: explanation
    """
    target_vol = target_vol or get_vol_alert_threshold()
    
    # Get VIX regime for context
    vix_regime = vix_data.get("regime", "normal")
    vix_current = vix_data.get("current", 18)
    
    # Calculate how much to de-risk based on vol overshoot
    vol_ratio = portfolio_vol / target_vol if target_vol > 0 else 1.0
    
    result = {
        "current_vol": round(portfolio_vol, 1),
        "target_vol": round(target_vol, 1),
        "vol_ratio": round(vol_ratio, 2),
        "vix_regime": vix_regime,
    }
    
    if vol_ratio > 1.3:  # Vol significantly above target
        # Suggest reducing exposure
        suggested_reduction = min(30, (vol_ratio - 1) * 25)  # Up to 30% reduction
        target_cash = min(40, current_cash_pct + suggested_reduction)
        
        result.update({
            "action": "reduce_risk",
            "action_label": "⚠️ Reduce Exposure",
            "target_cash_pct": round(target_cash, 0),
            "reduction_pct": round(suggested_reduction, 0),
            "reasoning": f"Portfolio vol ({portfolio_vol:.1f}%) is {(vol_ratio-1)*100:.0f}% above target ({target_vol:.0f}%). "
                        f"Consider moving {suggested_reduction:.0f}% to cash/short-term bonds.",
        })
    elif vol_ratio > 1.1:  # Vol moderately above target
        suggested_reduction = min(15, (vol_ratio - 1) * 20)
        target_cash = min(30, current_cash_pct + suggested_reduction)
        
        result.update({
            "action": "monitor",
            "action_label": "🟡 Monitor Closely",
            "target_cash_pct": round(target_cash, 0),
            "reduction_pct": round(suggested_reduction, 0),
            "reasoning": f"Portfolio vol ({portfolio_vol:.1f}%) is slightly above target ({target_vol:.0f}%). "
                        f"Monitor positions and consider trimming high-beta names.",
        })
    elif vol_ratio < 0.7 and vix_regime in ("low", "normal"):  # Vol below target in calm market
        suggested_increase = min(15, (1 - vol_ratio) * 20)
        
        result.update({
            "action": "add_risk",
            "action_label": "🟢 Room to Add",
            "target_cash_pct": max(5, current_cash_pct - suggested_increase),
            "increase_pct": round(suggested_increase, 0),
            "reasoning": f"Portfolio vol ({portfolio_vol:.1f}%) is well below target ({target_vol:.0f}%) "
                        f"and VIX is {vix_regime}. Room to add risk if opportunities arise.",
        })
    else:
        result.update({
            "action": "hold",
            "action_label": "✅ On Target",
            "target_cash_pct": round(current_cash_pct, 0),
            "reasoning": f"Portfolio vol ({portfolio_vol:.1f}%) is within acceptable range of target ({target_vol:.0f}%).",
        })
    
    return result


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
    current_drawdown: Optional[float] = None,
    term_structure: Optional[Dict[str, Any]] = None,
    vix_threshold: Optional[float] = None,
    vol_threshold: Optional[float] = None,
    beta_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Generate risk alerts based on VIX, portfolio metrics, and drawdown.
    
    Thresholds are read from environment variables if not provided:
    - VIX_ALERT_THRESHOLD (default: 25)
    - VOL_ALERT_THRESHOLD (default: 20)
    - BETA_ALERT_THRESHOLD (default: 1.3)
    - DRAWDOWN_ALERT_THRESHOLD (default: -10)
    - DRAWDOWN_CRITICAL_THRESHOLD (default: -15)
    
    Returns list of alerts with severity (warning/critical) and message.
    """
    # Use environment thresholds if not provided
    vix_threshold = vix_threshold or get_vix_alert_threshold()
    vol_threshold = vol_threshold or get_vol_alert_threshold()
    beta_threshold = beta_threshold or get_beta_alert_threshold()
    dd_warning = get_drawdown_alert_threshold()
    dd_critical = get_drawdown_critical_threshold()
    
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
    
    # Drawdown alerts
    if current_drawdown is not None:
        if current_drawdown <= dd_critical:
            alerts.append({
                "type": "drawdown_critical",
                "severity": "critical",
                "title": "Severe Drawdown",
                "message": f"Portfolio is down {abs(current_drawdown):.1f}% from peak. Consider defensive action or rebalancing.",
                "value": current_drawdown,
                "threshold": dd_critical,
            })
        elif current_drawdown <= dd_warning:
            alerts.append({
                "type": "drawdown_warning",
                "severity": "warning",
                "title": "Drawdown Alert",
                "message": f"Portfolio is down {abs(current_drawdown):.1f}% from peak. Monitor closely.",
                "value": current_drawdown,
                "threshold": dd_warning,
            })
    
    # VIX term structure alert (backwardation = fear)
    if term_structure and "ratio" in term_structure:
        ratio = term_structure["ratio"]
        if ratio > 1.15:  # Strong backwardation
            alerts.append({
                "type": "vix_backwardation",
                "severity": "warning",
                "title": "VIX Backwardation",
                "message": f"VIX term structure inverted ({ratio:.2f}x). Near-term fear elevated - hedge demand high.",
                "value": ratio,
                "threshold": 1.15,
            })
    
    return alerts


def build_risk_dashboard(
    holdings: List[Dict[str, Any]],
    ticker_betas: Dict[str, float],
    portfolio_vol: Optional[float] = None,
    portfolio_beta: Optional[float] = None,
    current_drawdown: Optional[float] = None,
    current_cash_pct: float = 0,
    as_of: Optional[date] = None,
    fred_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build complete risk dashboard data for report.
    
    Returns:
        - vix: VIX data and regime classification
        - term_structure: VIX vs VIX3M analysis
        - stress_test: SPY -10% scenario analysis
        - vol_targeting: volatility targeting recommendation
        - alerts: list of risk alerts
        - summary: human-readable summary
    """
    # Fetch VIX
    vix_data = fetch_vix_from_fred(api_key=fred_api_key, as_of=as_of)
    
    # Remove DataFrame from serializable output
    vix_display = {k: v for k, v in vix_data.items() if k != "history"}
    
    # Fetch VIX term structure
    term_structure = fetch_vix_term_structure(api_key=fred_api_key, as_of=as_of)
    
    # Run stress test
    stress_test = stress_test_portfolio(holdings, ticker_betas, spy_shock_pct=-10.0)
    
    # Volatility targeting recommendation
    vol_targeting = None
    if portfolio_vol is not None:
        vol_targeting = get_volatility_targeting_recommendation(
            portfolio_vol=portfolio_vol,
            vix_data=vix_data,
            current_cash_pct=current_cash_pct,
        )
    
    # Generate alerts (including drawdown and term structure)
    alerts = generate_risk_alerts(
        vix_data=vix_data,
        portfolio_vol=portfolio_vol,
        portfolio_beta=portfolio_beta,
        current_drawdown=current_drawdown,
        term_structure=term_structure if "error" not in term_structure else None,
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
    
    result = {
        "vix": vix_display,
        "stress_test": stress_test,
        "alerts": alerts,
        "summary": " | ".join(summary_parts),
        "has_alerts": len(alerts) > 0,
        "has_critical_alerts": any(a["severity"] == "critical" for a in alerts),
    }
    
    # Add term structure if available
    if "error" not in term_structure:
        result["term_structure"] = term_structure
    
    # Add volatility targeting if available
    if vol_targeting:
        result["vol_targeting"] = vol_targeting
    
    # Add drawdown if provided
    if current_drawdown is not None:
        result["current_drawdown"] = round(current_drawdown, 2)
    
    return result


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
    
    # Term structure section
    term = risk_data.get("term_structure", {})
    if term and "ratio" in term:
        lines.append(f"\n### VIX Term Structure")
        lines.append(f"- VIX: {term['vix']:.2f} | VIX3M: {term['vix3m']:.2f}")
        lines.append(f"- Ratio: {term['ratio']:.3f} ({term.get('structure_label', 'N/A')})")
        lines.append(f"- {term.get('interpretation', '')}")
    
    # Drawdown section
    dd = risk_data.get("current_drawdown")
    if dd is not None:
        lines.append(f"\n### Drawdown Status")
        lines.append(f"- Current drawdown from peak: {dd:.1f}%")
    
    # Volatility targeting section
    vol_target = risk_data.get("vol_targeting", {})
    if vol_target:
        lines.append(f"\n### Volatility Targeting")
        lines.append(f"- Current vol: {vol_target.get('current_vol', 0):.1f}%")
        lines.append(f"- Target vol: {vol_target.get('target_vol', 0):.1f}%")
        lines.append(f"- Action: {vol_target.get('action_label', 'N/A')}")
        lines.append(f"- {vol_target.get('reasoning', '')}")
    
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
