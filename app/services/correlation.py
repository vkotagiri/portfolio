# app/services/correlation.py
"""
Correlation analysis for portfolio diversification insights.
Calculates correlation matrix and extracts actionable insights for AI analysis.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..models import Price

logger = logging.getLogger(__name__)


def get_returns_matrix(
    sess: Session,
    tickers: List[str],
    end_date: date,
    lookback_days: int = 60,
    min_data_points: int = 30
) -> pd.DataFrame:
    """
    Build a DataFrame of daily returns for all tickers.
    
    Only includes tickers that have sufficient historical data.
    
    Returns: DataFrame with dates as index, tickers as columns, daily returns as values
    """
    start_date = end_date - timedelta(days=lookback_days * 2)  # Buffer for trading days
    
    frames = []
    ticker_data_counts = {}
    
    for ticker in tickers:
        result = sess.query(Price.date, Price.adj_close).filter(
            Price.ticker == ticker,
            Price.date >= start_date.isoformat(),
            Price.date <= end_date.isoformat()
        ).order_by(Price.date).all()
        
        if result and len(result) >= min_data_points:
            df = pd.DataFrame(result, columns=["date", ticker])
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            frames.append(df)
            ticker_data_counts[ticker] = len(result)
        else:
            logger.debug(f"Skipping {ticker} for correlation: only {len(result) if result else 0} data points")
    
    if not frames:
        return pd.DataFrame()
    
    # Combine all price series
    prices = pd.concat(frames, axis=1, join="outer")
    
    # Fill missing values - ffill then bfill to handle edges
    prices = prices.ffill().bfill()
    
    # Drop any columns that still have NaN (shouldn't happen after bfill but safety check)
    prices = prices.dropna(axis=1)
    
    if prices.empty:
        return pd.DataFrame()
    
    # Take last N trading days
    prices = prices.tail(lookback_days + 1)
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    logger.info(f"Correlation matrix: {len(returns.columns)} tickers, {len(returns)} days of returns")
    
    return returns


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix from returns."""
    if returns.empty or len(returns) < 20:
        return pd.DataFrame()
    return returns.corr()


def extract_correlation_insights(
    corr_matrix: pd.DataFrame,
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    high_threshold: float = 0.70,
    low_threshold: float = 0.30
) -> Dict[str, Any]:
    """
    Extract actionable insights from correlation matrix.
    
    Returns dict with:
    - high_corr_pairs: Pairs with correlation > threshold
    - negative_corr_pairs: Pairs with negative correlation (hedges)
    - low_corr_pairs: Pairs with low correlation (diversifiers)
    - avg_correlation: Portfolio-weighted average correlation
    - sector_correlations: Average correlation within each sector
    - diversification_score: 0-100 score (lower avg corr = higher score)
    - concentration_clusters: Groups of highly correlated stocks
    """
    if corr_matrix.empty:
        return {
            "high_corr_pairs": [],
            "negative_corr_pairs": [],
            "low_corr_pairs": [],
            "avg_correlation": None,
            "diversification_score": None,
            "sector_correlations": {},
            "concentration_clusters": [],
            "hedging_positions": [],
            "redundant_positions": [],
        }
    
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)
    
    # Extract all pairs
    high_corr_pairs = []
    negative_corr_pairs = []
    low_corr_pairs = []
    all_correlations = []
    
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = tickers[i], tickers[j]
            corr = corr_matrix.iloc[i, j]
            
            if pd.isna(corr):
                continue
                
            all_correlations.append(corr)
            
            # Weight by position sizes
            w1 = weights.get(t1, 0)
            w2 = weights.get(t2, 0)
            combined_weight = w1 + w2
            
            pair_info = {
                "ticker1": t1,
                "ticker2": t2,
                "correlation": round(corr, 3),
                "sector1": sector_map.get(t1, "Unknown"),
                "sector2": sector_map.get(t2, "Unknown"),
                "combined_weight_pct": round(combined_weight * 100, 1),
            }
            
            if corr >= high_threshold:
                high_corr_pairs.append(pair_info)
            elif corr <= -0.1:  # Negative correlation
                negative_corr_pairs.append(pair_info)
            elif corr <= low_threshold:
                low_corr_pairs.append(pair_info)
    
    # Sort by correlation (high pairs descending, negative ascending)
    high_corr_pairs.sort(key=lambda x: x["correlation"], reverse=True)
    negative_corr_pairs.sort(key=lambda x: x["correlation"])
    low_corr_pairs.sort(key=lambda x: x["correlation"])
    
    # Calculate weighted average correlation
    avg_corr = np.mean(all_correlations) if all_correlations else None
    
    # Diversification score (0-100, higher is better)
    # avg_corr of 0 = 100, avg_corr of 1 = 0
    div_score = None
    if avg_corr is not None:
        div_score = max(0, min(100, int((1 - avg_corr) * 100)))
    
    # Sector correlations
    sector_corrs = {}
    sectors = set(sector_map.values())
    for sector in sectors:
        sector_tickers = [t for t in tickers if sector_map.get(t) == sector]
        if len(sector_tickers) >= 2:
            sector_corrs_list = []
            for i, t1 in enumerate(sector_tickers):
                for t2 in sector_tickers[i+1:]:
                    if t1 in corr_matrix.columns and t2 in corr_matrix.columns:
                        c = corr_matrix.loc[t1, t2]
                        if not pd.isna(c):
                            sector_corrs_list.append(c)
            if sector_corrs_list:
                sector_corrs[sector] = round(np.mean(sector_corrs_list), 3)
    
    # Identify concentration clusters (groups of 3+ stocks all >0.7 correlated)
    clusters = _find_correlation_clusters(corr_matrix, tickers, threshold=0.7)
    
    # Identify hedging positions (negative correlation with majority of portfolio)
    hedging_positions = _find_hedging_positions(corr_matrix, tickers, weights)
    
    # Identify redundant positions (very high correlation, consider consolidating)
    redundant = [p for p in high_corr_pairs if p["correlation"] >= 0.85][:5]
    
    return {
        "high_corr_pairs": high_corr_pairs[:10],  # Top 10
        "negative_corr_pairs": negative_corr_pairs[:5],
        "low_corr_pairs": low_corr_pairs[:5],
        "avg_correlation": round(avg_corr, 3) if avg_corr else None,
        "diversification_score": div_score,
        "sector_correlations": sector_corrs,
        "concentration_clusters": clusters,
        "hedging_positions": hedging_positions,
        "redundant_positions": redundant,
        "total_pairs_analyzed": len(all_correlations),
    }


def _find_correlation_clusters(
    corr_matrix: pd.DataFrame,
    tickers: List[str],
    threshold: float = 0.7
) -> List[Dict]:
    """Find clusters of highly correlated stocks."""
    clusters = []
    used = set()
    
    for ticker in tickers:
        if ticker in used:
            continue
        
        # Find all tickers highly correlated with this one
        cluster = [ticker]
        for other in tickers:
            if other != ticker and other not in used:
                if ticker in corr_matrix.columns and other in corr_matrix.columns:
                    corr = corr_matrix.loc[ticker, other]
                    if not pd.isna(corr) and corr >= threshold:
                        cluster.append(other)
        
        # Only keep clusters of 3+
        if len(cluster) >= 3:
            # Verify all pairs in cluster are correlated
            valid = True
            for i, t1 in enumerate(cluster):
                for t2 in cluster[i+1:]:
                    c = corr_matrix.loc[t1, t2]
                    if pd.isna(c) or c < threshold - 0.1:  # Slight tolerance
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Calculate average correlation within cluster
                corrs = []
                for i, t1 in enumerate(cluster):
                    for t2 in cluster[i+1:]:
                        corrs.append(corr_matrix.loc[t1, t2])
                
                clusters.append({
                    "tickers": cluster,
                    "size": len(cluster),
                    "avg_correlation": round(np.mean(corrs), 3),
                })
                used.update(cluster)
    
    # Sort by cluster size
    clusters.sort(key=lambda x: x["size"], reverse=True)
    return clusters[:5]  # Top 5 clusters


def _find_hedging_positions(
    corr_matrix: pd.DataFrame,
    tickers: List[str],
    weights: Dict[str, float]
) -> List[Dict]:
    """Find positions that act as hedges (low/negative correlation with rest)."""
    hedges = []
    
    for ticker in tickers:
        if ticker not in corr_matrix.columns:
            continue
        
        # Calculate weighted average correlation with rest of portfolio
        correlations = []
        total_weight = 0
        
        for other in tickers:
            if other != ticker and other in corr_matrix.columns:
                w = weights.get(other, 0)
                if w > 0:
                    corr = corr_matrix.loc[ticker, other]
                    if not pd.isna(corr):
                        correlations.append(corr * w)
                        total_weight += w
        
        if total_weight > 0:
            weighted_avg_corr = sum(correlations) / total_weight
            
            # Consider it a hedge if avg correlation < 0.3
            if weighted_avg_corr < 0.3:
                hedges.append({
                    "ticker": ticker,
                    "avg_correlation_with_portfolio": round(weighted_avg_corr, 3),
                    "weight_pct": round(weights.get(ticker, 0) * 100, 1),
                    "hedge_effectiveness": "Strong" if weighted_avg_corr < 0 else "Moderate",
                })
    
    # Sort by correlation (lowest first = best hedges)
    hedges.sort(key=lambda x: x["avg_correlation_with_portfolio"])
    return hedges[:5]


def get_correlation_analysis(
    sess: Session,
    tickers: List[str],
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    end_date: date,
    lookback_days: int = 60
) -> Dict[str, Any]:
    """
    Full correlation analysis for portfolio.
    
    Returns comprehensive insights for AI analysis.
    """
    # Get returns matrix
    returns = get_returns_matrix(sess, tickers, end_date, lookback_days)
    
    if returns.empty:
        return {"error": "Insufficient price data for correlation analysis"}
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(returns)
    
    if corr_matrix.empty:
        return {"error": "Could not calculate correlations"}
    
    # Extract insights
    insights = extract_correlation_insights(corr_matrix, weights, sector_map)
    
    # Add metadata
    insights["lookback_days"] = lookback_days
    insights["tickers_analyzed"] = len(corr_matrix.columns)
    insights["data_points"] = len(returns)
    
    return insights


def format_correlation_for_llm(insights: Dict[str, Any]) -> str:
    """Format correlation insights for LLM consumption."""
    if "error" in insights:
        return f"Correlation analysis unavailable: {insights['error']}"
    
    lines = []
    lines.append("=== CORRELATION & DIVERSIFICATION ANALYSIS ===")
    lines.append(f"Analyzed {insights.get('tickers_analyzed', 0)} positions over {insights.get('lookback_days', 60)} trading days")
    lines.append("")
    
    # Diversification score
    score = insights.get("diversification_score")
    avg_corr = insights.get("avg_correlation")
    if score is not None:
        level = "Excellent" if score >= 70 else "Good" if score >= 50 else "Moderate" if score >= 30 else "Low"
        lines.append(f"DIVERSIFICATION SCORE: {score}/100 ({level})")
        lines.append(f"Average pairwise correlation: {avg_corr}")
        lines.append("")
    
    # Concentration clusters (RISK)
    clusters = insights.get("concentration_clusters", [])
    if clusters:
        lines.append("CONCENTRATION CLUSTERS (move together in stress):")
        for c in clusters:
            lines.append(f"  - {', '.join(c['tickers'])} (avg corr: {c['avg_correlation']})")
        lines.append("")
    
    # High correlation pairs (potential redundancy)
    high_pairs = insights.get("high_corr_pairs", [])
    if high_pairs:
        lines.append("HIGHLY CORRELATED PAIRS (>0.70, consider consolidating):")
        for p in high_pairs[:7]:
            lines.append(f"  - {p['ticker1']} & {p['ticker2']}: {p['correlation']} (combined {p['combined_weight_pct']}% of portfolio)")
        lines.append("")
    
    # Redundant positions
    redundant = insights.get("redundant_positions", [])
    if redundant:
        lines.append("POTENTIALLY REDUNDANT (>0.85 correlation):")
        for p in redundant:
            lines.append(f"  - {p['ticker1']} & {p['ticker2']}: {p['correlation']}")
        lines.append("")
    
    # Hedging positions (GOOD)
    hedges = insights.get("hedging_positions", [])
    if hedges:
        lines.append("HEDGING/DIVERSIFYING POSITIONS (low correlation with portfolio):")
        for h in hedges:
            lines.append(f"  - {h['ticker']}: avg corr {h['avg_correlation_with_portfolio']} ({h['hedge_effectiveness']} hedge, {h['weight_pct']}% weight)")
        lines.append("")
    
    # Sector correlations
    sector_corrs = insights.get("sector_correlations", {})
    if sector_corrs:
        lines.append("WITHIN-SECTOR CORRELATIONS:")
        for sector, corr in sorted(sector_corrs.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {sector}: {corr}")
        lines.append("")
    
    return "\n".join(lines)
