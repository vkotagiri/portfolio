# app/services/reporting.py
from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..db import get_session
from ..repositories.holdings import all_holdings
from .technicals import rsi14, macd_12_26_9
from .risk_free import latest_1w_tbill

# --- Optional news + earnings (soft imports so report still works if missing) ---
try:
    from .news import get_top_news_last_7d  # throttled provider(s) behind the scenes
except Exception:
    def get_top_news_last_7d(tickers: List[str]) -> Dict[str, List[dict]]:
        return {}

try:
    from .earnings import upcoming_earnings_next_14d  # throttled provider(s)
except Exception:
    def upcoming_earnings_next_14d(tickers: List[str], asof: date) -> List[Dict[str, object]]:
        return []

ROOT_DIR = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT_DIR / "templates"
REPORTS_DIR = ROOT_DIR / "reports"

# --------------------- Jinja env ---------------------
def _ensure_templates():
    TEMPLATES.mkdir(parents=True, exist_ok=True)

def _env() -> Environment:
    _ensure_templates()
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

# --------------------- DB helpers ---------------------
def _series_for(sess, ticker: str, end: date, lookback_days: int = 260) -> pd.Series:
    """Adj-close series for ticker up to `end` (inclusive)."""
    start = end - timedelta(days=lookback_days * 2)  # buffer for non-sessions
    q = """
        select date, adj_close
        from prices
        where ticker=? and date between ? and ?
        order by date
    """
    df = pd.read_sql_query(
        q, sess.bind,
        params=(ticker, start.isoformat(), end.isoformat()),
        parse_dates=["date"]
    )
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["adj_close"].astype(float)

def _vol_series_for(sess, ticker: str, end: date, lookback_days: int = 260) -> pd.Series:
    """Volume series if available; else empty."""
    start = end - timedelta(days=lookback_days * 2)
    try:
        q = """
            select date, volume
            from prices
            where ticker=? and date between ? and ?
            order by date
        """
        df = pd.read_sql_query(
            q, sess.bind,
            params=(ticker, start.isoformat(), end.isoformat()),
            parse_dates=["date"]
        )
        if df.empty or "volume" not in df.columns:
            return pd.Series(dtype=float)
        s = df.set_index("date")["volume"]
        s = pd.to_numeric(s, errors="coerce")
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)

def _spy_series(sess, end: date, lookback_days: int = 260) -> pd.Series:
    """Adj-close for SPY from whichever benchmark table exists."""
    start = end - timedelta(days=lookback_days * 2)
    for tbl in ("benchmark", "benchmarks", "benchmark_prices"):
        try:
            q = f"""
                select date, adj_close
                from {tbl}
                where symbol='SPY' and date between ? and ?
                order by date
            """
            df = pd.read_sql_query(
                q, sess.bind,
                params=(start.isoformat(), end.isoformat()),
                parse_dates=["date"]
            )
            if not df.empty:
                return df.set_index("date")["adj_close"].astype(float)
        except Exception:
            continue
    return pd.Series(dtype=float)

def _latest_price_row(sess, ticker: str, end: date) -> Dict[str, Any] | None:
    q = """
        select date, adj_close, source, asof_ts
        from prices
        where ticker=? and date<=?
        order by date desc
        limit 1
    """
    df = pd.read_sql_query(q, sess.bind, params=(ticker, end.isoformat()), parse_dates=["date"])
    if df.empty:
        return None
    r = df.iloc[0]
    return {
        "date": r["date"].date().isoformat(),
        "adj_close": float(r["adj_close"]),
        "source": r.get("source") if "source" in df.columns else None,
        "asof_ts": r.get("asof_ts") if "asof_ts" in df.columns else None,
    }

def _get_week_window(sess, end: date, s: pd.Series) -> pd.Series:
    """Last 5 trading bars for that series."""
    return s.tail(5)

# --------------------- Portfolio math helpers ---------------------
def _weights_on_day(shares_map: Dict[str, float], price_map: Dict[str, float]) -> Dict[str, float]:
    mvs = {t: shares_map[t] * price_map[t] for t in shares_map if t in price_map and price_map[t] is not None}
    total = sum(mvs.values())
    if total <= 0:
        return {}
    return {t: mv / total for t, mv in mvs.items()}

def _build_portfolio_value_series(
    sess, shares: Dict[str, float], tickers: List[str], end: date, lookback_days: int
) -> pd.Series:
    """
    Build daily portfolio market value series:
    - Outer-join across tickers to avoid losing dates due to one missing series.
    - Forward-fill up to 2 trading days to bridge minor gaps.
    - Drop days with very poor coverage (<60% of tickers present).
    """
    frames = []
    for t in tickers:
        s = _series_for(sess, t, end, lookback_days)
        if not s.empty:
            frames.append(s.rename(t))
    if not frames:
        return pd.Series(dtype=float)

    df = pd.concat(frames, axis=1, join="outer").sort_index()
    df = df.ffill(limit=2)

    n = df.shape[1]
    valid_row = df.notna().sum(axis=1) >= max(2, int(0.6 * n))
    df = df[valid_row]
    if df.empty:
        return pd.Series(dtype=float)

    for t in df.columns:
        df[t] = df[t].astype(float) * float(shares.get(t, 0.0))

    pv = df.fillna(0.0).sum(axis=1)
    pv.name = "portfolio_value"
    return pv

def _beta_alpha_ols(y: pd.Series, x: pd.Series) -> Tuple[float, float, float] | None:
    """
    Return (beta, alpha_daily, r2) for y ~ alpha + beta * x
    Tries statsmodels OLS, falls back to NumPy closed-form.
    """
    try:
        y = pd.Series(y).astype(float)
        x = pd.Series(x).astype(float)
        df = pd.concat([y, x], axis=1, join="inner").dropna()
        if len(df) < 30:
            return None

        try:
            import statsmodels.api as sm
            X = sm.add_constant(df.iloc[:, 1].values)
            Y = df.iloc[:, 0].values
            model = sm.OLS(Y, X).fit()
            beta = float(model.params[1])
            alpha = float(model.params[0])  # daily
            r2 = float(model.rsquared)
            if not np.isfinite(beta) or not np.isfinite(alpha) or not np.isfinite(r2):
                raise ValueError("non-finite regression outputs")
            return beta, alpha, r2
        except Exception:
            X = df.iloc[:, 1].values
            Y = df.iloc[:, 0].values
            Xc = X - X.mean()
            Yc = Y - Y.mean()
            varX = float(np.var(Xc))
            if varX == 0.0 or not np.isfinite(varX):
                return None
            beta = float(np.cov(Xc, Yc, ddof=0)[0, 1] / varX)
            alpha = float(Y.mean() - beta * X.mean())
            y_hat = alpha + beta * X
            ss_tot = float(((Y - Y.mean()) ** 2).sum())
            ss_res = float(((Y - y_hat) ** 2).sum())
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            return beta, alpha, r2
    except Exception:
        return None

def _max_drawdown(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return float(dd.min())

def _adv20_and_dtl(sess, t: str, end: date, shares: float, price_series: pd.Series) -> Tuple[float | None, float | None]:
    """Return (ADV20_$, DTL at 20% ADV) or (None, None) if volume not available."""
    vol = _vol_series_for(sess, t, end, 60)
    if vol.empty or price_series.empty:
        return None, None
    df = pd.concat([price_series.rename("p"), vol.rename("v")], axis=1, join="inner").dropna().tail(20)
    if df.empty:
        return None, None
    adv20_dollars = float((df["p"] * df["v"]).mean())
    if adv20_dollars <= 0:
        return None, None
    position_value = float(shares) * float(price_series.dropna().iloc[-1])
    sell_rate = 0.2 * adv20_dollars  # 20% of ADV in $
    dtl = (position_value / sell_rate) if sell_rate > 0 else None
    return adv20_dollars, (None if dtl is None else float(dtl))

# --------------------- Main report ---------------------
def build_weekly_report(
    week_end: date,
    out_html_path: str | None = None,
    fetch_outlook: bool = False,  # <— NEW: gate news/earnings work
) -> Dict[str, Any]:
    env = _env()
    tpl = env.get_template("weekly.html")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    week_end = week_end if isinstance(week_end, date) else date.fromisoformat(str(week_end))
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    with get_session() as sess:
        holds = list(all_holdings(sess))
        tickers = [h.ticker for h in holds]
        shares_map = {h.ticker: float(h.shares) for h in holds}

        # ---- latest price snapshot (for weights & MV) ----
        latest_map: Dict[str, Dict[str, Any]] = {}
        total_mv = 0.0
        for h in holds:
            lp = _latest_price_row(sess, h.ticker, week_end)
            if lp:
                latest_map[h.ticker] = lp
                total_mv += float(h.shares) * float(lp["adj_close"])
        weights_now = {}
        if total_mv > 0:
            weights_now = {
                t: (float(holds[i].shares) * float(latest_map[t]["adj_close"])) / total_mv
                for i, t in enumerate(tickers) if t in latest_map
            }

        # ---- per holding rows + technicals + weekly stats ----
        rows: List[Dict[str, Any]] = []
        breadth_up = breadth_dn = 0
        macd_recent_bull = macd_recent_bear = 0

        week_first_prices: Dict[str, float] = {}
        week_last_prices: Dict[str, float] = {}

        for h in holds:
            t = h.ticker
            s = _series_for(sess, t, week_end, 260)
            latest = latest_map.get(t)
            price = latest["adj_close"] if latest else None
            price_date = latest["date"] if latest else None
            src = latest.get("source") if latest else None
            asof = latest.get("asof_ts") if latest else None

            # weekly change
            wk_change_pct = None
            wk = _get_week_window(sess, week_end, s) if not s.empty else pd.Series(dtype=float)
            if not wk.empty and len(wk) >= 2:
                wk_change_pct = (wk.iloc[-1] / wk.iloc[0] - 1.0) * 100.0
                if wk.iloc[-1] >= wk.iloc[0]:
                    breadth_up += 1
                else:
                    breadth_dn += 1
                week_first_prices[t] = float(wk.iloc[0])
                week_last_prices[t] = float(wk.iloc[-1])

            # P&L
            total_cost = None
            upl = None
            upl_pct = None
            if h.avg_cost is not None and price is not None:
                total_cost = float(h.avg_cost) * float(h.shares)
                mv = float(price) * float(h.shares)
                upl = mv - total_cost
                if total_cost != 0:
                    upl_pct = (mv / total_cost - 1.0) * 100.0

            # technicals (RSI + MACD)
            rsi_val = rsi14(s) if not s.empty else None
            rsi_label = "Data not available"
            rsi_class = "na"
            if rsi_val is not None:
                if rsi_val >= 70:
                    rsi_label = "Overbought"; rsi_class = "overbought"
                elif rsi_val <= 30:
                    rsi_label = "Oversold"; rsi_class = "oversold"
                else:
                    rsi_label = "Neutral"; rsi_class = "neutral"

            macd = macd_12_26_9(s, recent_sessions=2) if not s.empty else {
                "direction": "Data not available",
                "last_crossover": "Data not available",
                "recent_crossover": False,
                "recent_crossover_type": "unknown",
            }
            if macd["recent_crossover"]:
                if macd["recent_crossover_type"] == "bullish":
                    macd_recent_bull += 1
                elif macd["recent_crossover_type"] == "bearish":
                    macd_recent_bear += 1

            macd_badge = None
            macd_badge_color = None
            if macd.get("recent_crossover"):
                macd_badge = f"MACD {macd.get('recent_crossover_type','').capitalize()} crossover (≤2)"
                macd_badge_color = "red" if macd.get("recent_crossover_type") == "bearish" else "green"

            rows.append({
                "ticker": t,
                "shares": float(h.shares),
                "avg_cost": None if h.avg_cost is None else float(h.avg_cost),
                "total_cost": total_cost,
                "price": price,
                "price_date": price_date,
                "weight": weights_now.get(t),
                "weight_str": f"{weights_now.get(t, 0.0)*100:.2f}%" if t in weights_now else "0.00%",
                "weekly_change_pct": None if wk_change_pct is None else round(wk_change_pct, 2),
                "market_value": None if price is None else round(float(price) * float(h.shares), 2),
                "unreal_dollar": None if upl is None else round(upl, 2),
                "unreal_pct": None if upl_pct is None else round(upl_pct, 2),
                "source": src or "Data not available",
                "asof_ts": asof or "Data not available",
                # Technicals
                "rsi14": None if rsi_val is None else round(rsi_val, 2),
                "rsi_label": rsi_label,
                "rsi_class": rsi_class,
                "macd_direction": macd["direction"],
                "macd_last_cross": macd["last_crossover"],
                "macd_crossover_recent": macd["recent_crossover"],
                "macd_crossover_type": macd["recent_crossover_type"],
                "macd_badge": macd_badge,
                "macd_badge_color": macd_badge_color,
            })

        # ---- SPY weekly return ----
        spy = _spy_series(sess, week_end, 260)
        spy_week = None
        if not spy.empty:
            wk = spy.tail(5)
            if len(wk) >= 2:
                spy_week = (wk.iloc[-1] / wk.iloc[0] - 1.0) * 100.0

        # ---- Portfolio weekly return (weighted by week-start MV) ----
        port_week = None
        ctr_rows: List[Dict[str, Any]] = []
        if week_first_prices:
            w0 = _weights_on_day(shares_map, {t: week_first_prices[t] for t in week_first_prices})
            for h in holds:
                t = h.ticker
                if t in week_first_prices and t in week_last_prices and t in w0:
                    r = (week_last_prices[t] / week_first_prices[t]) - 1.0
                    ctr = w0[t] * r
                    dollar = float(h.shares) * (week_last_prices[t] - week_first_prices[t])
                    ctr_rows.append({
                        "ticker": t,
                        "ret_pct": round(r * 100.0, 2),
                        "weight_start_pct": round(w0[t] * 100.0, 2),
                        "ctr_pct": round(ctr * 100.0, 2),
                        "dollar": round(dollar, 2),
                    })
            if ctr_rows:
                port_week = sum(c["ctr_pct"] for c in ctr_rows)  # % units

        # ---- Multi-horizon returns (portfolio vs SPY) ----
        def _horizon_ret(n_bars: int) -> Tuple[str, str]:
            pv = _build_portfolio_value_series(sess, shares_map, tickers, week_end, lookback_days=max(280, n_bars+10))
            if pv.empty or len(pv) < (n_bars + 1):
                return "Data not available", "Data not available"
            pv_w = pv.tail(n_bars + 1)
            pr = (pv_w.iloc[-1] / pv_w.iloc[0] - 1.0) * 100.0
            s = spy
            if s.empty or len(s) < (n_bars + 1):
                return f"{pr:.2f}%", "Data not available"
            sw = s.tail(n_bars + 1)
            sr = (sw.iloc[-1] / sw.iloc[0] - 1.0) * 100.0
            return f"{pr:.2f}%", f"{sr:.2f}%"

        ret_1m_p, ret_1m_s = _horizon_ret(21)
        ret_3m_p, ret_3m_s = _horizon_ret(63)
        ret_6m_p, ret_6m_s = _horizon_ret(126)
        ret_12m_p, ret_12m_s = _horizon_ret(252)

        # ---- Risk-free ----
        rf_dict = latest_1w_tbill(week_end)
        rf_val = rf_dict.get("value")  # percent (weekly) or None
        rf_date = rf_dict.get("date")
        rf_src = rf_dict.get("source")

        # ---- Risk metrics (daily) ----
        pv_60 = _build_portfolio_value_series(sess, shares_map, tickers, week_end, 80)
        pv_252 = _build_portfolio_value_series(sess, shares_map, tickers, week_end, 400)
        risk: Dict[str, Any] = {
            "sigma60": "Data not available",
            "beta60": "Data not available",
            "alpha60_annual": "Data not available",
            "r2_60": "Data not available",
            "sharpe60": "Data not available",   # <— NEW
            "te_252": "Data not available",
            "ir_252": "Data not available",
            "mdd_252": "Data not available",
            "samples_60": 0,
            "samples_252": 0,
        }

        # 60D block
        if not pv_60.empty and len(pv_60) >= 30 and not spy.empty:
            pr60 = pv_60.pct_change().dropna()
            sr60 = spy.reindex(pr60.index).pct_change().dropna()
            idx = pr60.index.intersection(sr60.index)
            pr60 = pr60.loc[idx]
            sr60 = sr60.loc[idx]
            if len(pr60) >= 30:
                sigma60 = float(pr60.std() * np.sqrt(252) * 100.0)
                risk["sigma60"] = f"{sigma60:.2f}%"
                ba = _beta_alpha_ols(pr60, sr60)
                if ba:
                    beta, alpha_daily, r2 = ba
                    risk["beta60"] = f"{beta:.3f}"
                    risk["alpha60_annual"] = f"{(alpha_daily * 252.0) * 100.0:.2f}%"
                    risk["r2_60"] = f"{r2:.3f}"
                risk["samples_60"] = int(len(pr60))

                # --- NEW: Sharpe (annualized) using 1-week T-bill -> daily RF
                if rf_val is not None and np.isfinite(pr60.std()) and pr60.std() > 0:
                    rf_week = float(rf_val) / 100.0              # weekly decimal
                    rf_daily = rf_week / 5.0                     # approx daily RF
                    ex = pr60 - rf_daily                         # daily excess returns
                    sharpe = (ex.mean() / pr60.std()) * np.sqrt(252.0)
                    if np.isfinite(sharpe):
                        risk["sharpe60"] = f"{sharpe:.2f}"
                # if RF not available, leave "Data not available"

        # 252D block
        if not pv_252.empty and len(pv_252) >= 100 and not spy.empty:
            pr252 = pv_252.pct_change().dropna()
            sr252 = spy.reindex(pr252.index).pct_change().dropna()
            idx = pr252.index.intersection(sr252.index)
            pr252 = pr252.loc[idx]
            sr252 = sr252.loc[idx]
            if len(pr252) >= 100:
                active = pr252 - sr252
                te = float(active.std() * np.sqrt(252) * 100.0)
                risk["te_252"] = f"{te:.2f}%"
                er = float((pr252.mean() - sr252.mean()) * 252.0 * 100.0)  # annualized %
                risk["ir_252"] = ("Data not available" if te == 0 else f"{er / te:.2f}")
                mdd = _max_drawdown(pv_252.tail(252))
                if mdd is not None:
                    risk["mdd_252"] = f"{mdd * 100.0:.2f}%"
                risk["samples_252"] = int(len(pr252))

        # ---- Concentration: weights, HHI, Effective N ----
        w_sorted = sorted([(t, weights_now.get(t, 0.0)) for t in tickers if t in weights_now],
                          key=lambda x: x[1], reverse=True)
        top1 = sum([w for _, w in w_sorted[:1]])
        top5 = sum([w for _, w in w_sorted[:5]])
        top10 = sum([w for _, w in w_sorted[:10]])
        hhi = sum([w ** 2 for _, w in w_sorted]) if w_sorted else 0.0
        eff_n = (1.0 / hhi) if hhi > 0 else None
        concentration = {
            "top1": f"{top1*100:.2f}%" if w_sorted else "Data not available",
            "top5": f"{top5*100:.2f}%" if w_sorted else "Data not available",
            "top10": f"{top10*100:.2f}%" if w_sorted else "Data not available",
            "largest_weight": f"{(w_sorted[0][1]*100):.2f}%" if w_sorted else "Data not available",
            "largest_ticker": (w_sorted[0][0] if w_sorted else "Data not available"),
            "hhi": f"{hhi:.4f}" if w_sorted else "Data not available",
            "effective_n": f"{eff_n:.2f}" if eff_n else "Data not available",
            "over_7pct": [t for t, w in w_sorted if w > 0.07],
        }

        # ---- Risk contributions (60D covariance) ----
        risk_contrib: List[Dict[str, Any]] = []
        if w_sorted:
            frames = []
            cols = []
            for t, _w in w_sorted:
                s = _series_for(sess, t, week_end, 80)
                if not s.empty:
                    frames.append(s.rename(t))
                    cols.append(t)
            if len(frames) >= 2:
                df = pd.concat(frames, axis=1, join="inner").dropna()
                rets = df.pct_change().dropna()
                if not rets.empty and len(rets) >= 30:
                    w_vec = np.array([weights_now.get(t, 0.0) for t in rets.columns], dtype=float)
                    Sigma = np.cov(rets.values, rowvar=False)
                    port_var = float(w_vec @ Sigma @ w_vec)
                    if port_var > 0:
                        sigma_p = np.sqrt(port_var)
                        mcr = (Sigma @ w_vec) / sigma_p
                        prc = (w_vec * (Sigma @ w_vec)) / port_var
                        order = np.argsort(-prc)
                        for idx in order[:10]:
                            risk_contrib.append({
                                "ticker": rets.columns[idx],
                                "weight_pct": f"{weights_now.get(rets.columns[idx],0.0)*100:.2f}%",
                                "mcr": f"{mcr[idx]:.4f}",
                                "prc_pct": f"{prc[idx]*100:.2f}%",
                            })

        # ---- Breadth / technical counts (SMA & breakout) ----
        above50 = above200 = 0
        golden = death = 0
        breakout = breakdown = 0
        for h in holds:
            t = h.ticker
            s = _series_for(sess, t, week_end, 260)
            if s.empty or len(s) < 50:
                continue
            sma50 = s.rolling(50).mean()
            sma200 = s.rolling(200).mean() if len(s) >= 200 else None
            price = s.iloc[-1]
            if not np.isnan(sma50.iloc[-1]) and price > sma50.iloc[-1]:
                above50 += 1
            if sma200 is not None and not np.isnan(sma200.iloc[-1]) and price > sma200.iloc[-1]:
                above200 += 1
            if sma200 is not None and len(sma200.dropna()) > 0:
                if sma50.iloc[-1] > sma200.iloc[-1]:
                    golden += 1
                elif sma50.iloc[-1] < sma200.iloc[-1]:
                    death += 1
            if len(s) >= 20:
                hh = s.tail(20).max()
                ll = s.tail(20).min()
                if price >= hh:
                    breakout += 1
                if price <= ll:
                    breakdown += 1

        breadth_block = {
            "pct_above_50d": f"{(above50/len(holds))*100:.0f}%" if holds else "Data not available",
            "pct_above_200d": f"{(above200/len(holds))*100:.0f}%" if holds else "Data not available",
            "golden_cross": golden,
            "death_cross": death,
            "breakouts20": breakout,
            "breakdowns20": breakdown,
            "macd_recent_bull": macd_recent_bull,
            "macd_recent_bear": macd_recent_bear,
        }

        # ---- Liquidity (ADV20, DTL) ----
        adv_values = []
        low_adv_count = 0  # ADV < $5M
        worst_dtl = ("Data not available", "Data not available")
        worst_dtl_val = -1.0
        for h in holds:
            t = h.ticker
            s = _series_for(sess, t, week_end, 60)
            adv20, dtl = _adv20_and_dtl(sess, t, week_end, float(h.shares), s)
            if adv20 is None:
                continue
            adv_values.append(adv20)
            if adv20 < 5_000_000:
                low_adv_count += 1
            if dtl is not None and dtl > worst_dtl_val:
                worst_dtl = (t, f"{dtl:.1f}")
                worst_dtl_val = dtl
        if adv_values and len(holds) > 0:
            liq = {
                "weighted_adv20": f"${(sum(adv_values)/len(adv_values)):.0f}",
                "pct_low_adv": f"{(low_adv_count/len(holds))*100:.0f}%",
                "worst_dtl": f"{worst_dtl[0]}: {worst_dtl[1]} days" if worst_dtl_val >= 0 else "Data not available",
            }
        else:
            liq = {
                "weighted_adv20": "Data not available",
                "pct_low_adv": "Data not available",
                "worst_dtl": "Data not available",
            }

        # ---- (Optional) News & Upcoming Earnings ----
        if fetch_outlook:
            try:
                news_by_t = get_top_news_last_7d(tickers)
            except Exception:
                news_by_t = {}
            for row in rows:
                row["news"] = news_by_t.get(row["ticker"], [])

            try:
                upcoming_earn = upcoming_earnings_next_14d(tickers, week_end)
            except Exception:
                upcoming_earn = []
        else:
            for row in rows:
                row["news"] = []
            upcoming_earn = []

    # --------- Render ----------
    out_dir = REPORTS_DIR / week_end.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_html_path or str(out_dir / "weekly.html")

    summary = {
        "week_end": week_end.isoformat(),
        "generated_ts": timestamp,
        "portfolio_weekly_return": "Data not available" if port_week is None else f"{float(port_week):.2f}%",
        "spy_weekly_return": "Data not available" if spy_week is None else f"{spy_week:.2f}%",
        "risk_free": "Data not available" if rf_val is None else f"{rf_val:.2f}% (as of {rf_date}, source: {rf_src})",
        "breadth": f"{breadth_up} up / {breadth_dn} down",
        "ret_1m_p": ret_1m_p, "ret_1m_s": ret_1m_s,
        "ret_3m_p": ret_3m_p, "ret_3m_s": ret_3m_s,
        "ret_6m_p": ret_6m_p, "ret_6m_s": ret_6m_s,
        "ret_12m_p": ret_12m_p, "ret_12m_s": ret_12m_s,
        "upcoming_earnings": upcoming_earn,  # shown by template only if present
    }

    ctr_sorted = sorted(ctr_rows, key=lambda r: r["ctr_pct"])
    top5 = list(reversed(ctr_sorted[-5:])) if ctr_sorted else []
    bot5 = ctr_sorted[:5] if ctr_sorted else []

    html = tpl.render(
        title="Weekly Deep Dive Report",
        summary=summary,
        holdings=rows,
        risk=risk,
        concentration=concentration,
        breadth_block=breadth_block,
        liquidity=liq,
        top_contrib=top5,
        bottom_contrib=bot5,
    )

    Path(out_html).write_text(html, encoding="utf-8")
    return {"status": "ok", "path": out_html}
