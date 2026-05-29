"""
S³ Core — Investment Analysis
=============================
Post-processes ``per_trade_df`` (already produced by the engine) into a
₹-denominated portfolio analysis: equity curve, summary metrics and a
year-wise breakdown.

This module is PURE computation (numpy / pandas only) and is shared by both
the Streamlit UI (app.py) and the Excel exporter so the two never disagree.

Capital model
-------------
Each *window* is treated as one rebalance step. ``window_return_pct`` is the
equal-weight average of the per-stock ``return_pct`` inside that window.

    reinvest = True   →  equity[i] = equity[i-1] * (1 + window_return_pct/100)
    reinvest = False  →  equity[i] = initial * (1 + cumsum(window_return_pct)/100)

Allocation modes only affect how the *per-stock* ₹ figures are attributed
(allocated_capital / profit_inr); the portfolio equity curve is the same:

    equal       → per_window_capital split equally across stocks in the window
    independent → every stock gets the FULL initial_capital (single-stock
                  "what-if", not a real portfolio)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Indian-number formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_inr(v) -> str:
    """Compact Indian formatting (Cr / L) — used for metric cards."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    v = float(v)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}₹{a/1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a/1e5:.2f} L"
    return f"{sign}₹{a:,.0f}"


def fmt_inr_full(v) -> str:
    """Exact value with Indian digit grouping, e.g. ₹1,00,000."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    v = float(v)
    sign = "-" if v < 0 else ""
    n = abs(int(round(v)))
    s = str(n)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}₹{grouped}"


def fmt_pct(v, decimals: int = 2, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return (f"{v:+.{decimals}f}%" if signed else f"{v:.{decimals}f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_investment_analysis(
    per_trade_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    alloc_mode: str = "equal",          # "equal" | "independent"
    reinvest: bool = False,
) -> dict | None:
    """
    Returns a dict with:
        per_trade_enriched : per_trade_df + allocated_capital, profit_inr,
                             cumulative_profit_inr, equity_inr
        window_table       : one row per window (chronological)
        equity_curve       : DataFrame[exit_date, equity_inr]
        metrics            : dict of headline numbers
        yearwise           : DataFrame (one row per calendar year + TOTAL)
    Returns None if there are fewer than 2 trade windows.
    """
    if per_trade_df is None or per_trade_df.empty:
        return None

    df = per_trade_df.copy()
    if "window_idx" not in df.columns or "return_pct" not in df.columns:
        return None

    df["return_pct"] = pd.to_numeric(df["return_pct"], errors="coerce")
    df = df.dropna(subset=["return_pct"])
    if df.empty:
        return None

    for dc in ("entry_date", "exit_date"):
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    # ── Group by window in chronological order ────────────────────────────────
    windows = sorted(df["window_idx"].dropna().unique().tolist())
    if len(windows) < 2:
        return None

    equal = (alloc_mode != "independent")

    win_rows = []
    enriched_parts = []
    running_equity = float(initial_capital)
    cum_profit = 0.0

    for w in windows:
        wdf = df[df["window_idx"] == w].copy()
        n_stocks = len(wdf)
        if n_stocks == 0:
            continue

        per_window_capital = running_equity if reinvest else float(initial_capital)

        # equal-weight average stock return drives the portfolio for this window
        window_return_pct = float(wdf["return_pct"].mean())

        # per-stock ₹ attribution
        if equal:
            per_stock_capital = per_window_capital / n_stocks
            wdf["allocated_capital"] = per_stock_capital
        else:  # independent: full capital into each stock (what-if)
            wdf["allocated_capital"] = float(initial_capital)
        wdf["profit_inr"] = wdf["allocated_capital"] * (wdf["return_pct"] / 100.0)

        total_window_profit = float(wdf["profit_inr"].sum()) if equal \
            else per_window_capital * (window_return_pct / 100.0)

        # equity step (portfolio level)
        if reinvest:
            equity_after = per_window_capital * (1 + window_return_pct / 100.0)
            running_equity = equity_after
        else:
            cum_profit += per_window_capital * (window_return_pct / 100.0)
            equity_after = float(initial_capital) + cum_profit

        # per-window dates
        exit_d = wdf["exit_date"].dropna().max() if "exit_date" in wdf.columns else pd.NaT
        entry_d = wdf["entry_date"].dropna().min() if "entry_date" in wdf.columns else pd.NaT
        avg_days = float(wdf["days_held"].mean()) if "days_held" in wdf.columns \
            and wdf["days_held"].notna().any() else np.nan

        win_rows.append({
            "window_idx"        : int(w),
            "window"            : int(w) + 1,
            "entry_date"        : entry_d,
            "exit_date"         : exit_d,
            "n_stocks"          : n_stocks,
            "per_window_capital": per_window_capital,
            "window_return_pct" : window_return_pct,
            "window_profit_inr" : per_window_capital * (window_return_pct / 100.0),
            "equity_inr"        : equity_after,
            "avg_alpha"         : float(wdf["alpha"].mean()) if "alpha" in wdf.columns else np.nan,
            "avg_days_held"     : avg_days,
        })

        wdf["equity_inr"] = equity_after
        enriched_parts.append(wdf)

    if not win_rows:
        return None

    window_table = pd.DataFrame(win_rows)
    window_table["cumulative_profit_inr"] = window_table["equity_inr"] - float(initial_capital)

    enriched = pd.concat(enriched_parts, ignore_index=True)
    # attach window-level cumulative profit onto each trade row
    cum_map = dict(zip(window_table["window_idx"], window_table["cumulative_profit_inr"]))
    enriched["cumulative_profit_inr"] = enriched["window_idx"].map(cum_map)
    if "entry_date" in enriched.columns:
        enriched = enriched.sort_values(["entry_date", "window_idx"]).reset_index(drop=True)
    else:
        enriched = enriched.sort_values("window_idx").reset_index(drop=True)

    equity_curve = window_table[["exit_date", "equity_inr"]].copy()

    metrics = _compute_metrics(df, window_table, float(initial_capital))

    yearwise = _yearwise(df, float(initial_capital), reinvest)

    return {
        "initial_capital"   : float(initial_capital),
        "alloc_mode"        : alloc_mode,
        "reinvest"          : bool(reinvest),
        "per_trade_enriched": enriched,
        "window_table"      : window_table,
        "equity_curve"      : equity_curve,
        "metrics"           : metrics,
        "yearwise"          : yearwise,
    }


def _compute_metrics(df: pd.DataFrame, window_table: pd.DataFrame, initial: float) -> dict:
    eq = window_table["equity_inr"].astype(float).values
    final_equity = float(eq[-1]) if len(eq) else initial
    total_pl = final_equity - initial
    total_pl_pct = (total_pl / initial * 100.0) if initial else np.nan

    # CAGR
    cagr = np.nan
    entry_dates = df["entry_date"].dropna() if "entry_date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    exit_dates = df["exit_date"].dropna() if "exit_date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    years = np.nan
    if len(entry_dates) and len(exit_dates):
        years = (exit_dates.max() - entry_dates.min()).days / 365.25
        if years and years > 0 and initial > 0 and final_equity > 0:
            cagr = ((final_equity / initial) ** (1.0 / years) - 1.0) * 100.0

    # Max drawdown on the equity curve
    mdd = 0.0
    if len(eq):
        peak = -np.inf
        for v in eq:
            peak = max(peak, v)
            if peak > 0:
                dd = (v - peak) / peak
                mdd = min(mdd, dd)
    mdd_pct = mdd * 100.0

    # Calmar / CAR-MDD
    calmar = (cagr / abs(mdd_pct)) if (mdd_pct != 0 and not np.isnan(cagr)) else np.nan

    # Sharpe on window returns
    wr = window_table["window_return_pct"].astype(float).values
    avg_days_win = float(np.nanmean(window_table["avg_days_held"].values)) \
        if window_table["avg_days_held"].notna().any() else np.nan
    sharpe = np.nan
    if len(wr) > 1 and np.std(wr) > 0 and avg_days_win and avg_days_win > 0 and not np.isnan(avg_days_win):
        sharpe = (np.mean(wr) / np.std(wr)) * np.sqrt(252.0 / avg_days_win)

    # Win rates
    rets = df["return_pct"].astype(float)
    win_rate = float((rets > 0).mean() * 100.0) if len(rets) else np.nan
    alpha_win_rate = np.nan
    if "alpha" in df.columns:
        a = pd.to_numeric(df["alpha"], errors="coerce").dropna()
        if len(a):
            alpha_win_rate = float((a > 0).mean() * 100.0)

    n_trades = int(len(df))
    avg_profit_trade = float(window_table["window_profit_inr"].sum() / n_trades) if n_trades else np.nan

    # Best / worst window by ₹ profit
    best_window = worst_window = None
    if not window_table.empty:
        bi = window_table["window_profit_inr"].idxmax()
        wi = window_table["window_profit_inr"].idxmin()
        br, wrow = window_table.loc[bi], window_table.loc[wi]
        best_window = {
            "profit": float(br["window_profit_inr"]),
            "window": int(br["window"]),
            "date": br["exit_date"],
        }
        worst_window = {
            "profit": float(wrow["window_profit_inr"]),
            "window": int(wrow["window"]),
            "date": wrow["exit_date"],
        }

    avg_holding = float(df["days_held"].mean()) if "days_held" in df.columns \
        and df["days_held"].notna().any() else np.nan

    return {
        "final_equity"   : final_equity,
        "total_pl"       : total_pl,
        "total_pl_pct"   : total_pl_pct,
        "cagr"           : cagr,
        "mdd_pct"        : mdd_pct,
        "calmar"         : calmar,
        "sharpe"         : sharpe,
        "win_rate"       : win_rate,
        "alpha_win_rate" : alpha_win_rate,
        "n_trades"       : n_trades,
        "avg_profit_trade": avg_profit_trade,
        "best_window"    : best_window,
        "worst_window"   : worst_window,
        "avg_holding_days": avg_holding,
        "years"          : years,
    }


def _yearwise(df: pd.DataFrame, initial: float, reinvest: bool) -> pd.DataFrame:
    """Per calendar-year breakdown (by entry_date) + a TOTAL row."""
    if "entry_date" not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"], errors="coerce")
    d = d.dropna(subset=["entry_date"])
    if d.empty:
        return pd.DataFrame()
    d["year"] = d["entry_date"].dt.year

    rows = []
    cum_equity = float(initial)
    for yr in sorted(d["year"].unique()):
        ydf = d[d["year"] == yr]
        n_windows = int(ydf["window_idx"].nunique())
        n_trades = int(len(ydf))
        # equal-weight per-window returns inside this year (chronological)
        win_returns = ydf.groupby("window_idx")["return_pct"].mean()

        if not reinvest:
            # fixed capital: each window funded with `initial`
            invested = float(initial) * n_windows
            profit = float((win_returns / 100.0 * initial).sum())
        else:
            # compounding: capital base is the running equity at year start
            invested = cum_equity
            year_growth = float((np.prod(1 + win_returns.values / 100.0) - 1.0))
            profit = float(cum_equity * year_growth)
        cum_equity += profit
        # Return% is consistently profit-on-capital-deployed for the year
        year_ret_pct = (profit / invested * 100.0) if invested else np.nan

        avg_alpha = float(ydf["alpha"].mean()) if "alpha" in ydf.columns else np.nan
        win_rate = float((ydf["return_pct"] > 0).mean() * 100.0) if len(ydf) else np.nan
        rows.append({
            "Year": int(yr),
            "Windows": n_windows,
            "Trades": n_trades,
            "Invested": invested,
            "Profit": profit,
            "Return%": year_ret_pct,
            "Avg Alpha": avg_alpha,
            "Win Rate": win_rate,
            "Cumulative Equity": cum_equity,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    _tot_inv = float(out["Invested"].sum())
    _tot_pft = float(out["Profit"].sum())
    total = {
        "Year": "TOTAL",
        "Windows": int(out["Windows"].sum()),
        "Trades": int(out["Trades"].sum()),
        "Invested": _tot_inv,
        "Profit": _tot_pft,
        "Return%": (_tot_pft / _tot_inv * 100.0) if _tot_inv else np.nan,
        "Avg Alpha": float(d["alpha"].mean()) if "alpha" in d.columns else np.nan,
        "Win Rate": float((d["return_pct"] > 0).mean() * 100.0) if len(d) else np.nan,
        "Cumulative Equity": cum_equity,
    }
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out
