"""
S³ Core — Multi-Leg Engine
===========================

EXECUTION MODEL
---------------
Step 1  PATTERN WINDOWS
    User picks a leg pattern (e.g. Rise-Fall) and a window count.
    The engine slides through the phase schedule and finds every
    contiguous occurrence of that pattern.

Step 2  STOCK SELECTION PER WINDOW
    For each leg in the window, get Top-N stocks by alpha (or return_pct).
    Find COMMON stocks across ALL legs in the window.
    Optionally rank by Persistence Score.

Step 3  NIFTY-THRESHOLD ENTRY (Buy Phase = next phase after the window)
    Record NIFTY close at buy-phase start (base).
    Trigger = base × (1 + entry_threshold%)
    Scan forward: first date NIFTY ≥ trigger → BUY.
    Fallback: if threshold never hit → BUY at phase start.

Step 4  NIFTY-THRESHOLD EXIT (Sell Phase = phase after buy phase)
    Record NIFTY close at sell-phase start (base).
    Trigger = base × (1 − exit_threshold%)
    Scan forward: first date NIFTY ≤ trigger → SELL.
    Fallback: if threshold never hit → SELL at sell-phase end.

Step 5  PRICE LOOKUP (Aux2 rules)
    Entry: stock close on trigger date if Aux2=1, else walk FORWARD.
    Exit : stock close on trigger date if Aux2=1, else walk BACKWARD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from core.phase_engine import quartile_label, quartile_from_fraction


# ─────────────────────────────────────────────────────────────────────────────
# NIFTY helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nifty_near(nifty: pd.DataFrame, date: pd.Timestamp, direction: str = "forward") -> Optional[float]:
    if nifty.empty:
        return None
    if direction == "forward":
        sub = nifty[nifty.index >= date]
        return float(sub["close"].iloc[0]) if not sub.empty else None
    sub = nifty[nifty.index <= date]
    return float(sub["close"].iloc[-1]) if not sub.empty else None


def _nifty_trigger(
    nifty: pd.DataFrame,
    phase_start: pd.Timestamp,
    phase_end: pd.Timestamp,
    threshold_pct: float,
    direction: str,  # "rise" or "fall"
) -> tuple[Optional[pd.Timestamp], float, float]:
    """
    Returns (trigger_date, nifty_base, nifty_at_trigger).
    trigger_date=None means threshold never hit.
    """
    base = _nifty_near(nifty, phase_start, "forward")
    if base is None:
        return None, 0.0, 0.0
    if threshold_pct == 0.0:
        return phase_start, base, base

    window = nifty[(nifty.index >= phase_start) & (nifty.index <= phase_end)]
    if window.empty:
        return None, base, base

    if direction == "rise":
        level = base * (1.0 + threshold_pct / 100.0)
        hits = window[window["close"] >= level]
    else:
        level = base * (1.0 - threshold_pct / 100.0)
        hits = window[window["close"] <= level]

    if hits.empty:
        return None, base, float(window["close"].iloc[-1])

    return hits.index[0], base, float(hits["close"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Stock price helpers (Aux2 rules)
# ─────────────────────────────────────────────────────────────────────────────

def _entry_price(stock: pd.DataFrame, trigger: pd.Timestamp, phase_end: pd.Timestamp
                 ) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    """Entry: trigger date if Aux2=1, else walk FORWARD within phase."""
    sub = stock[(stock.index >= trigger) & (stock.index <= phase_end) & (stock["aux2"] == 1)]
    if sub.empty:
        return None, None
    return float(sub["close"].iloc[0]), sub.index[0]


def _exit_price(stock: pd.DataFrame, trigger: pd.Timestamp, phase_start: pd.Timestamp
                ) -> tuple[Optional[float], Optional[pd.Timestamp]]:
    """Exit: trigger date if Aux2=1, else walk BACKWARD within phase."""
    sub = stock[(stock.index >= phase_start) & (stock.index <= trigger) & (stock["aux2"] == 1)]
    if sub.empty:
        return None, None
    return float(sub["close"].iloc[-1]), sub.index[-1]


def _window_vol(stock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Std of daily close-to-close returns over [start, end] (no Aux2 constraint)."""
    if stock is None:
        return float("nan")
    sub = stock[(stock.index >= start) & (stock.index <= end)]
    if len(sub) < 3:
        return float("nan")
    dr = sub["close"].pct_change().dropna()
    return float(dr.std()) if len(dr) > 1 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Pattern window finder
# ─────────────────────────────────────────────────────────────────────────────

def find_pattern_windows(phases: pd.DataFrame, pattern: list[str]) -> list[dict]:
    """
    Slide through phases and find every contiguous occurrence of `pattern`.
    Each window carries:
      - phase_ids, entry_dates, exit_dates  → the pattern legs
      - next_*  → buy phase (immediately after pattern)
      - nn_*    → sell phase (phase after buy phase)
    """
    if phases.empty or not pattern:
        return []

    rows = phases[["phase_id", "trade", "entry_date", "exit_date"]].reset_index(drop=True)
    n, L = len(rows), len(pattern)
    windows = []

    for start in range(n - L + 1):
        if not all(rows.iloc[start + j]["trade"] == pattern[j] for j in range(L)):
            continue

        leg_ids    = rows.iloc[start:start+L]["phase_id"].tolist()
        leg_entry  = rows.iloc[start:start+L]["entry_date"].tolist()
        leg_exit   = rows.iloc[start:start+L]["exit_date"].tolist()

        buy_idx = start + L
        if buy_idx < n:
            br = rows.iloc[buy_idx]
            next_phase_id, next_entry, next_exit, next_trade = int(br["phase_id"]), br["entry_date"], br["exit_date"], br["trade"]
        else:
            next_phase_id = next_entry = next_exit = next_trade = None

        sell_idx = start + L + 1
        if sell_idx < n:
            sr = rows.iloc[sell_idx]
            nn_phase_id, nn_entry, nn_exit, nn_trade = int(sr["phase_id"]), sr["entry_date"], sr["exit_date"], sr["trade"]
        else:
            nn_phase_id = nn_entry = nn_exit = nn_trade = None

        windows.append({
            "window_idx"   : len(windows),
            "phase_ids"    : leg_ids,
            "entry_dates"  : leg_entry,
            "exit_dates"   : leg_exit,
            "next_phase_id": next_phase_id,
            "next_entry"   : next_entry,
            "next_exit"    : next_exit,
            "next_trade"   : next_trade,
            "nn_phase_id"  : nn_phase_id,
            "nn_entry"     : nn_entry,
            "nn_exit"      : nn_exit,
            "nn_trade"     : nn_trade,
        })

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Stock frequency / common-stock selector
# ─────────────────────────────────────────────────────────────────────────────

def compute_stock_frequency(
    returns_df: pd.DataFrame,
    windows: list[dict],
    top_n: int,
    sort_by: str = "alpha",
) -> pd.DataFrame:
    """
    For each window: find top-N per leg, then count how many windows a
    stock appears in ALL legs (full_window_count) vs ANY leg (any_leg_count).
    """
    if not windows or returns_df.empty:
        return pd.DataFrame()

    n_legs = len(windows[0]["phase_ids"])
    freq: dict[str, dict] = {}

    def _init(tkr):
        if tkr not in freq:
            freq[tkr] = {"full_window_count": 0, "any_leg_count": 0,
                         **{f"leg_{i}_count": 0 for i in range(n_legs)},
                         "_alphas": [], "_returns": []}

    for win in windows:
        leg_sets = []
        for li, pid in enumerate(win["phase_ids"]):
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
            if ph.empty:
                leg_sets.append(set())
                continue
            top = ph.nlargest(top_n, sort_by)
            s = set(top["ticker"])
            leg_sets.append(s)
            for tkr in s:
                _init(tkr)
                freq[tkr][f"leg_{li}_count"] += 1
                row = top[top["ticker"] == tkr]
                if not row.empty:
                    freq[tkr]["_alphas"].append(float(row["alpha"].iloc[0]))
                    freq[tkr]["_returns"].append(float(row["return_pct"].iloc[0]))

        any_l  = set.union(*leg_sets)       if leg_sets else set()
        full_l = set.intersection(*leg_sets) if leg_sets else set()
        for tkr in any_l:
            _init(tkr)
            freq[tkr]["any_leg_count"] += 1
        for tkr in full_l:
            _init(tkr)
            freq[tkr]["full_window_count"] += 1

    rows = []
    for tkr, d in freq.items():
        rows.append({
            "ticker"            : tkr,
            "full_window_count" : d["full_window_count"],
            "any_leg_count"     : d["any_leg_count"],
            **{f"leg_{i}_count": d[f"leg_{i}_count"] for i in range(n_legs)},
            "avg_alpha"  : round(float(np.mean(d["_alphas"])),  4) if d["_alphas"]  else 0.0,
            "avg_return" : round(float(np.mean(d["_returns"])), 4) if d["_returns"] else 0.0,
        })
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    tw = len(windows)
    out["full_window_pct"] = (out["full_window_count"] / tw * 100).round(1)
    out["any_leg_pct"]     = (out["any_leg_count"]     / tw * 100).round(1)
    out = out.sort_values("full_window_count", ascending=False).reset_index(drop=True)
    out["freq_rank"] = range(1, len(out) + 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Persistence score
# ─────────────────────────────────────────────────────────────────────────────

def compute_persistence_score(
    freq_df: pd.DataFrame,
    windows: list[dict],
    returns_df: pd.DataFrame,
    sort_by: str = "alpha",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Score = 0.50 × appearance_rate + 30 × consistency + 20 × trend_norm
    """
    if freq_df.empty or not windows:
        return pd.DataFrame()

    twa: dict[str, list[float]] = {}
    for win in windows:
        for pid in win["phase_ids"]:
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
            if ph.empty:
                continue
            for _, r in ph.nlargest(top_n, sort_by).iterrows():
                tkr = r["ticker"]
                twa.setdefault(tkr, []).append(float(r.get("alpha", 0) or 0))

    rows = []
    for _, fr in freq_df.iterrows():
        tkr    = fr["ticker"]
        alphas = twa.get(tkr, [])
        if not alphas:
            continue
        ar      = float(fr["full_window_pct"])
        mean_a  = float(np.mean(alphas))
        std_a   = float(np.std(alphas)) if len(alphas) > 1 else 0.0
        eps     = 1e-6
        consist = max(0.0, min(1.0, 1.0 - std_a / (abs(mean_a) + eps)))
        trend   = float(np.polyfit(np.arange(len(alphas)), alphas, 1)[0]) if len(alphas) > 1 else 0.0
        t_norm  = 1 / (1 + np.exp(-trend * 0.5))
        score   = 0.50 * ar + 30.0 * consist + 20.0 * t_norm
        rows.append({
            "ticker"            : tkr,
            "freq_rank"         : int(fr["freq_rank"]),
            "full_window_count" : int(fr["full_window_count"]),
            "full_window_pct"   : round(ar, 1),
            "mean_alpha"        : round(mean_a, 4),
            "std_alpha"         : round(std_a, 4),
            "consistency_score" : round(consist, 4),
            "trend_score"       : round(trend, 4),
            "persistence_score" : round(score, 2),
            "n_appearances"     : len(alphas),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("persistence_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESHUFFLE FILTER  —  skip windows whose Fall leg exceeds a drawdown threshold
# ─────────────────────────────────────────────────────────────────────────────

def _nifty_leg_return(nifty: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """NIFTY % return between two dates (Aux-free)."""
    e = _nifty_near(nifty, pd.Timestamp(start), "forward")
    x = _nifty_near(nifty, pd.Timestamp(end),   "backward")
    if e and e > 0 and x is not None:
        return round((x - e) / e * 100, 4)
    return None


def _nifty_leg_max_drawdown(nifty: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    """Worst peak-to-trough drawdown (negative %) of NIFTY inside [start, end]."""
    win = nifty[(nifty.index >= pd.Timestamp(start)) & (nifty.index <= pd.Timestamp(end))]
    if win.empty:
        return None
    closes = win["close"].values
    peak = closes[0]
    mdd = 0.0
    for c in closes:
        if c > peak:
            peak = c
        dd = (c - peak) / peak * 100
        if dd < mdd:
            mdd = dd
    return round(mdd, 4)


def evaluate_reshuffle(
    window              : dict,
    pattern             : list[str],
    nifty_df            : pd.DataFrame,
    reshuffle_threshold : float,
    enabled             : bool,
    use_max_drawdown    : bool = True,
) -> dict:
    """
    Decide whether a window must be RESHUFFLED (skipped) because NIFTY fell
    more than `reshuffle_threshold`% during any Fall leg of the pattern.

    Returns dict: reshuffled(bool), reason(str), fall_drawdown_pct(float|None),
                  worst_fall_leg(int|None)
    """
    info = {"reshuffled": False, "reason": "", "fall_drawdown_pct": None, "worst_fall_leg": None}
    if not enabled or reshuffle_threshold <= 0:
        return info

    worst = None
    worst_leg = None
    for li, trade in enumerate(pattern):
        if str(trade).capitalize() != "Fall":
            continue
        s = window["entry_dates"][li]
        e = window["exit_dates"][li]
        val = (_nifty_leg_max_drawdown(nifty_df, s, e) if use_max_drawdown
               else _nifty_leg_return(nifty_df, s, e))
        if val is None:
            continue
        if worst is None or val < worst:
            worst = val
            worst_leg = li

    info["fall_drawdown_pct"] = worst
    info["worst_fall_leg"]    = worst_leg

    if worst is not None and worst <= -abs(reshuffle_threshold):
        info["reshuffled"] = True
        info["reason"] = (
            f"NIFTY fell {worst:.2f}% during Fall leg {worst_leg + 1} "
            f"(exceeds reshuffle threshold of {reshuffle_threshold:.1f}%). "
            f"Window skipped — re-scanning the next Rise→Fall pattern."
        )
    return info


# ─────────────────────────────────────────────────────────────────────────────
# THREE-LEG COMMON STOCK SELECTOR (per window)
# ─────────────────────────────────────────────────────────────────────────────

def _leg_topn(returns_df: pd.DataFrame, pid: int, top_n: int, sort_by: str):
    """Top-N tickers for one pattern leg → (metrics dict, set). Includes quartile."""
    ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
    if ph.empty:
        return {}, set()
    top = ph.nlargest(top_n, sort_by).reset_index(drop=True)
    n = len(top)
    metrics = {}
    for i, (_, r) in enumerate(top.iterrows()):
        rank = i + 1
        metrics[r["ticker"]] = {
            "rank"      : rank,
            "leg_size"  : n,
            "quartile"  : quartile_label(rank, n),
            "alpha"     : round(float(r.get("alpha", 0) or 0), 4),
            "return_pct": round(float(r.get("return_pct", 0) or 0), 4),
        }
    return metrics, set(metrics.keys())


def _entry_segment_topn(
    stock_dict : dict,
    nifty_df   : pd.DataFrame,
    seg_start  : pd.Timestamp,
    seg_end    : pd.Timestamp,
    top_n      : int,
):
    """
    Top-N tickers over the ENTRY SEGMENT — from the Fall exit / next-Rise entry
    (buy-phase start) up to the NIFTY entry-trigger date (the '% to buy' point).
    Ranked by segment alpha. Returns (metrics dict, set, nifty_seg_return).
    """
    seg_start = pd.Timestamp(seg_start)
    seg_end   = pd.Timestamp(seg_end)
    if seg_end < seg_start:
        seg_start, seg_end = seg_end, seg_start

    nbase = _nifty_near(nifty_df, seg_start, "forward")
    nend  = _nifty_near(nifty_df, seg_end,   "backward")
    nifty_seg = round((nend - nbase) / nbase * 100, 4) if (nbase and nbase > 0 and nend is not None) else 0.0

    rows = []
    for tkr, sdf in stock_dict.items():
        epx, _ = _entry_price(sdf, seg_start, seg_end)   # forward Aux2 within segment
        xpx, _ = _exit_price(sdf, seg_end, seg_start)    # backward Aux2 within segment
        if epx is None or xpx is None or epx <= 0:
            continue
        ret = round((xpx - epx) / epx * 100, 4)
        rows.append((tkr, ret, round(ret - nifty_seg, 4)))

    if not rows:
        return {}, set(), nifty_seg

    df = (pd.DataFrame(rows, columns=["ticker", "seg_return", "seg_alpha"])
            .sort_values("seg_alpha", ascending=False)
            .head(top_n)
            .reset_index(drop=True))
    n = len(df)
    metrics = {}
    for i, (_, r) in enumerate(df.iterrows()):
        rank = i + 1
        metrics[r["ticker"]] = {
            "rank"      : rank,
            "leg_size"  : n,
            "quartile"  : quartile_label(rank, n),
            "alpha"     : float(r["seg_alpha"]),
            "return_pct": float(r["seg_return"]),
        }
    return metrics, set(metrics.keys()), nifty_seg


def entry_segment_topn_df(
    window              : dict,
    nifty_df            : pd.DataFrame,
    stock_dict          : dict,
    entry_threshold_pct : float,
    top_n               : int,
) -> tuple[pd.DataFrame, dict]:
    """
    Public helper for the UI: the top-N stocks of the ENTRY SEGMENT for one
    window (Fall exit / next-Rise entry → NIFTY entry-trigger), ranked, with
    quartile. Returns (df, meta). df columns: rank, ticker, quartile,
    seg_alpha, seg_return.
    """
    if window is None or window.get("next_entry") is None:
        return pd.DataFrame(), {}

    buy_start = pd.Timestamp(window["next_entry"])
    buy_end   = pd.Timestamp(window["next_exit"])
    entry_tdate, _base, _at = _nifty_trigger(nifty_df, buy_start, buy_end, entry_threshold_pct, "rise")
    if entry_tdate is None:
        entry_tdate = buy_start

    seg_start = buy_start
    seg_end   = pd.Timestamp(entry_tdate)
    basis = "buy-start → entry-trigger"
    if seg_end <= seg_start:
        seg_end = buy_end
        basis = "buy-start → buy-phase-end (entry triggered at start)"

    seg_m, _seg_s, nifty_seg = _entry_segment_topn(stock_dict, nifty_df, seg_start, seg_end, top_n)
    rows = []
    for tkr, md in seg_m.items():
        rows.append({
            "rank"      : md["rank"],
            "ticker"    : tkr,
            "quartile"  : md.get("quartile", "—"),
            "seg_alpha" : md["alpha"],
            "seg_return": md["return_pct"],
        })
    df = (pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
          if rows else pd.DataFrame())
    meta = {
        "basis"             : basis,
        "seg_start"         : seg_start,
        "seg_end"           : seg_end,
        "entry_trigger_date": entry_tdate,
        "nifty_segment_ret" : nifty_seg,
        "n"                 : len(df),
    }
    return df, meta


def compute_window_candidates(
    window              : dict,
    returns_df          : pd.DataFrame,
    nifty_df            : pd.DataFrame,
    stock_dict          : dict,
    pattern             : list[str],
    top_n               : int,
    top_k_common        : int,
    sort_by             : str,
    entry_trigger_date  : pd.Timestamp,
    buy_start           : pd.Timestamp,
    buy_end             : pd.Timestamp,
    include_entry_segment : bool = True,
    vol_mode            : str = "off",   # "off" | "low" | "high"
    vol_pct             : int = 50,
) -> tuple[list[str], pd.DataFrame, dict]:
    """
    Build the per-window candidate list as the COMMON stocks across the legs:
        leg 1..L : the pattern phases (e.g. Rise, Fall)  → top-N by `sort_by`
        leg L+1  : the entry segment (buy-start → entry-trigger) → top-N by alpha
                   ONLY included when `include_entry_segment` is True.

    Each leg's top-N carries a quartile bucket (Q1 top-25% … Q4 below-25%), and
    every common stock gets per-leg quartile columns + an overall quartile.

    Returns (ordered_candidate_tickers, detail_df, meta)
    """
    leg_metrics = []   # list of (label, metrics_dict, set)

    # ── Pattern legs ──────────────────────────────────────────────────────────
    for li, pid in enumerate(window["phase_ids"]):
        m, s = _leg_topn(returns_df, pid, top_n, sort_by)
        label = f"Leg {li + 1} ({pattern[li] if li < len(pattern) else '?'})"
        leg_metrics.append((label, m, s))

    # ── Entry-segment leg (optional toggle) ────────────────────────────────────
    basis = "not included (toggle off)"
    nifty_seg = 0.0
    if include_entry_segment:
        seg_start = pd.Timestamp(buy_start)
        seg_end   = pd.Timestamp(entry_trigger_date)
        basis = "buy-start → entry-trigger"
        if seg_end <= seg_start:
            # Threshold hit immediately (or 0%): use the whole buy phase instead.
            seg_end = pd.Timestamp(buy_end)
            basis = "buy-start → buy-phase-end (entry triggered at start)"
        seg_m, seg_s, nifty_seg = _entry_segment_topn(stock_dict, nifty_df, seg_start, seg_end, top_n)
        leg_metrics.append(("Entry Segment", seg_m, seg_s))

    # ── Common across all NON-EMPTY legs ────────────────────────────────────────
    non_empty = [s for (_, _, s) in leg_metrics if s]
    common = set.intersection(*non_empty) if non_empty else set()

    # ── Build detail rows + rank by mean alpha across legs ──────────────────────
    rows = []
    for tkr in common:
        alphas, rets, fracs = [], [], []
        row = {"ticker": tkr}
        for (label, m, _s) in leg_metrics:
            md = m.get(tkr)
            if md:
                row[f"{label} | rank"]  = md["rank"]
                row[f"{label} | Q"]     = md.get("quartile", "—")
                row[f"{label} | alpha"] = md["alpha"]
                row[f"{label} | ret"]   = md["return_pct"]
                alphas.append(md["alpha"]); rets.append(md["return_pct"])
                lsz = md.get("leg_size", 0) or 0
                if lsz > 0:
                    fracs.append(md["rank"] / lsz)
        row["mean_alpha"]  = round(float(np.mean(alphas)), 4) if alphas else 0.0
        row["mean_return"] = round(float(np.mean(rets)),  4) if rets   else 0.0
        row["mean_rank_frac"] = round(float(np.mean(fracs)), 4) if fracs else 1.0
        row["quartile"]    = quartile_from_fraction(row["mean_rank_frac"])
        rows.append(row)

    detail_df = pd.DataFrame(rows)

    # ── Window-volatility gate (pattern-start → buy-trigger date) ──────────────
    # Measure each common candidate's daily-return volatility over the span from
    # the FIRST pattern leg's entry to the NIFTY buy-trigger date, then keep the
    # calmest (low) or wildest (high) percentile.  Acts as a gate; alpha still ranks.
    vol_basis = "off"
    if (not detail_df.empty) and vol_mode in ("low", "high"):
        try:
            v_start = pd.Timestamp(window["entry_dates"][0])
        except Exception:
            v_start = pd.Timestamp(buy_start)
        v_end = pd.Timestamp(entry_trigger_date)
        if v_end <= v_start:
            v_end = pd.Timestamp(buy_end)
        vols = {t: _window_vol(stock_dict.get(t), v_start, v_end)
                for t in detail_df["ticker"].tolist()}
        detail_df["window_vol"] = detail_df["ticker"].map(vols)
        valid = detail_df["window_vol"].dropna()
        if len(valid) >= 2:
            pct = max(1, min(99, int(vol_pct)))
            if vol_mode == "low":
                thr = float(np.percentile(valid.values, pct))
                keep = detail_df["window_vol"].notna() & (detail_df["window_vol"] <= thr)
                vol_basis = f"low ≤ p{pct} ({thr:.4f}) over {v_start:%d-%b-%Y}→{v_end:%d-%b-%Y}"
            else:  # high
                thr = float(np.percentile(valid.values, 100 - pct))
                keep = detail_df["window_vol"].notna() & (detail_df["window_vol"] >= thr)
                vol_basis = f"high ≥ p{100-pct} ({thr:.4f}) over {v_start:%d-%b-%Y}→{v_end:%d-%b-%Y}"
            if keep.any():
                detail_df = detail_df[keep].reset_index(drop=True)

    if not detail_df.empty:
        detail_df = detail_df.sort_values("mean_alpha", ascending=False).reset_index(drop=True)
        detail_df["common_rank"] = range(1, len(detail_df) + 1)

    ordered = detail_df["ticker"].tolist()[:top_k_common] if not detail_df.empty else []
    meta = {
        "entry_segment_basis"  : basis,
        "entry_segment_included": include_entry_segment,
        "leg_labels"           : [lbl for (lbl, _, _) in leg_metrics],
        "n_legs_used"          : len(non_empty),
        "nifty_segment_ret"    : nifty_seg,
        "n_common_total"       : len(common),
        "vol_mode"             : vol_mode,
        "vol_basis"            : vol_basis,
    }
    return ordered, detail_df, meta


# ─────────────────────────────────────────────────────────────────────────────
# FULL PER-WINDOW STOCK RANKING LOG
# ─────────────────────────────────────────────────────────────────────────────

def compute_trade_phase_ranks(
    windows     : list[dict],
    per_trade_df: pd.DataFrame,
    stock_dict  : dict,
    nifty_df    : pd.DataFrame,
) -> pd.DataFrame:
    """
    NEW QUARTILE SYSTEM
    ====================
    For every traded window, rank ALL eligible stocks by their actual
    return during the real trade period — i.e. from the NIFTY entry-trigger
    date (buy date) to the NIFTY exit-trigger date (sell date), using the
    same Aux2 forward/backward walk rules as the main engine.

    This replaces the old pattern-leg (Rise/Fall window) quartile with a
    post-hoc quartile: "given everything that could have been bought on
    buy-day and sold on sell-day, where did our picked stocks rank?"

    Returns DataFrame columns:
        window_idx, ticker, entry_date, exit_date,
        entry_price, exit_price,
        return_pct, nifty_return, alpha,
        rank, n_eligible, quartile, traded
    """
    if per_trade_df is None or per_trade_df.empty or not stock_dict:
        return pd.DataFrame()

    # ── Gather per-window trigger dates from per_trade_df ─────────────────
    win_meta: dict[int, dict] = {}
    for _, row in per_trade_df.iterrows():
        widx = int(row.get("window_idx", 0))
        if widx in win_meta:
            continue
        entry_t = row.get("entry_triggered_on_date")
        exit_t  = row.get("exit_triggered_on_date")
        buy_end = row.get("buy_phase_end")
        if entry_t is None or exit_t is None:
            continue
        win_meta[widx] = {
            "entry_tdate": pd.Timestamp(entry_t),
            "exit_tdate" : pd.Timestamp(exit_t),
            "buy_end"    : pd.Timestamp(buy_end) if buy_end is not None else pd.Timestamp(exit_t),
        }

    # ── Set of actually traded (window_idx, ticker) ───────────────────────
    traded_set: set[tuple] = set()
    for _, row in per_trade_df.iterrows():
        traded_set.add((int(row.get("window_idx", 0)), str(row.get("ticker", ""))))

    out_blocks = []
    for widx, meta in win_meta.items():
        entry_t = meta["entry_tdate"]
        exit_t  = meta["exit_tdate"]
        buy_end = meta["buy_end"]

        # NIFTY return over the trade period
        n_entry = _nifty_near(nifty_df, entry_t, "forward")
        n_exit  = _nifty_near(nifty_df, exit_t,  "backward")
        nifty_ret = (
            round((n_exit - n_entry) / n_entry * 100, 4)
            if (n_entry and n_entry > 0 and n_exit is not None)
            else None
        )

        stock_rows = []
        for tkr, sdf in stock_dict.items():
            # Entry: first Aux2=1 on/after the buy-trigger, within the buy phase
            epx, edate = _entry_price(sdf, entry_t, buy_end)
            if epx is None or epx <= 0:
                continue
            # Exit: last Aux2=1 on/before the sell-trigger, not before entry
            xpx, xdate = _exit_price(sdf, exit_t, edate)
            if xpx is None:
                continue
            ret   = round((xpx - epx) / epx * 100, 4)
            alpha = round(ret - nifty_ret, 4) if nifty_ret is not None else None
            stock_rows.append({
                "window_idx"  : widx,
                "ticker"      : tkr,
                "entry_date"  : edate,
                "exit_date"   : xdate,
                "entry_price" : round(epx, 4),
                "exit_price"  : round(xpx, 4),
                "return_pct"  : ret,
                "nifty_return": nifty_ret,
                "alpha"       : alpha,
                "traded"      : (widx, tkr) in traded_set,
            })

        if not stock_rows:
            continue

        # Rank descending by return_pct
        block = (pd.DataFrame(stock_rows)
                   .sort_values("return_pct", ascending=False)
                   .reset_index(drop=True))
        n = len(block)
        block["rank"]        = range(1, n + 1)
        block["n_eligible"]  = n
        block["quartile"]    = [quartile_label(r, n) for r in block["rank"]]
        out_blocks.append(block)

    return pd.concat(out_blocks, ignore_index=True) if out_blocks else pd.DataFrame()


def compute_window_stock_ranks(
    windows          : list[dict],
    returns_df       : pd.DataFrame,
    pattern          : list[str],
    sort_by          : str,
    candidates_df    : pd.DataFrame = None,
    window_status_df : pd.DataFrame = None,
) -> pd.DataFrame:
    """
    For EVERY (non-reshuffled) window, rank ALL stocks that are valid across the
    whole window — i.e. a stock must have a valid Aux2-based return in EVERY
    pattern leg. (If a stock's entry Aux2=0 in any leg it has no return there, so
    it drops out of the window entirely.)

    Each stock gets, per leg: rank, quartile, alpha, return; plus a window-level
    rank + quartile (by mean alpha across legs), and a `buying` flag marking the
    stocks actually selected to BUY for that window.
    """
    if returns_df is None or returns_df.empty or not windows:
        return pd.DataFrame()

    # Reshuffled / no-buy windows → excluded from the ranking log
    resh, nobuy = set(), set()
    if window_status_df is not None and not window_status_df.empty:
        for _, r in window_status_df.iterrows():
            w = int(r.get("window", 0))
            if bool(r.get("reshuffled")):
                resh.add(w)
            bps = r.get("buy_phase_start")
            try:
                no_bp = bps is None or pd.isna(bps)
            except Exception:
                no_bp = bps is None
            if no_bp:
                nobuy.add(w)

    # Which (window_idx, ticker) are we actually buying?
    buy = set()
    if candidates_df is not None and not candidates_df.empty:
        col = "selected_for_trade" if "selected_for_trade" in candidates_df.columns else "traded"
        for _, r in candidates_df.iterrows():
            if bool(r.get(col)):
                buy.add((int(r["window_idx"]), r["ticker"]))

    out_blocks = []
    for win in windows:
        w_disp = win["window_idx"] + 1
        if w_disp in resh or w_disp in nobuy:
            continue

        leg_full = []   # (label, {ticker: metrics}, set)
        for li, pid in enumerate(win["phase_ids"]):
            label = f"Leg {li + 1} ({pattern[li] if li < len(pattern) else '?'})"
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
            if ph.empty:
                leg_full.append((label, {}, set()))
                continue
            phs = ph.sort_values(sort_by, ascending=False).reset_index(drop=True)
            n = len(phs)
            m = {}
            for i, (_, rr) in enumerate(phs.iterrows()):
                rank = i + 1
                m[rr["ticker"]] = {
                    "rank"    : rank,
                    "leg_size": n,
                    "q"       : quartile_label(rank, n),
                    "alpha"   : round(float(rr.get("alpha", 0) or 0), 4),
                    "ret"     : round(float(rr.get("return_pct", 0) or 0), 4),
                }
            leg_full.append((label, m, set(m.keys())))

        non_empty = [s for (_, _, s) in leg_full if s]
        considered = set.intersection(*non_empty) if non_empty else set()
        if not considered:
            continue

        rows = []
        for tkr in considered:
            alphas, fracs = [], []
            row = {"window_idx": win["window_idx"], "window": w_disp, "ticker": tkr}
            for (lbl, m, _s) in leg_full:
                md = m.get(tkr)
                if md:
                    row[f"{lbl} | rank"]  = md["rank"]
                    row[f"{lbl} | Q"]     = md["q"]
                    row[f"{lbl} | alpha"] = md["alpha"]
                    row[f"{lbl} | ret"]   = md["ret"]
                    alphas.append(md["alpha"])
                    if md["leg_size"]:
                        fracs.append(md["rank"] / md["leg_size"])
            row["mean_alpha"]     = round(float(np.mean(alphas)), 4) if alphas else 0.0
            row["mean_rank_frac"] = round(float(np.mean(fracs)), 4) if fracs else 1.0
            row["window_quartile"] = quartile_from_fraction(row["mean_rank_frac"])
            row["buying"] = (win["window_idx"], tkr) in buy
            rows.append(row)

        block = pd.DataFrame(rows).sort_values("mean_alpha", ascending=False).reset_index(drop=True)
        block["window_rank"] = range(1, len(block) + 1)
        out_blocks.append(block)

    return pd.concat(out_blocks, ignore_index=True) if out_blocks else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# NIFTY-Threshold Trade Executor  ★  MAIN ENGINE ★
# ─────────────────────────────────────────────────────────────────────────────

def compute_nifty_threshold_trades(
    nifty_df            : pd.DataFrame,
    stock_dict          : dict,
    windows             : list[dict],
    freq_df             : pd.DataFrame,
    top_k_common        : int,
    entry_threshold_pct : float,
    exit_threshold_pct  : float,
    min_freq_pct        : float = 0.0,
    *,
    returns_df          : pd.DataFrame = None,
    pattern             : list[str]    = None,
    top_n               : int          = 20,
    sort_by             : str          = "alpha",
    reshuffle_enabled   : bool         = False,
    reshuffle_threshold : float        = 10.0,
    include_entry_segment : bool       = True,
    vol_mode            : str          = "off",
    vol_pct             : int          = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute trades for every window using the NIFTY-threshold model and the
    PER-WINDOW three-leg common-stock selection.

    Returns (per_trade_df, summary_df, window_status_df, candidates_df)

    • per_trade_df    — every executed trade (candidates differ per window)
    • summary_df      — per-ticker aggregated statistics of executed trades
    • window_status_df— one row per window: reshuffle status, reason, fall %,
                        #candidates, #trades, entry trigger info
    • candidates_df   — per-window three-leg common stocks with leg detail +
                        whether each was actually traded
    """
    if not windows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if returns_df is None:
        returns_df = pd.DataFrame()
    if pattern is None:
        # Reconstruct a best-effort pattern from the first window's legs.
        pattern = []

    per_trade_rows  = []
    status_rows     = []
    candidate_rows  = []

    for win in windows:
        w_disp = win["window_idx"] + 1
        pattern_start = win["entry_dates"][0]
        pattern_end   = win["exit_dates"][-1]

        # ── Reshuffle evaluation (uses pattern Fall legs) ──────────────────────
        rinfo = evaluate_reshuffle(win, pattern, nifty_df, reshuffle_threshold, reshuffle_enabled)

        if win["next_phase_id"] is None or win["next_entry"] is None:
            status_rows.append({
                "window_idx": win["window_idx"], "window": w_disp,
                "pattern_start": pattern_start, "pattern_end": pattern_end,
                "buy_phase_start": None, "buy_phase_end": None,
                "fall_drawdown_pct": rinfo["fall_drawdown_pct"],
                "reshuffled": rinfo["reshuffled"],
                "status": "No buy phase (window at end of schedule)",
                "reason": rinfo["reason"] or "No phase after the pattern to trade.",
                "n_common_candidates": 0, "n_trades": 0,
                "entry_trigger_date": None, "entry_threshold_hit": False,
            })
            continue

        buy_start  = pd.Timestamp(win["next_entry"])
        buy_end    = pd.Timestamp(win["next_exit"])
        buy_trade  = win["next_trade"]

        if win["nn_phase_id"] is not None and win["nn_entry"] is not None:
            sell_start = pd.Timestamp(win["nn_entry"])
            sell_end   = pd.Timestamp(win["nn_exit"])
            sell_trade = win["nn_trade"]
        else:
            sell_start = buy_end
            sell_end   = buy_end
            sell_trade = None

        # ── NIFTY entry trigger ───────────────────────────────────────────────
        entry_tdate, n_buy_base, n_at_entry = _nifty_trigger(
            nifty_df, buy_start, buy_end, entry_threshold_pct, "rise")
        entry_hit = entry_tdate is not None
        if entry_tdate is None:
            entry_tdate  = buy_start
            n_at_entry   = n_buy_base

        # ── If reshuffled → record reason, NO trades / NO stocks ───────────────
        if rinfo["reshuffled"]:
            status_rows.append({
                "window_idx": win["window_idx"], "window": w_disp,
                "pattern_start": pattern_start, "pattern_end": pattern_end,
                "buy_phase_start": buy_start, "buy_phase_end": buy_end,
                "fall_drawdown_pct": rinfo["fall_drawdown_pct"],
                "reshuffled": True,
                "status": "RESHUFFLED — skipped",
                "reason": rinfo["reason"],
                "n_common_candidates": 0, "n_trades": 0,
                "entry_trigger_date": entry_tdate, "entry_threshold_hit": entry_hit,
            })
            continue

        # ── Three-leg common candidate selection (per window) ──────────────────
        candidates, cand_detail, cmeta = compute_window_candidates(
            window=win, returns_df=returns_df, nifty_df=nifty_df, stock_dict=stock_dict,
            pattern=pattern, top_n=top_n, top_k_common=top_k_common, sort_by=sort_by,
            entry_trigger_date=entry_tdate, buy_start=buy_start, buy_end=buy_end,
            include_entry_segment=include_entry_segment,
            vol_mode=vol_mode, vol_pct=vol_pct,
        )

        # ── NIFTY exit trigger ────────────────────────────────────────────────
        exit_tdate, n_sell_base, n_at_exit = _nifty_trigger(
            nifty_df, sell_start, sell_end, exit_threshold_pct, "fall")
        exit_hit = exit_tdate is not None
        if exit_tdate is None:
            exit_tdate = sell_end
            n_at_exit  = n_sell_base

        # ── NIFTY trade return ────────────────────────────────────────────────
        n_entry_px = _nifty_near(nifty_df, entry_tdate, "forward") or n_buy_base
        n_exit_px  = _nifty_near(nifty_df, exit_tdate,  "backward") or n_at_exit
        nifty_trade_ret = (
            round((n_exit_px - n_entry_px) / n_entry_px * 100, 4)
            if n_entry_px and n_entry_px > 0 and n_exit_px else None
        )

        days_total = (exit_tdate - entry_tdate).days if exit_tdate >= entry_tdate else 0

        # NIFTY path within buy phase (for visualisation)
        nifty_path = nifty_df[
            (nifty_df.index >= buy_start) & (nifty_df.index <= buy_end)
        ].reset_index()

        # ── Per-stock execution ───────────────────────────────────────────────
        traded_tickers = set()
        for tkr in candidates:
            sdf = stock_dict.get(tkr)
            if sdf is None:
                continue

            epx, edate = _entry_price(sdf, entry_tdate, buy_end)
            if epx is None or edate is None or epx <= 0:
                continue

            # Exit at the n% fall trigger if Aux2=1 there; otherwise walk BACK to
            # the last date the FO filter (Aux2) was 1 — but never before we
            # actually bought. So the search window is [entry date → exit trigger].
            # This also means a stock whose Aux2 turns off mid-hold is exited on
            # its last Aux2=1 date rather than being dropped.
            xpx, xdate = _exit_price(sdf, exit_tdate, edate)

            if xpx is None:
                continue

            ret = round((xpx - epx) / epx * 100, 4)
            alpha = round(ret - nifty_trade_ret, 4) if nifty_trade_ret is not None else None
            days_held = (xdate - edate).days if edate and xdate else days_total
            traded_tickers.add(tkr)

            per_trade_rows.append({
                "window_idx"               : win["window_idx"],
                "pattern_start"            : win["entry_dates"][0],
                "pattern_end"              : win["exit_dates"][-1],
                "buy_phase"                : buy_trade,
                "buy_phase_start"          : buy_start,
                "buy_phase_end"            : buy_end,
                "sell_phase"               : sell_trade,
                "sell_phase_start"         : sell_start,
                "sell_phase_end"           : sell_end,
                # Entry NIFTY details
                "nifty_buy_base"           : round(n_buy_base, 2) if n_buy_base else None,
                "nifty_entry_trigger_level": round(n_buy_base * (1 + entry_threshold_pct/100), 2) if n_buy_base else None,
                "nifty_at_entry"           : round(n_at_entry, 2) if n_at_entry else None,
                "entry_triggered_on_date"  : entry_tdate,
                "entry_threshold_hit"      : entry_hit,
                "entry_threshold_pct"      : entry_threshold_pct,
                # Exit NIFTY details
                "nifty_sell_base"          : round(n_sell_base, 2) if n_sell_base else None,
                "nifty_exit_trigger_level" : round(n_sell_base * (1 - exit_threshold_pct/100), 2) if n_sell_base else None,
                "nifty_at_exit"            : round(n_at_exit, 2) if n_at_exit else None,
                "exit_triggered_on_date"   : exit_tdate,
                "exit_threshold_hit"       : exit_hit,
                "exit_threshold_pct"       : exit_threshold_pct,
                # Trade
                "ticker"                   : tkr,
                "entry_date"               : edate,
                "exit_date"                : xdate,
                "entry_price"              : round(epx, 4),
                "exit_price"               : round(xpx, 4),
                "return_pct"               : ret,
                "nifty_return"             : nifty_trade_ret,
                "alpha"                    : alpha,
                "days_held"                : days_held,
            })

        # ── Record per-window candidate detail (common stocks, all legs) ───────
        if not cand_detail.empty:
            for _, cr in cand_detail.iterrows():
                base = {
                    "window_idx" : win["window_idx"],
                    "window"     : w_disp,
                    "common_rank": int(cr.get("common_rank", 0)),
                    "ticker"     : cr["ticker"],
                    "quartile"   : cr.get("quartile", "—"),
                    "mean_alpha" : cr.get("mean_alpha", 0.0),
                    "mean_return": cr.get("mean_return", 0.0),
                    "selected_for_trade": cr["ticker"] in candidates[:top_k_common],
                    "traded"     : cr["ticker"] in traded_tickers,
                }
                # include each leg's rank/quartile/alpha/ret columns
                for col in cand_detail.columns:
                    if (col.endswith("| rank") or col.endswith("| Q")
                            or col.endswith("| alpha") or col.endswith("| ret")):
                        base[col] = cr.get(col)
                candidate_rows.append(base)

        # ── Window status (traded window) ──────────────────────────────────────
        n_tr = len(traded_tickers)
        status_rows.append({
            "window_idx": win["window_idx"], "window": w_disp,
            "pattern_start": pattern_start, "pattern_end": pattern_end,
            "buy_phase_start": buy_start, "buy_phase_end": buy_end,
            "fall_drawdown_pct": rinfo["fall_drawdown_pct"],
            "reshuffled": False,
            "status": "Traded" if n_tr > 0 else "No common stocks found",
            "reason": ("" if n_tr > 0 else
                       f"No stocks were common across all {cmeta['n_legs_used']} legs "
                       f"(pattern legs + entry segment) for this window."),
            "n_common_candidates": cmeta.get("n_common_total", 0),
            "n_trades": n_tr,
            "entry_trigger_date": entry_tdate, "entry_threshold_hit": entry_hit,
            "entry_segment_basis": cmeta.get("entry_segment_basis", ""),
        })

    # ── Assemble outputs ───────────────────────────────────────────────────────
    window_status_df = pd.DataFrame(status_rows)
    candidates_df    = pd.DataFrame(candidate_rows)

    if not per_trade_rows:
        return pd.DataFrame(), pd.DataFrame(), window_status_df, candidates_df

    per_trade_df = pd.DataFrame(per_trade_rows)

    # ── Summary per ticker ────────────────────────────────────────────────────
    summary_rows = []
    for tkr, grp in per_trade_df.groupby("ticker"):
        n        = len(grp)
        avg_ret  = grp["return_pct"].mean()
        avg_alp  = grp["alpha"].mean()       if grp["alpha"].notna().any() else None
        win_rate = (grp["alpha"] > 0).mean() * 100 if grp["alpha"].notna().any() else None
        avg_days = grp["days_held"].mean()   if "days_held" in grp.columns else None
        pos_rate = (grp["return_pct"] > 0).mean() * 100
        max_dd   = grp["return_pct"].min()

        ann_ret = None
        if avg_days and not pd.isna(avg_days) and avg_days > 0:
            ann_ret = round(((1 + avg_ret/100)**(365/avg_days) - 1)*100, 2)

        fr_row = freq_df[freq_df["ticker"] == tkr] if not freq_df.empty else pd.DataFrame()
        freq_rank = int(fr_row["freq_rank"].iloc[0]) if not fr_row.empty else 999
        full_pct  = float(fr_row["full_window_pct"].iloc[0]) if not fr_row.empty else 0.0

        summary_rows.append({
            "ticker"            : tkr,
            "freq_rank"         : freq_rank,
            "full_window_pct"   : full_pct,
            "n_windows_common"  : int(candidates_df[candidates_df["ticker"] == tkr]["window_idx"].nunique()) if not candidates_df.empty else 0,
            "n_trades"          : n,
            "avg_return"        : round(avg_ret, 2),
            "avg_alpha"         : round(float(avg_alp), 2) if avg_alp is not None else None,
            "win_rate"          : round(float(win_rate), 1) if win_rate is not None else None,
            "positive_ret_pct"  : round(pos_rate, 1),
            "max_drawdown"      : round(float(max_dd), 2),
            "avg_days_held"     : round(float(avg_days), 1) if avg_days is not None else None,
            "annualised_return" : ann_ret,
            "n_threshold_hits"  : int(grp["entry_threshold_hit"].sum()) if "entry_threshold_hit" in grp.columns else 0,
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("avg_alpha", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return per_trade_df, summary_df, window_status_df, candidates_df
