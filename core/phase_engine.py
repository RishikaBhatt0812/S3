"""
S³ Core — Phase Return Engine
==============================
Computes per-phase returns for every stock using Aux2 rules:
  Entry: close on entry_date if Aux2=1, else skip trade.
  Exit : close on exit_date  if Aux2=1, else walk backward to last Aux2=1 date.

Vectorised: no per-row Python loops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _nifty_return_per_phase(nifty_df: pd.DataFrame, phases: pd.DataFrame) -> dict[int, float]:
    """Compute NIFTY % return for each phase (no Aux2 constraint)."""
    out = {}
    if nifty_df.empty:
        return out
    for _, ph in phases.iterrows():
        pid = int(ph["phase_id"])
        entry_rows = nifty_df[nifty_df.index >= ph["entry_date"]]
        exit_rows  = nifty_df[nifty_df.index <= ph["exit_date"]]
        if entry_rows.empty or exit_rows.empty:
            continue
        ec = float(entry_rows["close"].iloc[0])
        xc = float(exit_rows["close"].iloc[-1])
        if ec > 0:
            out[pid] = round((xc - ec) / ec * 100, 4)
    return out


@st.cache_data(show_spinner=False)
def compute_all_phase_returns(
    _stock_dict: dict,
    _phases: pd.DataFrame,
    _nifty: pd.DataFrame,
    excluded_tickers: tuple = (),
) -> pd.DataFrame:
    """
    Compute (phase × stock) return table.

    Returns DataFrame: phase_id, trade, entry_date, exit_date,
                       ticker, entry_close, exit_close,
                       return_pct, nifty_return, alpha, year, month
    """
    if _phases.empty or not _stock_dict:
        return pd.DataFrame()

    excluded = set(excluded_tickers)
    nifty_ret_map = _nifty_return_per_phase(_nifty, _phases)

    # ── Stack all Aux2=1 rows ─────────────────────────────────────────────────
    parts = []
    for tkr, df in _stock_dict.items():
        if tkr in excluded:
            continue
        _tmp = df[df["aux2"] == 1].reset_index()
        # index name may be "date" after reset_index
        if "date" not in _tmp.columns:
            _tmp = _tmp.rename(columns={_tmp.columns[0]: "date"})
        aux1 = _tmp[["date", "close"]]
        aux1["ticker"] = tkr
        parts.append(aux1)
    if not parts:
        return pd.DataFrame()

    stacked = pd.concat(parts, ignore_index=True)
    stacked["date"] = pd.to_datetime(stacked["date"])
    stacked = stacked.sort_values("date")

    phases = _phases.copy()
    phases["entry_date"] = pd.to_datetime(phases["entry_date"])
    phases["exit_date"]  = pd.to_datetime(phases["exit_date"])

    # ── Aux2 (FO-filter) rule, per (phase, ticker) ───────────────────────────
    #   ENTRY = close on FIRST Aux2=1 date on/after entry_date (within phase).
    #           → if entry_date itself has Aux2=1, that close is used.
    #   EXIT  = close on LAST  Aux2=1 date on/before exit_date (within phase).
    #           → if exit_date has Aux2=1, that close is used; otherwise we walk
    #             back to the last date the FO filter (Aux2) was 1.
    rows = []
    for _, ph in phases.iterrows():
        pid    = int(ph["phase_id"])
        e_date = ph["entry_date"]
        x_date = ph["exit_date"]
        win = stacked[(stacked["date"] >= e_date) & (stacked["date"] <= x_date)]
        if win.empty:
            continue
        g = win.groupby("ticker", sort=False)
        agg = g.agg(
            entry_close      =("close", "first"),
            entry_close_date =("date",  "first"),
            exit_close       =("close", "last"),
            exit_close_date  =("date",  "last"),
        ).reset_index()
        agg["phase_id"]   = pid
        agg["trade"]      = ph["trade"]
        agg["entry_date"] = e_date
        agg["exit_date"]  = x_date
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=["entry_close", "exit_close"])
    result = result[result["entry_close"] > 0]
    if result.empty:
        return pd.DataFrame()

    result["return_pct"]   = ((result["exit_close"] - result["entry_close"]) / result["entry_close"] * 100).round(4)
    result["nifty_return"] = result["phase_id"].map(nifty_ret_map)
    result["nifty_return"] = pd.to_numeric(result["nifty_return"], errors="coerce").round(4)
    result["alpha"]        = (result["return_pct"] - result["nifty_return"]).round(4)
    result["entry_close"]  = result["entry_close"].round(4)
    result["exit_close"]   = result["exit_close"].round(4)
    result["year"]         = result["entry_date"].dt.year
    result["month"]        = result["entry_date"].dt.month

    cols = ["phase_id", "trade", "entry_date", "exit_date",
            "ticker", "entry_close", "exit_close",
            "entry_close_date", "exit_close_date",
            "return_pct", "nifty_return", "alpha", "year", "month"]
    return result[[c for c in cols if c in result.columns]].reset_index(drop=True)


def quartile_label(rank: int, n: int) -> str:
    """
    Quartile bucket for a 1-based `rank` within a ranked list of size `n`:
      Q4 = Top 25% (best), Q3 = Top 50%, Q2 = Top 75%, Q1 = Below 25% (others).
    """
    try:
        n = int(n)
        rank = int(rank)
    except Exception:
        return "—"
    if n <= 0 or rank <= 0:
        return "—"
    frac = rank / n
    if frac <= 0.25:
        return "Q4 (Top 25%)"
    if frac <= 0.50:
        return "Q3 (Top 50%)"
    if frac <= 0.75:
        return "Q2 (Top 75%)"
    return "Q1 (Below 25%)"


def quartile_from_fraction(frac: float) -> str:
    """Quartile bucket directly from a 0..1 position fraction (smaller = better)."""
    try:
        frac = float(frac)
    except Exception:
        return "—"
    if frac <= 0.25:
        return "Q4 (Top 25%)"
    if frac <= 0.50:
        return "Q3 (Top 50%)"
    if frac <= 0.75:
        return "Q2 (Top 75%)"
    return "Q1 (Below 25%)"


def get_top_n_per_phase(
    returns_df: pd.DataFrame,
    top_n: int = 20,
    sort_by: str = "alpha",
    trade_filter: str = "All",
) -> pd.DataFrame:
    """Return top-N stocks per phase, with rank + quartile bucket."""
    if returns_df.empty:
        return pd.DataFrame()
    df = returns_df.copy()
    if trade_filter != "All":
        df = df[df["trade"] == trade_filter]
    rows = []
    for pid, grp in df.groupby("phase_id"):
        grp = grp.dropna(subset=[sort_by])
        top = grp.nlargest(top_n, sort_by).copy()
        n = len(top)
        top["rank"] = range(1, n + 1)
        top["quartile"] = [quartile_label(r, n) for r in top["rank"]]
        rows.append(top)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
