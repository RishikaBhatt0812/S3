"""
S³ Export — Excel Trade Sheet Generator
========================================
Generates a professional Excel workbook with:
  - Trade Log sheet
  - Per-Stock summary sheet
  - Phase Analysis sheet
  - Window Analysis sheet
  - Portfolio Summary sheet
  - NIFTY Entry Visualisation sheet (shows NIFTY path + trigger point)
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from core.investment_analysis import compute_investment_analysis
except Exception:  # pragma: no cover - fallback if path differs
    from investment_analysis import compute_investment_analysis  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_GREEN_DARK   = "1A6B3C"
_GREEN_LIGHT  = "C6EFCE"
_RED_DARK     = "9C0006"
_RED_LIGHT    = "FFC7CE"
_BLUE_DARK    = "1F497D"
_BLUE_MID     = "BDD7EE"
_BLUE_LIGHT   = "DDEEFF"
_GOLD         = "FFD700"
_HEADER_BG    = "2E4057"
_HEADER_FG    = "FFFFFF"
_ALT_ROW      = "F2F7FF"
_WHITE        = "FFFFFF"
_BORDER       = "9EB6D4"


def _fmt_date(d) -> str:
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return ""
    try:
        return pd.Timestamp(d).strftime("%d-%b-%Y")
    except Exception:
        return str(d)


def _quartile_label_xl(rank: int, n: int) -> str:
    """Q4 top-25% (best) … Q1 below-25% for a 1-based rank within n."""
    try:
        n = int(n); rank = int(rank)
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


# ─────────────────────────────────────────────────────────────────────────────
# Performance metrics (period / yearwise)
# ─────────────────────────────────────────────────────────────────────────────

# Cumulative look-back periods requested. End year auto-extends to the data.
_PERIOD_DEFS = [("2008–2026", 2008), ("2011–2026", 2011), ("2020–2026", 2020)]

# Column layout shared by the period & yearwise summary tables
_SUMMARY_HEADERS = [
    "Period / Year", "Total Trades", "Gross Total %", "Avg %", "Win %",
    "Win Avg %", "Loss Avg %", "Avg Alpha %", "Avg NIFTY %", "Best %", "Worst %",
]
_SUMMARY_WIDTHS = [16, 12, 14, 10, 9, 11, 11, 12, 12, 10, 10]


def _trade_metrics(df: pd.DataFrame) -> dict:
    """Compute the summary stats for a set of trades (uses Return %).

    If the frame carries a `_qnum` column (1..4, where 4 = top quartile), the
    per-quartile trade counts and their % of the period's trades are added too.
    """
    base = {"n": 0, "gross": 0.0, "avg": 0.0, "winpct": 0.0, "winavg": 0.0,
            "lossavg": 0.0, "alpha": 0.0, "nifty": 0.0, "best": 0.0, "worst": 0.0,
            "q1": 0, "q2": 0, "q3": 0, "q4": 0, "q34": 0,
            "q1p": 0.0, "q2p": 0.0, "q3p": 0.0, "q4p": 0.0, "q34p": 0.0}
    if df is None or df.empty:
        return base
    ret = pd.to_numeric(df["return_pct"], errors="coerce").dropna()
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    def _m(s):
        return round(float(s.mean()), 2) if len(s) else 0.0
    alpha = pd.to_numeric(df["alpha"], errors="coerce").dropna() if "alpha" in df.columns else pd.Series(dtype=float)
    nifty = pd.to_numeric(df["nifty_return"], errors="coerce").dropna() if "nifty_return" in df.columns else pd.Series(dtype=float)

    m = {
        "n"      : int(len(ret)),
        "gross"  : round(float(ret.sum()), 2),
        "avg"    : _m(ret),
        "winpct" : round(float((ret > 0).mean() * 100), 1) if len(ret) else 0.0,
        "winavg" : _m(wins),
        "lossavg": _m(losses),
        "alpha"  : _m(alpha),
        "nifty"  : _m(nifty),
        "best"   : round(float(ret.max()), 2) if len(ret) else 0.0,
        "worst"  : round(float(ret.min()), 2) if len(ret) else 0.0,
    }

    # Quartile breakdown (counts + % of this period's trades)
    n = len(df)
    if "_qnum" in df.columns and n > 0:
        qn = pd.to_numeric(df["_qnum"], errors="coerce").fillna(0).astype(int)
        c1 = int((qn == 1).sum()); c2 = int((qn == 2).sum())
        c3 = int((qn == 3).sum()); c4 = int((qn == 4).sum())
        c34 = c3 + c4
        pct = lambda c: round(c / n * 100, 1) if n else 0.0
        m.update({"q1": c1, "q2": c2, "q3": c3, "q4": c4, "q34": c34,
                  "q1p": pct(c1), "q2p": pct(c2), "q3p": pct(c3),
                  "q4p": pct(c4), "q34p": pct(c34)})
    else:
        m.update({"q1": 0, "q2": 0, "q3": 0, "q4": 0, "q34": 0,
                  "q1p": 0.0, "q2p": 0.0, "q3p": 0.0, "q4p": 0.0, "q34p": 0.0})
    return m


def _attach_quartile(per_trade_df: pd.DataFrame, trade_phase_ranks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach `_qnum` (1..4, 4 = top quartile) to each trade using the
    TRADE-PHASE quartile — i.e. where the stock ranked among ALL eligible
    stocks during the actual buy-date → sell-date period.
    """
    df = per_trade_df.copy()
    qmap = {}
    if trade_phase_ranks_df is not None and not trade_phase_ranks_df.empty \
            and "quartile" in trade_phase_ranks_df.columns:
        for _, r in trade_phase_ranks_df.iterrows():
            qmap[(int(r["window_idx"]), str(r["ticker"]))] = r.get("quartile", "")

    def _qnum(row):
        lbl  = str(qmap.get((int(row.get("window_idx", -1)), str(row.get("ticker", ""))), ""))
        head = lbl.split()[0] if lbl.split() else ""
        return int(head[1]) if head[:1] == "Q" and head[1:2].isdigit() else 0

    df["_qnum"] = df.apply(_qnum, axis=1)
    return df


def _with_years(per_trade_df: pd.DataFrame) -> pd.DataFrame:
    df = per_trade_df.copy()
    df["_yr"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.year
    return df


def _period_rows(per_trade_df: pd.DataFrame) -> list[tuple[str, dict]]:
    if per_trade_df is None or per_trade_df.empty:
        return []
    df = _with_years(per_trade_df)
    end_yr = int(df["_yr"].dropna().max()) if df["_yr"].notna().any() else 2026
    end_yr = max(end_yr, 2026)
    out = []
    for label, start in _PERIOD_DEFS:
        sub = df[(df["_yr"] >= start) & (df["_yr"] <= end_yr)]
        lbl = f"{start}–{end_yr}"
        out.append((lbl, _trade_metrics(sub)))
    return out


def _yearwise_rows(per_trade_df: pd.DataFrame) -> list[tuple[str, dict]]:
    if per_trade_df is None or per_trade_df.empty:
        return []
    df = _with_years(per_trade_df)
    out = []
    for yr in sorted(df["_yr"].dropna().unique()):
        sub = df[df["_yr"] == yr]
        out.append((str(int(yr)), _trade_metrics(sub)))
    return out


def _write_summary_table(workbook, ws, start_row: int, title: str,
                         rows: list[tuple[str, dict]], set_widths: bool = True,
                         include_quartiles: bool = False) -> int:
    """
    Write a titled summary table (period or yearwise) starting at `start_row`.
    When `include_quartiles` is True, quartile-count columns (Q1..Q4, Q3+Q4 with
    counts and %) are inserted right after the Total Trades column.
    Returns the next free row.
    """
    title_fmt = workbook.add_format({
        "bold": True, "font_size": 12, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": "4472C4",
        "border": 1, "border_color": _BORDER, "align": "left", "valign": "vcenter",
    })
    base_fmt  = _cell_fmt(workbook)
    label_fmt = _cell_fmt(workbook, bg=_BLUE_LIGHT, bold=True)
    green_fmt = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK, bold=True, align="right")
    red_fmt   = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True, align="right")
    q_fmt     = _cell_fmt(workbook, bg="FFF2CC", align="right")          # quartile count
    qpct_fmt  = _cell_fmt(workbook, bg="FFF7E0", align="right")          # quartile %
    q34_fmt   = _cell_fmt(workbook, bg="DDEBF7", bold=True, align="right")
    num_fmt_c = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                      "border": 1, "border_color": _BORDER, "align": "right"})
    int_fmt_c = workbook.add_format({"num_format": "0", "font_name": "Calibri", "font_size": 9,
                                      "border": 1, "border_color": _BORDER, "align": "right"})
    pct_fmt_c = workbook.add_format({"num_format": "0.0", "font_name": "Calibri", "font_size": 9,
                                      "border": 1, "border_color": _BORDER, "align": "right"})

    # Build headers (+ quartile columns after Total Trades when requested)
    q_headers = ["Q1 %", "Q2 %", "Q3 %", "Q4 %", "Q3+Q4 #", "Q3+Q4 %"]
    q_widths  = [7, 7, 7, 7, 9, 9]
    if include_quartiles:
        headers = _SUMMARY_HEADERS[:2] + q_headers + _SUMMARY_HEADERS[2:]
        widths  = _SUMMARY_WIDTHS[:2] + q_widths + _SUMMARY_WIDTHS[2:]
    else:
        headers = list(_SUMMARY_HEADERS)
        widths  = list(_SUMMARY_WIDTHS)

    ncols = len(headers)
    ws.merge_range(start_row, 0, start_row, ncols - 1, title, title_fmt)
    r = start_row + 1
    hdr_cell = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 9,
    })
    for c, h in enumerate(headers):
        ws.write(r, c, h, hdr_cell)
        if set_widths:
            ws.set_column(c, c, widths[c])
    r += 1

    if not rows:
        ws.write(r, 0, "No trades available.", base_fmt)
        return r + 2

    for label, m in rows:
        c = 0
        ws.write(r, c, label, label_fmt); c += 1
        ws.write(r, c, m["n"], int_fmt_c); c += 1
        if include_quartiles:
            ws.write(r, c, m.get("q1p", 0.0), qpct_fmt); c += 1
            ws.write(r, c, m.get("q2p", 0.0), qpct_fmt); c += 1
            ws.write(r, c, m.get("q3p", 0.0), qpct_fmt); c += 1
            ws.write(r, c, m.get("q4p", 0.0), qpct_fmt); c += 1
            ws.write(r, c, m.get("q34", 0), q34_fmt); c += 1
            ws.write(r, c, m.get("q34p", 0.0), q34_fmt); c += 1
        ws.write(r, c, m["gross"], green_fmt if m["gross"] >= 0 else red_fmt); c += 1
        ws.write(r, c, m["avg"],   green_fmt if m["avg"]   >= 0 else red_fmt); c += 1
        ws.write(r, c, m["winpct"], num_fmt_c); c += 1
        ws.write(r, c, m["winavg"], green_fmt if m["winavg"] >= 0 else red_fmt); c += 1
        ws.write(r, c, m["lossavg"], red_fmt if m["lossavg"] < 0 else num_fmt_c); c += 1
        ws.write(r, c, m["alpha"], green_fmt if m["alpha"] >= 0 else red_fmt); c += 1
        ws.write(r, c, m["nifty"], green_fmt if m["nifty"] >= 0 else red_fmt); c += 1
        ws.write(r, c, m["best"], num_fmt_c); c += 1
        ws.write(r, c, m["worst"], num_fmt_c); c += 1
        r += 1
    return r + 1


def _write_header_row(ws, row: int, headers: list[str], workbook, bold_fmt):
    """Write a styled header row."""
    hfmt = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 10,
    })
    for col, h in enumerate(headers):
        ws.write(row, col, h, hfmt)
    ws.set_row(row, 18)


def _pct_fmt(wb, positive_green: bool = True):
    return wb.add_format({
        "num_format": "0.00%", "font_name": "Calibri", "font_size": 9,
        "align": "right",
    })


def _num_fmt(wb):
    return wb.add_format({"num_format": "#,##0.00", "font_name": "Calibri", "font_size": 9})


def _date_fmt(wb):
    return wb.add_format({"num_format": "dd-mmm-yyyy", "font_name": "Calibri", "font_size": 9})


def _cell_fmt(wb, bg=None, bold=False, align="left", color="000000"):
    d = {"font_name": "Calibri", "font_size": 9, "border": 1,
         "border_color": _BORDER, "align": align, "valign": "vcenter", "font_color": color}
    if bg:
        d["bg_color"] = bg
    if bold:
        d["bold"] = True
    return wb.add_format(d)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 1 — Trade Log
# ─────────────────────────────────────────────────────────────────────────────

def _write_trade_log(workbook, per_trade_df: pd.DataFrame):
    ws = workbook.add_worksheet("Trade Log")
    ws.freeze_panes(1, 0)

    headers = [
        "Window", "Pattern Start", "Pattern End",
        "Buy Phase", "Buy Phase Start", "Buy Phase End",
        "NIFTY Base", "Entry Trigger Level", "NIFTY at Entry", "Entry Date", "Threshold Hit?",
        "Sell Phase", "Sell Phase Start", "Sell Phase End",
        "NIFTY Sell Base", "Exit Trigger Level", "NIFTY at Exit", "Exit Date",
        "Ticker", "Entry Price", "Exit Price",
        "Return %", "NIFTY Return %", "Alpha %", "Days Held"
    ]

    _write_header_row(ws, 0, headers, workbook, None)

    base_fmt   = _cell_fmt(workbook)
    alt_fmt    = _cell_fmt(workbook, bg=_ALT_ROW)
    green_fmt  = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK)
    red_fmt    = _cell_fmt(workbook, bg=_RED_LIGHT,   color=_RED_DARK)
    date_fmt_c = workbook.add_format({"num_format": "dd-mmm-yyyy", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER})
    num_fmt_c  = workbook.add_format({"num_format": "#,##0.00", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER, "align": "right"})
    pct_fmt_c  = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER, "align": "right"})

    col_widths = [8, 14, 14, 10, 14, 14, 12, 14, 12, 14, 12,
                  10, 14, 14, 12, 14, 12, 14, 12, 12, 12, 10, 12, 10, 10]

    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    for r_idx, (_, row) in enumerate(per_trade_df.iterrows()):
        r = r_idx + 1
        alt = (r_idx % 2 == 1)
        bf = alt_fmt if alt else base_fmt
        ret = row.get("return_pct", 0) or 0
        alp = row.get("alpha", 0) or 0

        vals = [
            row.get("window_idx", "") + 1,
            _fmt_date(row.get("pattern_start")),
            _fmt_date(row.get("pattern_end")),
            row.get("buy_phase", ""),
            _fmt_date(row.get("buy_phase_start")),
            _fmt_date(row.get("buy_phase_end")),
            row.get("nifty_buy_base", ""),
            row.get("nifty_entry_trigger_level", ""),
            row.get("nifty_at_entry", ""),
            _fmt_date(row.get("entry_date")),
            "✓" if row.get("entry_threshold_hit") else "—",
            row.get("sell_phase", ""),
            _fmt_date(row.get("sell_phase_start")),
            _fmt_date(row.get("sell_phase_end")),
            row.get("nifty_sell_base", ""),
            row.get("nifty_exit_trigger_level", ""),
            row.get("nifty_at_exit", ""),
            _fmt_date(row.get("exit_date")),
            row.get("ticker", ""),
            row.get("entry_price", ""),
            row.get("exit_price", ""),
            round(float(ret), 2) if ret != "" else "",
            round(float(row.get("nifty_return") or 0), 2),
            round(float(alp), 2) if alp != "" else "",
            row.get("days_held", ""),
        ]

        # Colour return/alpha cells
        ret_fmt = (green_fmt if ret > 0 else red_fmt) if isinstance(ret, (int, float)) else bf
        alp_fmt = (green_fmt if alp > 0 else red_fmt) if isinstance(alp, (int, float)) else bf

        for c_idx, val in enumerate(vals):
            if c_idx == 21:   # return_pct
                ws.write(r, c_idx, val, ret_fmt)
            elif c_idx == 23:  # alpha
                ws.write(r, c_idx, val, alp_fmt)
            elif c_idx in (6, 7, 8, 14, 15, 16):  # price/nifty
                ws.write(r, c_idx, val, num_fmt_c)
            elif c_idx in (19, 20):  # entry/exit price
                ws.write(r, c_idx, val, num_fmt_c)
            elif c_idx == 22:  # nifty return
                ws.write(r, c_idx, val, pct_fmt_c)
            else:
                ws.write(r, c_idx, val, bf)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 2 — Stock Summary
# ─────────────────────────────────────────────────────────────────────────────

def _write_stock_summary(workbook, summary_df: pd.DataFrame):
    ws = workbook.add_worksheet("Stock Summary")
    ws.freeze_panes(1, 1)

    headers = [
        "Rank", "Ticker", "Freq Rank", "Window Freq %",
        "# Trades", "Avg Return %", "Avg Alpha %", "Win Rate %",
        "Positive % Trades", "Max Drawdown %", "Avg Days Held",
        "Annualised Return %", "Threshold Hits"
    ]
    _write_header_row(ws, 0, headers, workbook, None)
    col_widths = [6, 12, 10, 14, 10, 14, 14, 12, 16, 14, 14, 18, 14]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    base_fmt  = _cell_fmt(workbook)
    alt_fmt   = _cell_fmt(workbook, bg=_ALT_ROW)
    green_fmt = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK)
    red_fmt   = _cell_fmt(workbook, bg=_RED_LIGHT,   color=_RED_DARK)
    gold_fmt  = _cell_fmt(workbook, bg=_GOLD, bold=True)
    num_fmt_c = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                      "border": 1, "border_color": _BORDER, "align": "right"})

    for r_idx, (_, row) in enumerate(summary_df.iterrows()):
        r   = r_idx + 1
        alt = (r_idx % 2 == 1)
        bf  = alt_fmt if alt else base_fmt
        avg_alp = row.get("avg_alpha", 0) or 0
        avg_ret = row.get("avg_return", 0) or 0
        win_r   = row.get("win_rate", 0) or 0

        vals = [
            r_idx + 1,
            row.get("ticker", ""),
            row.get("freq_rank", ""),
            row.get("full_window_pct", ""),
            row.get("n_trades", ""),
            round(float(avg_ret), 2),
            round(float(avg_alp), 2),
            round(float(win_r), 1),
            row.get("positive_ret_pct", ""),
            row.get("max_drawdown", ""),
            row.get("avg_days_held", ""),
            row.get("annualised_return", ""),
            row.get("n_threshold_hits", ""),
        ]

        for c_idx, val in enumerate(vals):
            if c_idx == 6:  # avg_alpha
                ws.write(r, c_idx, val, green_fmt if float(avg_alp) > 0 else red_fmt)
            elif c_idx == 5:  # avg_return
                ws.write(r, c_idx, val, green_fmt if float(avg_ret) > 0 else red_fmt)
            elif c_idx in (3, 7, 8, 9, 10, 11) and val != "":
                try:
                    ws.write(r, c_idx, float(val), num_fmt_c)
                except Exception:
                    ws.write(r, c_idx, val, bf)
            else:
                ws.write(r, c_idx, val, gold_fmt if r_idx < 3 and c_idx == 1 else bf)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 3 — NIFTY Entry Visualisation (key visual)
# ─────────────────────────────────────────────────────────────────────────────

def _write_nifty_entry_vis(workbook, per_trade_df: pd.DataFrame, nifty_df: pd.DataFrame):
    """
    For each unique window, show the NIFTY path during the buy phase
    with the base level, trigger level, and actual entry point highlighted.
    """
    ws = workbook.add_worksheet("NIFTY Entry Visualisation")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range("A1:L1", "NIFTY Entry Trigger Visualisation — Buy Phase NIFTY Path", title_fmt)
    ws.set_row(0, 22)

    headers = [
        "Window", "Buy Phase", "Phase Start", "Phase End",
        "NIFTY Base", "Entry Threshold %", "Trigger Level",
        "Date", "NIFTY Close", "vs Base", "vs Trigger", "Entry Triggered?"
    ]
    _write_header_row(ws, 1, headers, workbook, None)
    col_widths = [8, 10, 14, 14, 12, 14, 14, 14, 12, 10, 10, 16]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    # Get unique windows from per_trade_df
    if per_trade_df.empty or nifty_df.empty:
        ws.write(2, 0, "No trade data available.", workbook.add_format({"font_name": "Calibri"}))
        return

    base_fmt    = _cell_fmt(workbook)
    trigger_fmt = _cell_fmt(workbook, bg="FFF2CC", bold=True)  # yellow = trigger row
    hit_fmt     = _cell_fmt(workbook, bg=_GREEN_LIGHT, bold=True, color=_GREEN_DARK)
    below_fmt   = _cell_fmt(workbook, bg=_BLUE_LIGHT)
    num_fmt_c   = workbook.add_format({"num_format": "#,##0.00", "font_name": "Calibri",
                                        "font_size": 9, "border": 1, "border_color": _BORDER})
    pct_fmt_c   = workbook.add_format({"num_format": "0.00", "font_name": "Calibri",
                                        "font_size": 9, "border": 1, "border_color": _BORDER})

    current_row = 2
    seen_windows = set()

    for _, trade_row in per_trade_df.drop_duplicates("window_idx").iterrows():
        w_idx    = int(trade_row["window_idx"])
        if w_idx in seen_windows:
            continue
        seen_windows.add(w_idx)

        buy_start = pd.Timestamp(trade_row["buy_phase_start"])
        buy_end   = pd.Timestamp(trade_row["buy_phase_end"])
        n_base    = trade_row.get("nifty_buy_base") or 0
        thr_pct   = trade_row.get("entry_threshold_pct") or 0
        trig_lvl  = trade_row.get("nifty_entry_trigger_level") or (n_base * (1 + thr_pct/100))
        buy_phase = trade_row.get("buy_phase", "")
        entry_triggered_date = trade_row.get("entry_triggered_on_date")
        if entry_triggered_date is not None:
            try:
                entry_triggered_date = pd.Timestamp(entry_triggered_date)
            except Exception:
                entry_triggered_date = None

        # Get NIFTY data within buy phase
        phase_nifty = nifty_df[(nifty_df.index >= buy_start) & (nifty_df.index <= buy_end)]
        if phase_nifty.empty:
            continue

        phase_start_fmt = workbook.add_format({
            "bold": True, "font_name": "Calibri", "font_size": 9,
            "bg_color": _BLUE_MID, "border": 1, "border_color": _BORDER,
        })
        # Section header for this window
        ws.merge_range(current_row, 0, current_row, 11,
                       f"Window {w_idx + 1}  |  Buy Phase: {buy_start.strftime('%d-%b-%Y')} → {buy_end.strftime('%d-%b-%Y')}  |  NIFTY Base: {n_base:.2f}  |  Trigger @ +{thr_pct:.1f}% = {trig_lvl:.2f}",
                       phase_start_fmt)
        ws.set_row(current_row, 16)
        current_row += 1

        for date, nrow in phase_nifty.iterrows():
            close    = float(nrow["close"])
            vs_base  = round((close - n_base) / n_base * 100, 2) if n_base else 0
            vs_trig  = round((close - trig_lvl) / trig_lvl * 100, 2) if trig_lvl else 0
            is_entry = (entry_triggered_date is not None and date.date() == entry_triggered_date.date())
            is_above_trigger = close >= trig_lvl

            if is_entry:
                row_fmt = hit_fmt
                triggered_label = f"✓ TRIGGERED (+{thr_pct:.1f}%)"
            elif is_above_trigger:
                row_fmt = trigger_fmt
                triggered_label = "Above trigger"
            else:
                row_fmt = below_fmt
                triggered_label = "—"

            ws.write(current_row, 0,  w_idx + 1,                        row_fmt)
            ws.write(current_row, 1,  buy_phase,                         row_fmt)
            ws.write(current_row, 2,  buy_start.strftime("%d-%b-%Y"),    row_fmt)
            ws.write(current_row, 3,  buy_end.strftime("%d-%b-%Y"),      row_fmt)
            ws.write(current_row, 4,  n_base,                            num_fmt_c)
            ws.write(current_row, 5,  thr_pct,                           pct_fmt_c)
            ws.write(current_row, 6,  trig_lvl,                          num_fmt_c)
            ws.write(current_row, 7,  date.strftime("%d-%b-%Y"),         row_fmt)
            ws.write(current_row, 8,  close,                             num_fmt_c)
            ws.write(current_row, 9,  vs_base,                           pct_fmt_c)
            ws.write(current_row, 10, vs_trig,                           pct_fmt_c)
            ws.write(current_row, 11, triggered_label,                   hit_fmt if is_entry else row_fmt)
            current_row += 1

        current_row += 1  # blank row between windows


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 4 — Window Analysis (Common Stock Selection)
# ─────────────────────────────────────────────────────────────────────────────

def _write_window_analysis(workbook, windows: list[dict], freq_df: pd.DataFrame,
                            returns_df: pd.DataFrame, top_n: int, sort_by: str,
                            window_status_df: pd.DataFrame = None):
    ws = workbook.add_worksheet("Window Analysis")
    ws.freeze_panes(1, 0)

    headers = ["Window", "Leg", "Phase ID", "Trade", "Phase Start", "Phase End",
               "Ticker", "Rank in Leg", "Quartile", "Alpha %", "Return %", "Common Across Legs?"]
    _write_header_row(ws, 0, headers, workbook, None)
    col_widths = [8, 6, 10, 8, 14, 14, 12, 12, 14, 10, 10, 18]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    base_fmt   = _cell_fmt(workbook)
    common_fmt = _cell_fmt(workbook, bg=_GREEN_LIGHT, bold=True, color=_GREEN_DARK)
    alt_fmt    = _cell_fmt(workbook, bg=_ALT_ROW)
    num_fmt_c  = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER})

    current_row = 1
    # Map window number → reshuffled flag/reason
    resh_map = {}
    if window_status_df is not None and not window_status_df.empty and "reshuffled" in window_status_df.columns:
        for _, sr in window_status_df.iterrows():
            resh_map[int(sr.get("window", 0))] = (bool(sr.get("reshuffled")), sr.get("reason", ""))

    resh_banner_fmt = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True)

    for win in windows:
        w_disp = win["window_idx"] + 1
        flag = resh_map.get(w_disp)
        if flag and flag[0]:
            ws.merge_range(current_row, 0, current_row, 11,
                           f"Window {w_disp}  —  RESHUFFLED (skipped): {flag[1]}", resh_banner_fmt)
            current_row += 1
            continue
        n_legs = len(win["phase_ids"])
        leg_sets = []
        for li, pid in enumerate(win["phase_ids"]):
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
            leg_sets.append(set(ph.nlargest(top_n, sort_by)["ticker"].tolist()) if not ph.empty else set())
        common = set.intersection(*leg_sets) if leg_sets else set()

        for li, pid in enumerate(win["phase_ids"]):
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[sort_by])
            if ph.empty:
                continue
            top = ph.nlargest(top_n, sort_by).reset_index(drop=True)
            n_leg = len(top)

            for rank_idx, trow in top.iterrows():
                tkr = trow["ticker"]
                is_c = tkr in common
                rf = common_fmt if is_c else (alt_fmt if rank_idx % 2 else base_fmt)

                ws.write(current_row, 0,  win["window_idx"] + 1,         rf)
                ws.write(current_row, 1,  li + 1,                        rf)
                ws.write(current_row, 2,  pid,                           rf)
                ws.write(current_row, 3,  trow.get("trade", ""),         rf)
                ws.write(current_row, 4,  _fmt_date(trow.get("entry_date")), rf)
                ws.write(current_row, 5,  _fmt_date(trow.get("exit_date")),  rf)
                ws.write(current_row, 6,  tkr,                           rf)
                ws.write(current_row, 7,  rank_idx + 1,                  rf)
                ws.write(current_row, 8,  _quartile_label_xl(rank_idx + 1, n_leg), rf)
                ws.write(current_row, 9,  round(float(trow.get("alpha", 0) or 0), 2), num_fmt_c)
                ws.write(current_row, 10, round(float(trow.get("return_pct", 0) or 0), 2), num_fmt_c)
                ws.write(current_row, 11, "✓ Common" if is_c else "—",   common_fmt if is_c else rf)
                current_row += 1


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 5 — Phase Schedule
# ─────────────────────────────────────────────────────────────────────────────

def _write_phase_schedule(workbook, phases: pd.DataFrame):
    ws = workbook.add_worksheet("Phase Schedule")
    ws.freeze_panes(1, 0)
    headers = ["Phase ID", "Trade", "Entry Date", "Exit Date", "Days"]
    _write_header_row(ws, 0, headers, workbook, None)
    col_widths = [10, 10, 14, 14, 8]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    rise_fmt = _cell_fmt(workbook, bg="C6EFCE", color=_GREEN_DARK)
    fall_fmt = _cell_fmt(workbook, bg=_RED_LIGHT,  color=_RED_DARK)
    base_fmt = _cell_fmt(workbook)

    for r_idx, (_, row) in enumerate(phases.iterrows()):
        r = r_idx + 1
        trade = row.get("trade", "")
        rf = rise_fmt if trade == "Rise" else (fall_fmt if trade == "Fall" else base_fmt)
        ws.write(r, 0, int(row["phase_id"]), rf)
        ws.write(r, 1, trade, rf)
        ws.write(r, 2, _fmt_date(row["entry_date"]), rf)
        ws.write(r, 3, _fmt_date(row["exit_date"]),  rf)
        ws.write(r, 4, int(row.get("days", 0) or 0), rf)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet 6 — Portfolio Summary
# ─────────────────────────────────────────────────────────────────────────────

def _write_portfolio_summary(workbook, per_trade_df: pd.DataFrame, summary_df: pd.DataFrame,
                              config: dict, window_status_df: pd.DataFrame = None):
    ws = workbook.add_worksheet("Portfolio Summary")

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 14, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    section_fmt = workbook.add_format({
        "bold": True, "font_size": 11, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": "4472C4",
        "border": 1, "border_color": _BORDER,
    })
    label_fmt = workbook.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": _BLUE_LIGHT, "border": 1, "border_color": _BORDER,
    })
    val_fmt = workbook.add_format({
        "font_name": "Calibri", "font_size": 10,
        "border": 1, "border_color": _BORDER,
    })
    green_val = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True,
        "font_color": _GREEN_DARK, "bg_color": _GREEN_LIGHT,
        "border": 1, "border_color": _BORDER,
    })
    red_val = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True,
        "font_color": _RED_DARK, "bg_color": _RED_LIGHT,
        "border": 1, "border_color": _BORDER,
    })

    ws.set_column(0, 0, 30)
    ws.set_column(1, 1, 25)
    ws.set_column(2, 2, 30)
    ws.set_column(3, 3, 25)

    ws.merge_range("A1:D1", f"S³ Portfolio Summary — Generated {datetime.now().strftime('%d %b %Y %H:%M')}", title_fmt)
    ws.set_row(0, 24)

    # ── Config block (columns A:B, rows 2-10) ────────────────────────────────
    ws.merge_range("A2:B2", "Strategy Configuration", section_fmt)
    ws.set_row(1, 18)

    # ── Volatility-filter labels (so the active filters are visible here) ─────
    _lowvol_on = bool(config.get("lowvol_enabled"))
    _lowvol_lbl = (f"On — keep ≤ {config.get('lowvol_pct', 50)}th pct (calmest)"
                   if _lowvol_on else "Off")
    _winvol_mode = config.get("winvol_mode", "off")
    if _winvol_mode == "low":
        _winvol_lbl = f"Low (calmest) — keep bottom {config.get('winvol_pct', 50)}%"
    elif _winvol_mode == "high":
        _winvol_lbl = f"High (wildest) — keep top {config.get('winvol_pct', 50)}%"
    else:
        _winvol_lbl = "Off"

    config_rows = [
        ("Pattern", " → ".join(config.get("pattern", []))),
        ("Leg Count", config.get("leg_count", "")),
        ("Top N Stocks", config.get("top_n", "")),
        ("Top K Common", config.get("top_k_common", "")),
        ("Entry Threshold %", f"{config.get('entry_threshold_pct', 0):.1f}%"),
        ("Exit Threshold %",  f"{config.get('exit_threshold_pct', 0):.1f}%"),
        ("Reshuffle Enabled", "Yes" if config.get("reshuffle_enabled") else "No"),
        ("Reshuffle Threshold %", f"{config.get('reshuffle_threshold', 0):.1f}%" if config.get("reshuffle_enabled") else "—"),
        ("Entry Segment Included", "Yes" if config.get("include_entry_segment", True) else "No (pattern legs only)"),
        ("Persistence Enabled", "Yes" if config.get("persistence_enabled") else "No"),
        ("Sort By", config.get("sort_by", "alpha")),
        ("Low-Volatility Filter", _lowvol_lbl),
        ("Window Volatility Filter", _winvol_lbl),
    ]
    for i, (k, v) in enumerate(config_rows):
        ws.write(2 + i, 0, k, label_fmt)
        ws.write(2 + i, 1, str(v), val_fmt)

    # ── Portfolio metrics (columns C:D, rows 2-10) ────────────────────────────
    if not per_trade_df.empty:
        all_rets = per_trade_df["return_pct"].dropna()
        all_alp  = per_trade_df["alpha"].dropna()
        n_trades = len(per_trade_df)
        n_wins   = int((all_alp > 0).sum())
        win_rate = round(n_wins / len(all_alp) * 100, 1) if len(all_alp) > 0 else 0
        avg_ret  = round(all_rets.mean(), 2)
        avg_alp_ = round(all_alp.mean(), 2)
        max_dd   = round(all_rets.min(), 2)
        hit_count = int(per_trade_df.get("entry_threshold_hit", pd.Series()).sum()) if "entry_threshold_hit" in per_trade_df.columns else 0
        hit_rate = round(hit_count / len(per_trade_df.drop_duplicates("window_idx")) * 100, 1) if n_trades > 0 else 0

        # Header for metrics block — separate columns C:D, same row 2 (no overlap)
        ws.merge_range("C2:D2", "Portfolio Metrics", section_fmt)
        metric_rows = [
            ("Total Trades",        n_trades),
            ("Winning Trades (Alpha>0)", n_wins),
            ("Win Rate %",          win_rate),
            ("Avg Return %",        avg_ret),
            ("Avg Alpha %",         avg_alp_),
            ("Max Single Drawdown", max_dd),
            ("NIFTY Threshold Hit Rate %", hit_rate),
        ]
        for i, (k, v) in enumerate(metric_rows):
            ws.write(2 + i, 2, k, label_fmt)
            fmt_ = green_val if isinstance(v, (int, float)) and v > 0 else (red_val if isinstance(v, (int, float)) and v < 0 else val_fmt)
            ws.write(2 + i, 3, v, fmt_)

    # ── Window status block (rows below the config/metrics blocks) ────────────
    if window_status_df is not None and not window_status_df.empty:
        # Config block occupies rows 2 .. (2 + len(config_rows) - 1); leave a gap.
        base_r = 2 + len(config_rows) + 2
        ws.merge_range(base_r, 0, base_r, 3, "Window Outcomes", section_fmt)
        n_total  = len(window_status_df)
        n_resh   = int(window_status_df["reshuffled"].sum()) if "reshuffled" in window_status_df.columns else 0
        n_traded = int((window_status_df["n_trades"] > 0).sum()) if "n_trades" in window_status_df.columns else 0
        n_nostk  = n_total - n_resh - n_traded
        win_rows = [
            ("Total Windows Found", n_total),
            ("Windows Traded", n_traded),
            ("Windows Reshuffled (skipped)", n_resh),
            ("Windows w/ No Common Stocks / No Buy Phase", max(n_nostk, 0)),
        ]
        for i, (k, v) in enumerate(win_rows):
            ws.write(base_r + 1 + i, 0, k, label_fmt)
            ws.write(base_r + 1 + i, 1, v, val_fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Window Status (reshuffle reasons + per-window trade counts)
# ─────────────────────────────────────────────────────────────────────────────

def _write_window_status(workbook, window_status_df: pd.DataFrame):
    ws = workbook.add_worksheet("Window Status")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range("A1:I1", "Window Status — Traded / Reshuffled (with reasons)", title_fmt)
    ws.set_row(0, 22)

    headers = ["Window", "Pattern Start", "Pattern End", "Buy Phase Start",
               "Fall-Leg Drawdown %", "Status", "Common Candidates", "Trades", "Reason / Notes"]
    _write_header_row(ws, 1, headers, workbook, None)
    col_widths = [8, 14, 14, 14, 16, 20, 14, 8, 60]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    base_fmt   = _cell_fmt(workbook)
    resh_fmt   = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True)
    trade_fmt  = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK)
    none_fmt   = _cell_fmt(workbook, bg="FFF2CC")
    num_fmt_c  = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER, "align": "right"})

    if window_status_df is None or window_status_df.empty:
        ws.write(2, 0, "No window data.", base_fmt)
        return

    for r_idx, (_, row) in enumerate(window_status_df.iterrows()):
        r = r_idx + 2
        reshuffled = bool(row.get("reshuffled"))
        n_tr = int(row.get("n_trades", 0) or 0)
        rf = resh_fmt if reshuffled else (trade_fmt if n_tr > 0 else none_fmt)
        fdd = row.get("fall_drawdown_pct")
        ws.write(r, 0, int(row.get("window", r_idx + 1)), rf)
        ws.write(r, 1, _fmt_date(row.get("pattern_start")), rf)
        ws.write(r, 2, _fmt_date(row.get("pattern_end")), rf)
        ws.write(r, 3, _fmt_date(row.get("buy_phase_start")), rf)
        if fdd is None or (isinstance(fdd, float) and np.isnan(fdd)):
            ws.write(r, 4, "—", rf)
        else:
            ws.write(r, 4, round(float(fdd), 2), num_fmt_c)
        ws.write(r, 5, row.get("status", ""), rf)
        ws.write(r, 6, int(row.get("n_common_candidates", 0) or 0), rf)
        ws.write(r, 7, n_tr, rf)
        ws.write(r, 8, row.get("reason", ""), rf)


# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Window Candidates (per-window three-leg common stocks)
# ─────────────────────────────────────────────────────────────────────────────

def _write_window_candidates(workbook, candidates_df: pd.DataFrame, window_status_df: pd.DataFrame,
                             per_trade_df: pd.DataFrame = None):
    ws = workbook.add_worksheet("Window Candidates")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range("A1:H1", "Per-Window Common Candidates (with Quartile) — pattern legs (+ entry segment if enabled)", title_fmt)
    ws.set_row(0, 22)

    base_fmt    = _cell_fmt(workbook)
    hdr_fmt     = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 10,
    })
    win_fmt     = workbook.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": _BLUE_MID, "border": 1, "border_color": _BORDER,
    })
    resh_fmt    = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True)
    traded_fmt  = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK, bold=True)
    num_fmt_c   = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                        "border": 1, "border_color": _BORDER, "align": "right"})

    # ── Performance summary by period (top of sheet) ───────────────────────────
    start_after = 2
    if per_trade_df is not None and not per_trade_df.empty:
        next_r = _write_summary_table(
            workbook, ws, 2,
            "Performance Summary by Period (executed common-stock trades, by entry year)",
            _period_rows(per_trade_df), set_widths=True)
        start_after = next_r + 1  # leave a gap before the per-window blocks

    if (candidates_df is None or candidates_df.empty) and (window_status_df is None or window_status_df.empty):
        ws.write(start_after, 0, "No candidate data.", base_fmt)
        return

    # Build per-window blocks. Keep leg columns grouped: rank, Q, alpha, ret per leg.
    leg_cols = []
    if candidates_df is not None and not candidates_df.empty:
        leg_cols = [c for c in candidates_df.columns
                    if c.endswith("| rank") or c.endswith("| Q")
                    or c.endswith("| alpha") or c.endswith("| ret")]

    headers = ["Common Rank", "Ticker", "Quartile", "Mean Alpha", "Mean Return", "Traded?"] + leg_cols
    col_widths = [12, 12, 14, 12, 12, 10] + [16] * len(leg_cols)
    n_lead = 6

    current_row = start_after
    status_map = {}
    if window_status_df is not None and not window_status_df.empty:
        for _, sr in window_status_df.iterrows():
            status_map[int(sr.get("window", 0))] = sr

    windows_order = []
    if window_status_df is not None and not window_status_df.empty:
        windows_order = list(window_status_df["window"].astype(int))
    elif candidates_df is not None and not candidates_df.empty:
        windows_order = sorted(candidates_df["window"].astype(int).unique())

    for w in windows_order:
        sr = status_map.get(w)
        # Window header line
        if sr is not None and bool(sr.get("reshuffled")):
            ws.merge_range(current_row, 0, current_row, max(5, len(headers) - 1),
                           f"Window {w}  —  RESHUFFLED: {sr.get('reason','')}", resh_fmt)
            ws.set_row(current_row, 16)
            current_row += 2
            continue

        label = f"Window {w}"
        if sr is not None:
            label += f"  |  {sr.get('status','')}  |  Trades: {int(sr.get('n_trades',0) or 0)}"
        ws.merge_range(current_row, 0, current_row, max(5, len(headers) - 1), label, win_fmt)
        ws.set_row(current_row, 16)
        current_row += 1

        # Header row for the block
        for c_idx, h in enumerate(headers):
            ws.write(current_row, c_idx, h, hdr_fmt)
            if w == windows_order[0]:
                ws.set_column(c_idx, c_idx, col_widths[c_idx])
        current_row += 1

        block = pd.DataFrame()
        if candidates_df is not None and not candidates_df.empty:
            block = candidates_df[candidates_df["window"] == w].sort_values("common_rank")

        if block.empty:
            ws.write(current_row, 0, "No common stocks across the selected legs.", base_fmt)
            current_row += 2
            continue

        for _, cr in block.iterrows():
            traded = bool(cr.get("traded"))
            rf = traded_fmt if traded else base_fmt
            ws.write(current_row, 0, int(cr.get("common_rank", 0) or 0), rf)
            ws.write(current_row, 1, cr.get("ticker", ""), rf)
            ws.write(current_row, 2, cr.get("quartile", "—"), rf)
            ws.write(current_row, 3, round(float(cr.get("mean_alpha", 0) or 0), 2), num_fmt_c)
            ws.write(current_row, 4, round(float(cr.get("mean_return", 0) or 0), 2), num_fmt_c)
            ws.write(current_row, 5, "✓ Yes" if traded else "—", rf)
            for j, lc in enumerate(leg_cols):
                val = cr.get(lc)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    ws.write(current_row, n_lead + j, "—", rf)
                elif lc.endswith("| Q"):
                    ws.write(current_row, n_lead + j, str(val), rf)
                else:
                    try:
                        ws.write(current_row, n_lead + j, round(float(val), 2), num_fmt_c)
                    except Exception:
                        ws.write(current_row, n_lead + j, val, rf)
            current_row += 1
        current_row += 1  # blank line between windows


# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Window Stock Ranks (ALL considered stocks per window, ranked)
# ─────────────────────────────────────────────────────────────────────────────

def _write_window_stock_ranks(workbook, window_ranks_df: pd.DataFrame, window_status_df: pd.DataFrame):
    ws = workbook.add_worksheet("Window Stock Ranks")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range("A1:J1",
                   "Per-Window Full Ranking — every stock valid across all legs (BUY rows highlighted)",
                   title_fmt)
    ws.set_row(0, 22)

    base_fmt   = _cell_fmt(workbook)
    hdr_fmt    = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 10,
    })
    win_fmt    = workbook.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": _BLUE_MID, "border": 1, "border_color": _BORDER,
    })
    resh_fmt   = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True)
    buy_fmt    = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK, bold=True)
    num_fmt_c  = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                       "border": 1, "border_color": _BORDER, "align": "right"})

    if window_ranks_df is None or window_ranks_df.empty:
        ws.write(2, 0, "No ranking data (all windows reshuffled / no buy phase).", base_fmt)
        return

    leg_cols = [c for c in window_ranks_df.columns
                if c.endswith("| rank") or c.endswith("| Q")]
    headers = ["Window Rank", "Ticker", "Pattern Quartile (Ref)", "Mean Alpha", "Buying?"] + leg_cols
    col_widths = [12, 12, 16, 12, 10] + [16] * len(leg_cols)
    n_lead = 5

    # status map for reshuffle banners
    status_map = {}
    if window_status_df is not None and not window_status_df.empty:
        for _, sr in window_status_df.iterrows():
            status_map[int(sr.get("window", 0))] = sr

    # Show windows in natural order: union of status windows and ranked windows
    windows_order = []
    if window_status_df is not None and not window_status_df.empty:
        windows_order = list(window_status_df["window"].astype(int))
    else:
        windows_order = sorted(window_ranks_df["window"].astype(int).unique())

    current_row = 2
    first_block = True
    for w in windows_order:
        sr = status_map.get(w)
        if sr is not None and bool(sr.get("reshuffled")):
            ws.merge_range(current_row, 0, current_row, max(4, len(headers) - 1),
                           f"Window {w}  —  RESHUFFLED: {sr.get('reason','')}", resh_fmt)
            ws.set_row(current_row, 16)
            current_row += 2
            continue

        block = window_ranks_df[window_ranks_df["window"] == w].sort_values("window_rank")
        if block.empty:
            continue

        n_buy = int(block["buying"].sum()) if "buying" in block.columns else 0
        label = f"Window {w}   |   {len(block)} stocks considered   |   {n_buy} selected to BUY"
        ws.merge_range(current_row, 0, current_row, max(4, len(headers) - 1), label, win_fmt)
        ws.set_row(current_row, 16)
        current_row += 1

        for c_idx, h in enumerate(headers):
            ws.write(current_row, c_idx, h, hdr_fmt)
            if first_block:
                ws.set_column(c_idx, c_idx, col_widths[c_idx])
        first_block = False
        current_row += 1

        for _, rr in block.iterrows():
            buying = bool(rr.get("buying"))
            rf = buy_fmt if buying else base_fmt
            ws.write(current_row, 0, int(rr.get("window_rank", 0) or 0), rf)
            ws.write(current_row, 1, rr.get("ticker", ""), rf)
            ws.write(current_row, 2, rr.get("window_quartile", "—"), rf)
            ws.write(current_row, 3, round(float(rr.get("mean_alpha", 0) or 0), 2), num_fmt_c)
            ws.write(current_row, 4, "✓ BUY" if buying else "—", rf)
            for j, lc in enumerate(leg_cols):
                val = rr.get(lc)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    ws.write(current_row, n_lead + j, "—", rf)
                elif lc.endswith("| Q"):
                    ws.write(current_row, n_lead + j, str(val), rf)
                else:
                    try:
                        ws.write(current_row, n_lead + j, int(val), rf)
                    except Exception:
                        ws.write(current_row, n_lead + j, val, rf)
            current_row += 1
        current_row += 1  # blank line between windows


# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Common Stocks P&L (entry buy / exit sell / profit-loss)
# ─────────────────────────────────────────────────────────────────────────────

def _write_common_pnl(workbook, per_trade_df: pd.DataFrame, trade_phase_ranks_df: pd.DataFrame = None):
    ws = workbook.add_worksheet("Common Stocks P&L")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range("A1:L1", "Common Stocks — Entry (BUY), Exit (SELL), Trade-Phase Quartile & Profit / Loss", title_fmt)
    ws.set_row(0, 22)

    headers = ["Window", "Ticker", "Trade-Phase Quartile", "Entry (BUY) Date", "Entry Price",
               "Exit (SELL) Date", "Exit Price", "Return %", "NIFTY Return %",
               "Alpha %", "Days Held", "Result"]
    _write_header_row(ws, 1, headers, workbook, None)
    col_widths = [8, 12, 20, 16, 12, 16, 12, 10, 14, 10, 10, 10]
    for i, w in enumerate(col_widths):
        ws.set_column(i, i, w)

    base_fmt  = _cell_fmt(workbook)
    alt_fmt   = _cell_fmt(workbook, bg=_ALT_ROW)
    green_fmt = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK, bold=True)
    red_fmt   = _cell_fmt(workbook, bg=_RED_LIGHT, color=_RED_DARK, bold=True)
    num_fmt_c = workbook.add_format({"num_format": "0.00", "font_name": "Calibri", "font_size": 9,
                                      "border": 1, "border_color": _BORDER, "align": "right"})

    if per_trade_df is None or per_trade_df.empty:
        ws.write(2, 0, "No trades — all windows may have been reshuffled. See Window Status sheet.", base_fmt)
        return

    # Build quartile map from TRADE-PHASE ranking
    qmap = {}
    if trade_phase_ranks_df is not None and not trade_phase_ranks_df.empty \
            and "quartile" in trade_phase_ranks_df.columns:
        for _, cr in trade_phase_ranks_df.iterrows():
            qmap[(int(cr["window_idx"]), str(cr["ticker"]))] = cr.get("quartile", "—")

    df = per_trade_df.sort_values(["window_idx", "alpha"], ascending=[True, False])
    for r_idx, (_, row) in enumerate(df.iterrows()):
        r = r_idx + 2
        ret = float(row.get("return_pct", 0) or 0)
        bf = alt_fmt if r_idx % 2 else base_fmt
        res_fmt = green_fmt if ret > 0 else red_fmt
        q = qmap.get((int(row.get("window_idx", 0)), str(row.get("ticker", ""))), "—")
        ws.write(r, 0, int(row.get("window_idx", 0)) + 1, bf)
        ws.write(r, 1, row.get("ticker", ""), bf)
        ws.write(r, 2, q, bf)
        ws.write(r, 3, _fmt_date(row.get("entry_date")), bf)
        ws.write(r, 4, round(float(row.get("entry_price", 0) or 0), 2), num_fmt_c)
        ws.write(r, 5, _fmt_date(row.get("exit_date")), bf)
        ws.write(r, 6, round(float(row.get("exit_price", 0) or 0), 2), num_fmt_c)
        ws.write(r, 7, round(ret, 2), res_fmt)
        ws.write(r, 8, round(float(row.get("nifty_return", 0) or 0), 2), num_fmt_c)
        ws.write(r, 9, round(float(row.get("alpha", 0) or 0), 2), res_fmt)
        ws.write(r, 10, int(row.get("days_held", 0) or 0), bf)
        ws.write(r, 11, "PROFIT" if ret > 0 else "LOSS", res_fmt)



# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Trade Phase Rankings  ★ NEW
# ─────────────────────────────────────────────────────────────────────────────

def _write_trade_phase_ranks(workbook, trade_phase_ranks_df: pd.DataFrame):
    """
    NEW SHEET: For every traded window, show ALL eligible stocks ranked by their
    actual return from BUY date → SELL date.  Stocks that were actually traded
    (selected by the engine) are highlighted in green.

    The Quartile shown here (Q4 = top 25% … Q1 = bottom 25%) is the trade-phase
    quartile — this is the authoritative quartile used throughout the workbook.
    """
    ws = workbook.add_worksheet("Trade Phase Rankings")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range(
        "A1:M1",
        "Trade Phase Rankings — ALL eligible stocks ranked by actual Buy→Sell return  |  "
        "Green rows = actually traded  |  Quartile: Q4 = top 25%  …  Q1 = bottom 25%",
        title_fmt,
    )
    ws.set_row(0, 24)

    if trade_phase_ranks_df is None or trade_phase_ranks_df.empty:
        ws.write(2, 0, "No trade-phase data available (no trades executed).",
                 _cell_fmt(workbook))
        return

    headers = [
        "Window", "Rank", "Ticker", "Quartile",
        "Entry Date", "Exit Date",
        "Entry Price", "Exit Price",
        "Return %", "NIFTY Return %", "Alpha %",
        "Eligible Stocks", "Selected (Traded)?",
    ]
    col_widths = [8, 7, 13, 18, 14, 14, 12, 12, 10, 14, 10, 15, 18]

    hdr_fmt = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 10,
    })
    win_fmt = workbook.add_format({
        "bold": True, "font_name": "Calibri", "font_size": 10,
        "bg_color": _BLUE_MID, "border": 1, "border_color": _BORDER,
    })
    base_fmt    = _cell_fmt(workbook)
    alt_fmt     = _cell_fmt(workbook, bg=_ALT_ROW)
    traded_fmt  = _cell_fmt(workbook, bg=_GREEN_LIGHT, color=_GREEN_DARK, bold=True)
    green_fmt   = workbook.add_format({
        "num_format": "0.00", "font_name": "Calibri", "font_size": 9,
        "border": 1, "border_color": _BORDER, "align": "right",
        "bg_color": _GREEN_LIGHT, "font_color": _GREEN_DARK, "bold": True,
    })
    red_fmt     = workbook.add_format({
        "num_format": "0.00", "font_name": "Calibri", "font_size": 9,
        "border": 1, "border_color": _BORDER, "align": "right",
        "bg_color": _RED_LIGHT, "font_color": _RED_DARK, "bold": True,
    })
    num_fmt_c   = workbook.add_format({
        "num_format": "0.00", "font_name": "Calibri", "font_size": 9,
        "border": 1, "border_color": _BORDER, "align": "right",
    })
    num_trd_fmt = workbook.add_format({
        "num_format": "0.00", "font_name": "Calibri", "font_size": 9,
        "border": 1, "border_color": _BORDER, "align": "right",
        "bg_color": _GREEN_LIGHT, "font_color": _GREEN_DARK, "bold": True,
    })

    # Write header once
    for c_idx, (h, cw) in enumerate(zip(headers, col_widths)):
        ws.write(1, c_idx, h, hdr_fmt)
        ws.set_column(c_idx, c_idx, cw)
    ws.set_row(1, 18)

    current_row = 2
    for widx in sorted(trade_phase_ranks_df["window_idx"].unique()):
        block = (trade_phase_ranks_df[trade_phase_ranks_df["window_idx"] == widx]
                   .sort_values("rank")
                   .reset_index(drop=True))
        if block.empty:
            continue

        n_traded = int(block["traded"].sum())
        n_total  = len(block)
        label = (f"Window {widx + 1}   |   {n_total} eligible stocks ranked by buy→sell return"
                 f"   |   {n_traded} actually traded (highlighted)")
        ws.merge_range(current_row, 0, current_row, len(headers) - 1, label, win_fmt)
        ws.set_row(current_row, 16)
        current_row += 1

        for row_i, (_, rr) in enumerate(block.iterrows()):
            traded   = bool(rr.get("traded", False))
            ret      = float(rr.get("return_pct", 0) or 0)
            alpha    = float(rr.get("alpha", 0) or 0)

            # Base row format: green highlight for traded stocks, alt row for others
            if traded:
                rf = traded_fmt
                rn = num_trd_fmt
            else:
                rf = alt_fmt if row_i % 2 else base_fmt
                rn = num_fmt_c

            ret_fmt  = (green_fmt if ret   >= 0 else red_fmt) if not traded else rn
            alp_fmt  = (green_fmt if alpha >= 0 else red_fmt) if not traded else rn

            ws.write(current_row, 0,  int(widx) + 1,                              rf)
            ws.write(current_row, 1,  int(rr.get("rank", 0) or 0),                rf)
            ws.write(current_row, 2,  str(rr.get("ticker", "")),                  rf)
            ws.write(current_row, 3,  str(rr.get("quartile", "—")),               rf)
            ws.write(current_row, 4,  _fmt_date(rr.get("entry_date")),            rf)
            ws.write(current_row, 5,  _fmt_date(rr.get("exit_date")),             rf)
            ws.write(current_row, 6,  round(float(rr.get("entry_price", 0) or 0), 2), num_fmt_c)
            ws.write(current_row, 7,  round(float(rr.get("exit_price",  0) or 0), 2), num_fmt_c)
            ws.write(current_row, 8,  round(ret,   2),                            ret_fmt)
            ws.write(current_row, 9,  round(float(rr.get("nifty_return", 0) or 0), 2), num_fmt_c)
            ws.write(current_row, 10, round(alpha, 2),                            alp_fmt)
            ws.write(current_row, 11, int(rr.get("n_eligible", n_total)),          rf)
            ws.write(current_row, 12, "✓ TRADED" if traded else "—",              rf)
            current_row += 1

        current_row += 1   # blank separator between windows


# ─────────────────────────────────────────────────────────────────────────────
# Sheet — Yearwise Summary (period summary + per-year breakdown)
# ─────────────────────────────────────────────────────────────────────────────

def _write_yearwise_summary(workbook, per_trade_df: pd.DataFrame, trade_phase_ranks_df: pd.DataFrame = None):
    ws = workbook.add_worksheet("Yearwise Summary")
    ws.freeze_panes(1, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
    })
    ws.merge_range(0, 0, 0, len(_SUMMARY_HEADERS) + 5,
                   "Yearwise Output & Summary — executed common-stock trades (by entry year)  ·  Quartiles: Q4 = top 25% … Q1 = below 25%",
                   title_fmt)
    ws.set_row(0, 22)

    if per_trade_df is None or per_trade_df.empty:
        ws.write(2, 0, "No trades available (all windows may have been reshuffled).",
                 _cell_fmt(workbook))
        return

    # Attach each trade's quartile (1..4, 4 = top) from the TRADE-PHASE ranking
    df_q = _attach_quartile(per_trade_df, trade_phase_ranks_df)

    # Period summary (cumulative look-backs) — with quartile breakdown
    r = _write_summary_table(workbook, ws, 2, "Performance Summary by Period",
                             _period_rows(df_q), set_widths=True, include_quartiles=True)

    # Yearwise breakdown + an 'All Years' total row — with quartile breakdown
    year_rows = _yearwise_rows(df_q)
    total = _trade_metrics(df_q)
    year_rows_full = year_rows + [("All Years", total)]
    _write_summary_table(workbook, ws, r + 1, "Yearwise Breakdown",
                         year_rows_full, set_widths=False, include_quartiles=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE C — Investment Analysis sheets
# ─────────────────────────────────────────────────────────────────────────────

_INR_FMT = '"\u20b9"#,##,##0'          # Indian digit grouping with rupee sign


def _inr_fmt(wb, color=None, bold=False, bg=None):
    spec = {"num_format": _INR_FMT, "font_name": "Calibri", "font_size": 9,
            "border": 1, "border_color": _BORDER, "align": "right"}
    if color:
        spec["font_color"] = color
    if bold:
        spec["bold"] = True
    if bg:
        spec["bg_color"] = bg
    return wb.add_format(spec)


def _pct2_fmt(wb, color=None, bold=False, bg=None):
    spec = {"num_format": '0.00"%"', "font_name": "Calibri", "font_size": 9,
            "border": 1, "border_color": _BORDER, "align": "right"}
    if color:
        spec["font_color"] = color
    if bold:
        spec["bold"] = True
    if bg:
        spec["bg_color"] = bg
    return wb.add_format(spec)


def _write_portfolio_stats(workbook, ia: dict | None):
    """Sheet: Portfolio Stats — all B3 metrics + an equity-curve LineChart."""
    ws = workbook.add_worksheet("Portfolio Stats")
    ws.freeze_panes(1, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG, "align": "left", "valign": "vcenter",
    })
    ws.merge_range(0, 0, 0, 1, "Portfolio Statistics — ₹ Investment Analysis", title_fmt)
    ws.set_row(0, 22)
    ws.set_column(0, 0, 30)
    ws.set_column(1, 1, 22)

    if ia is None:
        ws.write(2, 0, "Not enough trade windows to compute investment analysis.",
                 _cell_fmt(workbook))
        return

    m = ia["metrics"]
    cap = ia["initial_capital"]

    hdr = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 10,
    })
    label_fmt = _cell_fmt(workbook, bg=_BLUE_LIGHT, bold=True)
    label_hi  = _cell_fmt(workbook, bg=_BLUE_MID, bold=True)   # CAGR/Calmar/Sharpe rows
    txt_fmt   = _cell_fmt(workbook, align="right")
    green_txt = _cell_fmt(workbook, color=_GREEN_DARK, bold=True, align="right")
    red_txt   = _cell_fmt(workbook, color=_RED_DARK, bold=True, align="right")

    r = 2
    ws.write(r, 0, "Metric", hdr); ws.write(r, 1, "Value", hdr); r += 1

    def _sgn_color(v):
        return _GREEN_DARK if (v is not None and not (isinstance(v, float) and np.isnan(v)) and v >= 0) else _RED_DARK

    def row_inr(label, v, lbl_fmt=label_fmt):
        nonlocal r
        ws.write(r, 0, label, lbl_fmt)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ws.write(r, 1, "N/A", txt_fmt)
        else:
            ws.write_number(r, 1, float(v), _inr_fmt(workbook, color=_sgn_color(v), bold=True))
        r += 1

    def row_pct(label, v, signed_color=True, lbl_fmt=label_fmt):
        nonlocal r
        ws.write(r, 0, label, lbl_fmt)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ws.write(r, 1, "N/A", txt_fmt)
        else:
            col = _sgn_color(v) if signed_color else None
            ws.write_number(r, 1, float(v), _pct2_fmt(workbook, color=col, bold=True))
        r += 1

    def row_num(label, v, fmt_txt=None, lbl_fmt=label_fmt):
        nonlocal r
        ws.write(r, 0, label, lbl_fmt)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ws.write(r, 1, "N/A", txt_fmt)
        elif fmt_txt is not None:
            ws.write(r, 1, fmt_txt, txt_fmt)
        else:
            ws.write_number(r, 1, float(v), _num_fmt(workbook))
        r += 1

    row_inr("Initial Capital", cap)
    row_inr("Final Equity", m["final_equity"])
    row_inr("Total Profit / Loss", m["total_pl"])
    row_pct("Total Return %", m["total_pl_pct"])
    row_pct("CAGR", m["cagr"], lbl_fmt=label_hi)
    row_pct("Max Drawdown", m["mdd_pct"], lbl_fmt=label_fmt)
    # Calmar (light blue)
    ws.write(r, 0, "CAR / MDD (Calmar)", label_hi)
    if m["calmar"] is None or np.isnan(m["calmar"]):
        ws.write(r, 1, "N/A", txt_fmt)
    else:
        ws.write_number(r, 1, float(m["calmar"]), _num_fmt(workbook)); 
    r += 1
    # Sharpe (light blue)
    ws.write(r, 0, "Sharpe Ratio", label_hi)
    if m["sharpe"] is None or np.isnan(m["sharpe"]):
        ws.write(r, 1, "N/A", txt_fmt)
    else:
        ws.write_number(r, 1, float(m["sharpe"]), _num_fmt(workbook))
    r += 1
    row_pct("Win Rate", m["win_rate"], signed_color=False)
    row_pct("Alpha Win Rate", m["alpha_win_rate"], signed_color=False)
    row_num("Total Trades", m["n_trades"])
    row_inr("Avg Profit / Trade", m["avg_profit_trade"])
    bw, ww = m["best_window"], m["worst_window"]
    row_num("Best Window",
            None,
            fmt_txt=(f"₹{bw['profit']:,.0f}  (Window {bw['window']}"
                     + (f", {_fmt_date(bw['date'])}" if bw and bw.get('date') is not None else "") + ")")
            if bw else "—")
    row_num("Worst Window",
            None,
            fmt_txt=(f"₹{ww['profit']:,.0f}  (Window {ww['window']}"
                     + (f", {_fmt_date(ww['date'])}" if ww and ww.get('date') is not None else "") + ")")
            if ww else "—")
    avg_days = m["avg_holding_days"]
    row_num("Avg Holding Days", None,
            fmt_txt=(f"{avg_days:.0f} days" if avg_days is not None and not np.isnan(avg_days) else "N/A"))

    # ── Equity-curve data (written to the right) + LineChart ──────────────────
    eq = ia["equity_curve"].dropna(subset=["equity_inr"]).reset_index(drop=True)
    data_col = 4   # column E
    dr = 2
    ws.write(dr, data_col, "Step", hdr)
    ws.write(dr, data_col + 1, "Equity (₹)", hdr)
    dr += 1
    first_data = dr
    for i, v in enumerate(eq["equity_inr"].tolist()):
        ws.write_number(dr, data_col, i + 1, _num_fmt(workbook))
        ws.write_number(dr, data_col + 1, float(v), _inr_fmt(workbook))
        dr += 1
    last_data = dr - 1
    ws.set_column(data_col, data_col + 1, 14)

    if last_data >= first_data:
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({
            "name": "Equity (₹)",
            "categories": ["Portfolio Stats", first_data, data_col, last_data, data_col],
            "values":     ["Portfolio Stats", first_data, data_col + 1, last_data, data_col + 1],
            "line": {"color": "#00C896", "width": 2.25},
        })
        chart.set_title({"name": "Equity Curve"})
        chart.set_x_axis({"name": "Window Step"})
        chart.set_y_axis({"name": "Equity (₹)"})
        chart.set_legend({"none": True})
        chart.set_size({"width": 25 * 38, "height": 14 * 28})
        # place chart two rows below the metrics table
        ws.insert_chart(r + 2, 0, chart)


def _write_yearwise_analysis(workbook, ia: dict | None):
    """Sheet: Year-wise Analysis — B4 table + combo BarChart/LineChart."""
    ws = workbook.add_worksheet("Year-wise Analysis")
    ws.freeze_panes(2, 0)

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG, "align": "left", "valign": "vcenter",
    })

    headers = ["Year", "Windows", "Trades", "Invested (₹)", "Profit (₹)",
               "Return %", "Avg Alpha", "Win Rate", "Cumulative Equity (₹)"]
    ws.merge_range(0, 0, 0, len(headers) - 1, "Year-wise Investment Performance", title_fmt)
    ws.set_row(0, 22)

    if ia is None or ia["yearwise"] is None or ia["yearwise"].empty:
        ws.write(2, 0, "No year-wise data available.", _cell_fmt(workbook))
        return

    yw = ia["yearwise"].copy()

    hdr = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 9,
    })
    base   = _cell_fmt(workbook)
    label  = _cell_fmt(workbook, bg=_BLUE_LIGHT, bold=True)
    intf   = workbook.add_format({"num_format": "0", "font_name": "Calibri", "font_size": 9,
                                  "border": 1, "border_color": _BORDER, "align": "right"})
    green_inr = _inr_fmt(workbook, color=_GREEN_DARK, bold=True)
    red_inr   = _inr_fmt(workbook, color=_RED_DARK, bold=True)
    inr       = _inr_fmt(workbook)

    widths = [10, 9, 8, 16, 16, 11, 11, 10, 20]
    for c, (h, w) in enumerate(zip(headers, widths)):
        ws.write(1, c, h, hdr)
        ws.set_column(c, c, w)

    r = 2
    first_data = r
    total_mask = yw["Year"].astype(str) == "TOTAL"
    for _, row in yw.iterrows():
        is_total = str(row["Year"]) == "TOTAL"
        lf = _cell_fmt(workbook, bg=_BLUE_MID, bold=True) if is_total else label
        ws.write(r, 0, str(row["Year"]), lf)
        ws.write_number(r, 1, int(row["Windows"]), intf)
        ws.write_number(r, 2, int(row["Trades"]), intf)
        ws.write_number(r, 3, float(row["Invested"]), inr)
        pf = green_inr if float(row["Profit"]) >= 0 else red_inr
        ws.write_number(r, 4, float(row["Profit"]), pf)
        ws.write_number(r, 5, float(row["Return%"]) if pd.notna(row["Return%"]) else 0.0,
                        _pct2_fmt(workbook))
        if pd.notna(row["Avg Alpha"]):
            ws.write_number(r, 6, float(row["Avg Alpha"]), _num_fmt(workbook))
        else:
            ws.write(r, 6, "—", base)
        ws.write_number(r, 7, float(row["Win Rate"]) if pd.notna(row["Win Rate"]) else 0.0,
                        _pct2_fmt(workbook))
        ws.write_number(r, 8, float(row["Cumulative Equity"]), inr)
        r += 1
    last_all = r - 1

    # exclude TOTAL row from chart
    n_years = int((~total_mask).sum())
    if n_years >= 1:
        last_year_row = first_data + n_years - 1
        col_chart = workbook.add_chart({"type": "column"})
        col_chart.add_series({
            "name": "Annual Profit (₹)",
            "categories": ["Year-wise Analysis", first_data, 0, last_year_row, 0],
            "values":     ["Year-wise Analysis", first_data, 4, last_year_row, 4],
            "fill": {"color": "#00C896"},
        })
        line_chart = workbook.add_chart({"type": "line"})
        line_chart.add_series({
            "name": "Cumulative Equity (₹)",
            "categories": ["Year-wise Analysis", first_data, 0, last_year_row, 0],
            "values":     ["Year-wise Analysis", first_data, 8, last_year_row, 8],
            "line": {"color": "#FFB300", "width": 2.25},
            "y2_axis": True,
        })
        col_chart.combine(line_chart)
        col_chart.set_title({"name": "Year-wise Performance"})
        col_chart.set_x_axis({"name": "Year"})
        col_chart.set_y_axis({"name": "Annual Profit (₹)"})
        line_chart.set_y2_axis({"name": "Cumulative Equity (₹)"})
        col_chart.set_size({"width": 28 * 38, "height": 16 * 28})
        ws.insert_chart(last_all + 2, 0, col_chart)


def _write_trade_pnl(workbook, ia: dict | None):
    """Sheet: Trade P&L — enriched per-trade rows, sorted, filtered, ₹-coloured."""
    ws = workbook.add_worksheet("Trade P&L")

    title_fmt = workbook.add_format({
        "bold": True, "font_size": 13, "font_name": "Calibri",
        "font_color": _HEADER_FG, "bg_color": _HEADER_BG, "align": "left", "valign": "vcenter",
    })

    cols = [
        ("Window", "window", 8),
        ("Ticker", "ticker", 14),
        ("Entry Date", "entry_date", 13),
        ("Exit Date", "exit_date", 13),
        ("Entry Price", "entry_price", 12),
        ("Exit Price", "exit_price", 12),
        ("Return %", "return_pct", 10),
        ("Alpha %", "alpha", 10),
        ("Days Held", "days_held", 10),
        ("Allocated Capital (₹)", "allocated_capital", 18),
        ("Profit (₹)", "profit_inr", 14),
        ("Cumulative Equity (₹)", "equity_inr", 20),
    ]
    ncols = len(cols)
    ws.merge_range(0, 0, 0, ncols - 1, "Trade-level P&L (₹ Investment Analysis)", title_fmt)
    ws.set_row(0, 22)

    if ia is None or ia["per_trade_enriched"] is None or ia["per_trade_enriched"].empty:
        ws.write(2, 0, "No trade data available.", _cell_fmt(workbook))
        return

    df = ia["per_trade_enriched"].copy()
    if "window" not in df.columns and "window_idx" in df.columns:
        df["window"] = df["window_idx"].astype(int) + 1
    if "entry_date" in df.columns:
        df = df.sort_values("entry_date").reset_index(drop=True)

    hdr = workbook.add_format({
        "bold": True, "font_color": _HEADER_FG, "bg_color": _HEADER_BG,
        "border": 1, "border_color": _BORDER, "align": "center",
        "valign": "vcenter", "font_name": "Calibri", "font_size": 9,
    })
    base   = _cell_fmt(workbook)
    intf   = workbook.add_format({"num_format": "0", "font_name": "Calibri", "font_size": 9,
                                  "border": 1, "border_color": _BORDER, "align": "right"})
    numf   = _num_fmt(workbook)
    pctf   = _pct2_fmt(workbook)
    inr    = _inr_fmt(workbook)
    g_inr  = _inr_fmt(workbook, color=_GREEN_DARK, bold=True)
    r_inr  = _inr_fmt(workbook, color=_RED_DARK, bold=True)
    datef  = _date_fmt(workbook)

    for c, (h, _key, w) in enumerate(cols):
        ws.write(1, c, h, hdr)
        ws.set_column(c, c, w)

    r = 2
    for _, row in df.iterrows():
        for c, (h, key, w) in enumerate(cols):
            v = row.get(key)
            if key == "window":
                ws.write_number(r, c, int(v) if pd.notna(v) else 0, intf)
            elif key in ("entry_date", "exit_date"):
                if pd.notna(v):
                    try:
                        ws.write_datetime(r, c, pd.Timestamp(v).to_pydatetime(), datef)
                    except Exception:
                        ws.write(r, c, _fmt_date(v), base)
                else:
                    ws.write(r, c, "", base)
            elif key in ("return_pct", "alpha"):
                ws.write_number(r, c, float(v) if pd.notna(v) else 0.0, pctf)
            elif key == "days_held":
                ws.write_number(r, c, int(v) if pd.notna(v) else 0, intf)
            elif key in ("entry_price", "exit_price"):
                ws.write_number(r, c, float(v) if pd.notna(v) else 0.0, numf)
            elif key in ("allocated_capital", "equity_inr"):
                ws.write_number(r, c, float(v) if pd.notna(v) else 0.0, inr)
            elif key == "profit_inr":
                pv = float(v) if pd.notna(v) else 0.0
                ws.write_number(r, c, pv, g_inr if pv >= 0 else r_inr)
            else:
                ws.write(r, c, "" if pd.isna(v) else str(v), base)
        r += 1

    ws.freeze_panes(2, 0)
    ws.autofilter(1, 0, r - 1, ncols - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_excel(
    per_trade_df : pd.DataFrame,
    summary_df   : pd.DataFrame,
    phases       : pd.DataFrame,
    returns_df   : pd.DataFrame,
    nifty_df     : pd.DataFrame,
    windows      : list[dict],
    freq_df      : pd.DataFrame,
    config       : dict,
    top_n        : int = 20,
    sort_by      : str = "alpha",
    window_status_df     : pd.DataFrame = None,
    candidates_df        : pd.DataFrame = None,
    window_ranks_df      : pd.DataFrame = None,
    trade_phase_ranks_df : pd.DataFrame = None,
    initial_capital      : float = 100_000.0,
    alloc_mode           : str = "equal",
    reinvest             : bool = False,
) -> bytes:
    """
    Generate and return the full Excel workbook as bytes.
    """
    if window_status_df is None:
        window_status_df = pd.DataFrame()
    if candidates_df is None:
        candidates_df = pd.DataFrame()
    if window_ranks_df is None:
        window_ranks_df = pd.DataFrame()
    if trade_phase_ranks_df is None:
        trade_phase_ranks_df = pd.DataFrame()

    buf = io.BytesIO()
    workbook = None

    try:
        import xlsxwriter
        workbook = xlsxwriter.Workbook(buf, {"in_memory": True, "remove_timezone": True})

        # Feature C — investment analysis (computed from the same per_trade_df)
        try:
            _ia = compute_investment_analysis(
                per_trade_df, initial_capital=float(initial_capital),
                alloc_mode=alloc_mode, reinvest=bool(reinvest),
            )
        except Exception:
            _ia = None

        _write_portfolio_summary(workbook, per_trade_df, summary_df, config, window_status_df)
        _write_portfolio_stats(workbook, _ia)
        _write_yearwise_analysis(workbook, _ia)
        _write_trade_pnl(workbook, _ia)
        _write_window_status(workbook, window_status_df)
        _write_window_candidates(workbook, candidates_df, window_status_df, per_trade_df)
        _write_window_stock_ranks(workbook, window_ranks_df, window_status_df)
        _write_trade_phase_ranks(workbook, trade_phase_ranks_df)
        _write_yearwise_summary(workbook, per_trade_df, trade_phase_ranks_df)
        _write_common_pnl(workbook, per_trade_df, trade_phase_ranks_df)
        _write_trade_log(workbook, per_trade_df)
        _write_stock_summary(workbook, summary_df)
        _write_nifty_entry_vis(workbook, per_trade_df, nifty_df)
        _write_window_analysis(workbook, windows, freq_df, returns_df, top_n, sort_by,
                               window_status_df=window_status_df)
        _write_phase_schedule(workbook, phases)

        workbook.close()
    except Exception as e:
        if workbook:
            try:
                workbook.close()
            except Exception:
                pass
        raise e

    return buf.getvalue()
