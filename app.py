"""
╔══════════════════════════════════════════════════════════════════╗
║     S³ — Multi-Leg Stock Selection System                       ║
║     NIFTY Threshold Entry/Exit + Window-Based Stock Selection    ║
╚══════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
import os
import warnings
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from core.data_loader    import load_phases, load_nifty, load_all_stocks, validate_files
from core.phase_engine   import compute_all_phase_returns, get_top_n_per_phase, quartile_label
from core.multi_leg_engine import (
    find_pattern_windows,
    compute_stock_frequency,
    compute_persistence_score,
    compute_nifty_threshold_trades,
    compute_window_stock_ranks,
    compute_trade_phase_ranks,
    entry_segment_topn_df,
)
from export.excel_exporter import generate_excel
from core.investment_analysis import (
    compute_investment_analysis, fmt_inr, fmt_inr_full,
)
from core.ml_model import run_ml


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="S³ Multi-Leg System",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A0A18 0%, #0D0D22 100%);
    border-right: 1px solid #1E1E3A;
}
button[data-baseweb="tab"] {
    font-size: 0.84rem; font-weight: 600; color: #8B9DB5; padding: 8px 18px;
}
button[data-baseweb="tab"]:hover { color: #A89FFF !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom: 3px solid #6C63FF !important;
    background: rgba(108,99,255,0.06) !important;
}
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(108,99,255,0.09) 0%, rgba(0,200,150,0.05) 100%);
    border: 1px solid rgba(108,99,255,0.22);
    border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 0.76rem; color: #8B9DB5; font-weight: 600;
    letter-spacing: 0.04em; text-transform: uppercase;
}
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 800; color: #E8E4FF; }
.sec {
    font-size: 1rem; font-weight: 700; color: #C9B7FF;
    border-left: 4px solid #6C63FF; padding: 4px 12px;
    margin: 18px 0 10px; background: rgba(108,99,255,0.04); border-radius: 0 6px 6px 0;
}
.card {
    background: rgba(108,99,255,0.06); border: 1px solid rgba(108,99,255,0.18);
    border-radius: 10px; padding: 14px 18px; margin: 6px 0; line-height: 1.7;
}
.logo-text {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #6C63FF 0%, #00C896 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.logo-sub { font-size: 0.78rem; color: #8B9DB5; margin-top: -4px; letter-spacing: 0.03em; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v
    return st.session_state[k]

def _fmt(v, decimals=2, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:+.{decimals}f}{suffix}" if suffix == "%" else f"{v:.{decimals}f}"

def _color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "gray"
    return "normal" if v >= 0 else "inverse"

def _section(txt):
    st.markdown(f'<div class="sec">{txt}</div>', unsafe_allow_html=True)

def _card(txt):
    st.markdown(f'<div class="card">{txt}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="logo-text">S³</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Multi-Leg Stock Selection System</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📁 Upload Data Files")
    dates_file = st.file_uploader(
        "Dates.xlsx  (Market Phases + NIFTY)",
        type=["xlsx"], key="dates_file",
        help="Sheet 1: Phase schedule (Trade, Entry Date, Exit Date). Sheet 2 'NF': NIFTY daily closes."
    )
    data_file = st.file_uploader(
        "Data.xlsx  (Stock Universe)",
        type=["xlsx"], key="data_file",
        help="One sheet per stock. Columns: Date, Close, Aux2."
    )
    st.divider()

    if dates_file and data_file:
        st.markdown("#### ⚙️ Multi-Leg Configuration")

        # ── Pattern / Legs ────────────────────────────────────────────────────
        st.caption("**Pattern Settings**")
        start_phase = st.radio("Starting Phase", ["Rise", "Fall"],
                               help="First leg of the pattern.", horizontal=True)
        leg_count = st.slider("Number of Legs", 1, 6, 2,
                              help="How many phases in the lookup window (e.g. 2 = Rise-Fall).")

        pattern = []
        cur = start_phase
        for _ in range(leg_count):
            pattern.append(cur)
            cur = "Fall" if cur == "Rise" else "Rise"

        st.markdown(f"**Pattern:** `{'  →  '.join(pattern)}`")
        st.divider()

        # ── Stock Selection ───────────────────────────────────────────────────
        st.caption("**Stock Selection**")
        top_n = st.select_slider("Top N Stocks per Leg", [10, 20, 30, 50, 75, 100], value=20,
                                  help="Top-N stocks by alpha/return per phase leg.")
        top_k_common = st.number_input("Common Stocks to Trade", 1, 50, 10,
                                        help="Final portfolio size — top-K common stocks.")
        sort_by = st.radio("Rank By", ["alpha", "return_pct"], horizontal=True,
                           help="Metric used to rank stocks within each leg.")
        persistence_on = st.checkbox("Enable Persistence Score", value=True,
                                     help="Use persistence-based ranking for final stock selection.")
        include_entry_seg = st.checkbox("Include Entry-Segment Leg", value=True,
                                        help="ON → common stocks must also appear in the entry-segment "
                                             "top-N (Fall exit / next-Rise entry → NIFTY entry-trigger). "
                                             "OFF → use only the pattern legs.")

        # ── Low-Volatility Filter (optional) ──────────────────────────────────
        lowvol_on = st.checkbox("Enable Low-Volatility Filter", value=False,
                                help="Drop the most volatile stocks (by daily-return std across all "
                                     "loaded days) BEFORE top-N selection. OFF → no change to behaviour.")
        lowvol_pct = 50
        if lowvol_on:
            lowvol_pct = st.slider("Max Volatility Percentile", 10, 90, 50, 10,
                                   help="Keep only stocks at or below this percentile of daily volatility. "
                                        "e.g. 50 → keep the calmer bottom half of the universe.")

        # ── Window Volatility Filter (per-window, pattern-start → buy-date) ───
        st.caption("**Window Volatility Filter**")
        winvol_mode_label = st.radio(
            "Select stocks by window volatility",
            ["Off", "Low volatility (calmest)", "High volatility (wildest)"],
            index=0, horizontal=False,
            help="Volatility measured per window over the span from the FIRST pattern leg's "
                 "entry up to the NIFTY buy-trigger date. Low keeps the calmest stocks, "
                 "High keeps the most volatile. Acts as a gate before the final Top-K; alpha still ranks.")
        winvol_mode = {"Off": "off",
                       "Low volatility (calmest)": "low",
                       "High volatility (wildest)": "high"}[winvol_mode_label]
        winvol_pct = 50
        if winvol_mode != "off":
            winvol_pct = st.slider(
                "Window Volatility Percentile", 10, 90, 50, 10,
                help="Low mode keeps the bottom P% by window volatility; "
                     "High mode keeps the top P%.")
        st.divider()

        # ── NIFTY Thresholds ──────────────────────────────────────────────────
        st.caption("**NIFTY Entry / Exit Thresholds**")
        entry_thr = st.slider("Entry Threshold % (NIFTY Rise → BUY)",
                              0.0, 15.0, 3.0, 0.5,
                              help="BUY when NIFTY rises this much from buy-phase start.")
        exit_thr  = st.slider("Exit Threshold % (NIFTY Fall → SELL)",
                              0.0, 15.0, 2.0, 0.5,
                              help="SELL when NIFTY falls this much from sell-phase start.")
        st.divider()

        # ── Reshuffle Threshold (optional) ────────────────────────────────────
        st.caption("**Reshuffle Threshold (optional)**")
        reshuffle_on = st.checkbox("Enable Reshuffle Threshold", value=False,
                                   help="Skip ('reshuffle') any window whose Fall leg drops more "
                                        "than the threshold. A reason is recorded for each skipped window.")
        reshuffle_thr = st.slider("Reshuffle if NIFTY Fall exceeds %",
                                  1.0, 30.0, 10.0, 0.5,
                                  disabled=not reshuffle_on,
                                  help="e.g. 10% → if NIFTY falls more than 10% (e.g. -11%) during the "
                                       "Fall leg, the window is reshuffled and the next pattern is checked.")
        if reshuffle_on:
            st.markdown(f"<small>Windows with a Fall-leg drawdown worse than "
                        f"<b>-{reshuffle_thr:.1f}%</b> will be skipped and the reason shown.</small>",
                        unsafe_allow_html=True)
        st.divider()

        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# NO FILES — welcome screen
# ══════════════════════════════════════════════════════════════════════════════

if not dates_file or not data_file:
    st.markdown('<div class="logo-text" style="text-align:center;padding:40px 0 10px">S³</div>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#8B9DB5'>Multi-Leg Stock Selection System</h3>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width:680px;margin:0 auto;text-align:left'>
    <div class='card'>
    <b>How It Works</b><br><br>
    1. Upload <code>Dates.xlsx</code> (phase schedule + NIFTY daily closes)<br>
    2. Upload <code>Data.xlsx</code> (stock universe — one sheet per ticker)<br>
    3. Configure the multi-leg pattern in the sidebar<br>
    4. Set NIFTY entry/exit thresholds<br>
    5. Click <b>Run Analysis</b>
    </div>
    <div class='card' style='margin-top:12px'>
    <b>NIFTY Threshold Logic</b><br><br>
    • <b>Entry:</b> After the pattern window, wait until NIFTY rises by N% from phase start → BUY<br>
    • <b>Exit:</b> During the following sell phase, wait until NIFTY falls by M% → SELL<br>
    • If threshold never hit → fallback to phase start (entry) or phase end (exit)
    </div>
    <div class='card' style='margin-top:12px'>
    <b>New: Reshuffle Threshold &amp; Three-Leg Selection</b><br><br>
    • <b>Reshuffle (optional):</b> if NIFTY falls more than your chosen % (e.g. -10%) during a
      Fall leg, that window is skipped ('reshuffled') and the reason is recorded.<br>
    • <b>Three-leg common stocks:</b> trading candidates are the common stocks across the
      pattern legs <i>and</i> the entry segment (Fall exit → NIFTY entry-trigger), computed
      <b>per window</b> — so each window has its own stock set.
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════


# ── Cache file bytes in session_state to avoid re-reading on every rerun ──────
def _get_bytes(key, uploader):
    if key not in st.session_state or st.session_state.get(f"{key}_name") != uploader.name:
        st.session_state[key] = uploader.read()
        st.session_state[f"{key}_name"] = uploader.name
    return st.session_state[key]

dates_bytes = _get_bytes("dates_bytes", dates_file)
data_bytes  = _get_bytes("data_bytes",  data_file)

with st.spinner("🔍 Validating files..."):
    val = validate_files(dates_bytes, data_bytes)

if val["errors"]:
    for e in val["errors"]:
        st.error(f"❌ {e}")
    st.stop()

with st.spinner("📅 Loading phase schedule..."):
    phases = load_phases(dates_bytes)
with st.spinner("📈 Loading NIFTY data..."):
    nifty_df = load_nifty(dates_bytes)
with st.spinner("🏢 Loading stock universe (cached after first load)..."):
    stock_dict = load_all_stocks(data_bytes)

# ── Data validation banner ────────────────────────────────────────────────────
st.markdown("""
<div style='background:rgba(108,99,255,0.06);border:1px solid rgba(108,99,255,0.2);
     border-radius:10px;padding:10px 16px;margin-bottom:8px'>
<span style='font-size:0.75rem;color:#8B9DB5;font-weight:600;letter-spacing:0.05em'>
✅ DATA LOADED SUCCESSFULLY
</span>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Total Phases",       val["n_phases"],   help="Number of Rise/Fall market phases loaded from Dates.xlsx")
c2.metric("📈 NIFTY Data Rows",    val["nifty_rows"], help="Daily NIFTY close prices available for analysis")
c3.metric("🏢 Stocks in Universe", val["n_stocks"],   help="Number of individual stock sheets loaded from Data.xlsx")
c4.metric("📊 Date Range", val.get("date_range", "—").replace(" → ", " → ") if val.get("date_range") else "—",
          help="Full date range covered by the phase schedule")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE PHASE RETURNS (cached)
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Computing phase returns..."):
    # ── Feature A: optional low-volatility filter (runs BEFORE top-N) ─────────
    _excluded_lowvol = ()
    if 'lowvol_on' in dir() and lowvol_on:
        _vols = {}
        for _tkr, _sdf in stock_dict.items():
            try:
                _dr = _sdf["close"].pct_change().dropna()
            except Exception:
                continue
            if len(_dr) > 1:
                _vols[_tkr] = float(_dr.std())
        if _vols:
            _thr = float(np.percentile(list(_vols.values()), lowvol_pct))
            _excluded_lowvol = tuple(t for t, v in _vols.items() if v > _thr)
        _n_before = len(stock_dict)
        _n_after = _n_before - len(_excluded_lowvol)
        st.info(f"Low-vol filter: universe reduced from {_n_before} → {_n_after} stocks "
                f"(keeping bottom {lowvol_pct}-th percentile by daily volatility)")

    returns_df = compute_all_phase_returns(
        stock_dict, phases, nifty_df, excluded_tickers=_excluded_lowvol
    )

if returns_df.empty:
    st.error("No phase returns could be computed. Check your data files.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_windows, tab_trades, tab_vis, tab_export, tab_ml = st.tabs([
    "📋 Overview",
    "🔍 Window Analysis",
    "📊 Trade Results",
    "📈 NIFTY Entry Visualisation",
    "⬇️ Export",
    "🤖 ML Portfolio",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    _section("Phase Schedule")

    phase_disp = phases.copy()
    phase_disp["entry_date"] = phase_disp["entry_date"].dt.strftime("%d-%b-%Y")
    phase_disp["exit_date"]  = phase_disp["exit_date"].dt.strftime("%d-%b-%Y")

    def _highlight_trade(s):
        return [
            "background-color:#C6EFCE;color:#1A6B3C" if v == "Rise"
            else "background-color:#FFC7CE;color:#9C0006" if v == "Fall"
            else ""
            for v in s
        ]

    st.dataframe(
        phase_disp.style.apply(_highlight_trade, subset=["trade"]),
        use_container_width=True, height=300
    )

    _section("Phase Distribution")
    rise_count = (phases["trade"] == "Rise").sum()
    fall_count = (phases["trade"] == "Fall").sum()
    fig_pie = go.Figure(go.Pie(
        labels=["Rise", "Fall"],
        values=[rise_count, fall_count],
        hole=0.45,
        marker=dict(colors=["#00C896", "#FF4B6E"]),
        textfont_size=14,
    ))
    fig_pie.update_layout(
        height=280, margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#C9C9C9",
        legend=dict(font=dict(color="#C9C9C9")),
    )
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.plotly_chart(fig_pie, use_container_width=True)
        m1, m2 = st.columns(2)
        m1.metric("Rise Phases", rise_count)
        m2.metric("Fall Phases", fall_count)

    with col_r:
        _section("NIFTY Phase Returns")
        nifty_phase = returns_df[["phase_id", "trade", "nifty_return"]].drop_duplicates("phase_id").dropna(subset=["nifty_return"])
        if not nifty_phase.empty:
            fig_nifty = go.Figure()
            fig_nifty.add_trace(go.Bar(
                x=nifty_phase["phase_id"],
                y=nifty_phase["nifty_return"],
                marker_color=["#00C896" if x > 0 else "#FF4B6E" for x in nifty_phase["nifty_return"]],
                text=[f"{v:+.1f}%" for v in nifty_phase["nifty_return"]],
                textposition="outside",
                name="NIFTY Return",
            ))
            fig_nifty.update_layout(
                height=260, margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C9C9C9", xaxis_title="Phase ID", yaxis_title="Return %",
                xaxis=dict(color="#8B9DB5"), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
            )
            st.plotly_chart(fig_nifty, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RUN MULTI-LEG ANALYSIS (triggered by button OR when results exist in session)
# ══════════════════════════════════════════════════════════════════════════════

run_btn_local = "run_btn" in locals() and run_btn

if run_btn_local or "last_results" in st.session_state:
    if run_btn_local:
        with st.spinner("Running multi-leg analysis..."):
            # ── Find pattern windows ──────────────────────────────────────────
            all_windows = find_pattern_windows(phases, pattern)

            if not all_windows:
                st.error(f"No occurrences of pattern **{'→'.join(pattern)}** found in the phase data.")
                st.stop()

            # ── Stock frequency across windows ────────────────────────────────
            freq_df = compute_stock_frequency(returns_df, all_windows, top_n, sort_by)

            # ── Persistence score ─────────────────────────────────────────────
            persist_df = pd.DataFrame()
            if persistence_on and not freq_df.empty:
                persist_df = compute_persistence_score(freq_df, all_windows, returns_df, sort_by, top_n)

            # ── NIFTY-threshold trades ────────────────────────────────────────
            # Use freq_df or persist_df for ordering
            selection_df = persist_df if persistence_on and not persist_df.empty else freq_df

            # Re-order freq_df to match persist order for top_k selection
            if persistence_on and not persist_df.empty:
                # Merge persistence rank into freq_df order
                freq_ordered = (
                    persist_df[["ticker"]].merge(freq_df, on="ticker", how="left")
                    .reset_index(drop=True)
                )
                freq_ordered["freq_rank"] = range(1, len(freq_ordered) + 1)
            else:
                freq_ordered = freq_df

            per_trade_df, summary_df, window_status_df, candidates_df = compute_nifty_threshold_trades(
                nifty_df=nifty_df,
                stock_dict=stock_dict,
                windows=all_windows,
                freq_df=freq_ordered,
                top_k_common=top_k_common,
                entry_threshold_pct=entry_thr,
                exit_threshold_pct=exit_thr,
                returns_df=returns_df,
                pattern=pattern,
                top_n=top_n,
                sort_by=sort_by,
                reshuffle_enabled=reshuffle_on,
                reshuffle_threshold=reshuffle_thr,
                include_entry_segment=include_entry_seg,
                vol_mode=winvol_mode,
                vol_pct=winvol_pct,
            )

            # Full per-window ranking of all considered stocks (valid in every leg)
            window_ranks_df = compute_window_stock_ranks(
                windows=all_windows, returns_df=returns_df, pattern=pattern,
                sort_by=sort_by, candidates_df=candidates_df, window_status_df=window_status_df,
            )

            # ── Trade-phase ranking: ALL stocks ranked by actual buy→sell return ──
            trade_phase_ranks_df = compute_trade_phase_ranks(
                windows=all_windows,
                per_trade_df=per_trade_df,
                stock_dict=stock_dict,
                nifty_df=nifty_df,
            )

        st.session_state["last_results"] = {
            "all_windows"  : all_windows,
            "freq_df"      : freq_df,
            "freq_ordered" : freq_ordered,
            "persist_df"   : persist_df,
            "per_trade_df" : per_trade_df,
            "summary_df"   : summary_df,
            "window_status_df": window_status_df,
            "candidates_df": candidates_df,
            "window_ranks_df": window_ranks_df,
            "trade_phase_ranks_df": trade_phase_ranks_df,
            "pattern"      : pattern,
            "top_n"        : top_n,
            "top_k_common" : top_k_common,
            "entry_thr"    : entry_thr,
            "exit_thr"     : exit_thr,
            "sort_by"      : sort_by,
            "persistence_on": persistence_on,
            "reshuffle_on" : reshuffle_on,
            "reshuffle_thr": reshuffle_thr,
            "include_entry_seg": include_entry_seg,
            "winvol_mode"  : winvol_mode,
            "winvol_pct"   : winvol_pct,
            "lowvol_on"    : bool(lowvol_on),
            "lowvol_pct"   : int(lowvol_pct),
        }

    res           = st.session_state["last_results"]
    all_windows   = res["all_windows"]
    freq_df       = res["freq_df"]
    freq_ordered  = res["freq_ordered"]
    persist_df    = res["persist_df"]
    per_trade_df  = res["per_trade_df"]
    summary_df    = res["summary_df"]
    window_status_df = res.get("window_status_df", pd.DataFrame())
    candidates_df = res.get("candidates_df", pd.DataFrame())
    window_ranks_df = res.get("window_ranks_df", pd.DataFrame())
    trade_phase_ranks_df = res.get("trade_phase_ranks_df", pd.DataFrame())
    _pattern      = res["pattern"]
    _top_n        = res["top_n"]
    _top_k        = res["top_k_common"]
    _entry_thr    = res["entry_thr"]
    _exit_thr     = res["exit_thr"]
    _sort_by      = res["sort_by"]
    _pers_on      = res["persistence_on"]
    _reshuffle_on = res.get("reshuffle_on", False)
    _reshuffle_thr= res.get("reshuffle_thr", 10.0)
    _include_seg  = res.get("include_entry_seg", True)
    _winvol_mode  = res.get("winvol_mode", "off")
    _winvol_pct   = res.get("winvol_pct", 50)
    _lowvol_on    = res.get("lowvol_on", False)
    _lowvol_pct   = res.get("lowvol_pct", 50)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — WINDOW ANALYSIS
    # ════════════════════════════════════════════════════════════════════════

    with tab_windows:
        _section(f"Pattern: {'  →  '.join(_pattern)}  |  {len(all_windows)} Windows Found")

        # Window selector
        win_nums = [f"Window {w['window_idx']+1}  ({pd.Timestamp(w['entry_dates'][0]).strftime('%d-%b-%Y')} → {pd.Timestamp(w['exit_dates'][-1]).strftime('%d-%b-%Y')})"
                    for w in all_windows]
        sel_win_label = st.selectbox("Select Window to Inspect", win_nums, key="win_sel")
        sel_win_idx   = win_nums.index(sel_win_label)
        win           = all_windows[sel_win_idx]

        # ── Reshuffle / status banner for this window ─────────────────────────
        if not window_status_df.empty:
            wstat_row = window_status_df[window_status_df["window_idx"] == win["window_idx"]]
            if not wstat_row.empty:
                ws_r = wstat_row.iloc[0]
                if bool(ws_r.get("reshuffled")):
                    st.markdown(
                        f"<div style='background:rgba(255,75,110,0.12);border:1px solid #FF4B6E;"
                        f"border-radius:10px;padding:12px 16px;margin:6px 0'>"
                        f"<b style='color:#FF4B6E'>🔄 RESHUFFLED — this window was skipped</b><br>"
                        f"<small style='color:#E8B4C0'>{ws_r.get('reason','')}</small></div>",
                        unsafe_allow_html=True)
                else:
                    fdd = ws_r.get("fall_drawdown_pct")
                    fdd_txt = f"{fdd:.2f}%" if fdd is not None and not (isinstance(fdd,float) and np.isnan(fdd)) else "—"
                    st.markdown(
                        f"<div style='background:rgba(0,200,150,0.10);border:1px solid rgba(0,200,150,0.4);"
                        f"border-radius:10px;padding:10px 16px;margin:6px 0'>"
                        f"<b style='color:#00C896'>✓ {ws_r.get('status','Traded')}</b> — "
                        f"<small style='color:#A9C7BC'>Fall-leg drawdown: {fdd_txt} · "
                        f"Common candidates: {int(ws_r.get('n_common_candidates',0))} · "
                        f"Trades: {int(ws_r.get('n_trades',0))}</small></div>",
                        unsafe_allow_html=True)

        # Per-leg top stocks + common highlighting
        n_legs   = len(win["phase_ids"])
        leg_cols = st.columns(n_legs)
        leg_sets = []

        for li, pid in enumerate(win["phase_ids"]):
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[_sort_by])
            if ph.empty:
                leg_sets.append(set())
                with leg_cols[li]:
                    st.info("No data for this phase.")
                continue
            top = ph.nlargest(_top_n, _sort_by).reset_index(drop=True)
            leg_sets.append(set(top["ticker"].tolist()))

        common = set.intersection(*leg_sets) if leg_sets else set()

        for li, pid in enumerate(win["phase_ids"]):
            ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[_sort_by])
            if ph.empty:
                continue
            top = ph.nlargest(_top_n, _sort_by).reset_index(drop=True)
            entry_d = pd.Timestamp(win["entry_dates"][li]).strftime("%d-%b-%Y")
            exit_d  = pd.Timestamp(win["exit_dates"][li]).strftime("%d-%b-%Y")
            trade_  = top["trade"].iloc[0] if "trade" in top.columns else _pattern[li]

            with leg_cols[li]:
                color = "#00C896" if trade_ == "Rise" else "#FF4B6E"
                st.markdown(f"<b style='color:{color}'>Leg {li+1}: {trade_}</b><br>"
                            f"<small>{entry_d} → {exit_d}</small>", unsafe_allow_html=True)

                disp = top[["ticker", "return_pct", "alpha"]].copy()
                _n_leg = len(disp)
                disp.insert(1, "Quartile", [quartile_label(i + 1, _n_leg) for i in range(_n_leg)])
                disp.columns = ["Ticker", "Quartile", "Ret %", "Alpha %"]
                disp["Common"] = disp["Ticker"].isin(common).map({True: "✓", False: ""})

                def _hl_common(row):
                    base = ["background-color:#C6EFCE;color:#1A6B3C"] * len(row) if row["Common"] == "✓" else [""] * len(row)
                    return base

                st.dataframe(
                    disp.style.apply(_hl_common, axis=1)
                        .format({"Ret %": "{:+.2f}", "Alpha %": "{:+.2f}"}),
                    use_container_width=True, height=400,
                )

        # ── Entry-Segment Leg — Top N (only when the toggle is ON) ────────────
        if _include_seg:
            _section(f"Entry-Segment Leg — Top {_top_n} (Fall exit / next-Rise entry → NIFTY entry-trigger)")
            _card(
                "When the entry-segment leg is enabled, these are its <b>own Top-N</b> stocks, ranked "
                "by segment alpha, with Quartile. This leg is intersected with the pattern legs to "
                "form the common stocks we actually trade. ✓ marks stocks that ended up in the final "
                "common (traded) set for this window."
            )
            try:
                seg_df, seg_meta = entry_segment_topn_df(win, nifty_df, stock_dict, _entry_thr, _top_n)
            except Exception:
                seg_df, seg_meta = pd.DataFrame(), {}

            if not seg_df.empty:
                common_traded = set()
                if not candidates_df.empty:
                    common_traded = set(
                        candidates_df[candidates_df["window_idx"] == win["window_idx"]]["ticker"].tolist()
                    )
                seg_disp = seg_df[["rank", "ticker", "quartile", "seg_return", "seg_alpha"]].copy()
                seg_disp["Common"] = seg_disp["ticker"].isin(common_traded).map({True: "✓", False: ""})
                seg_disp.columns = ["Rank", "Ticker", "Quartile", "Seg Ret %", "Seg Alpha %", "Common"]

                seg_start = pd.Timestamp(seg_meta.get("seg_start")).strftime("%d-%b-%Y") if seg_meta.get("seg_start") is not None else "—"
                seg_end   = pd.Timestamp(seg_meta.get("seg_end")).strftime("%d-%b-%Y") if seg_meta.get("seg_end") is not None else "—"
                st.caption(
                    f"Segment window: {seg_start} → {seg_end}  ·  basis: {seg_meta.get('basis','')}  ·  "
                    f"NIFTY segment return: {seg_meta.get('nifty_segment_ret', 0):+.2f}%  ·  "
                    f"{len(seg_disp)} stocks"
                )

                def _hl_seg(row):
                    return (["background-color:#C6EFCE;color:#1A6B3C"] * len(row)
                            if row["Common"] == "✓" else [""] * len(row))

                st.dataframe(
                    seg_disp.style.apply(_hl_seg, axis=1)
                            .format({"Seg Ret %": "{:+.2f}", "Seg Alpha %": "{:+.2f}"}),
                    use_container_width=True, height=400,
                )
            else:
                st.info("No entry-segment data for this window (it may be reshuffled or have no buy phase).")

        # Common stocks summary
        _section(f"Common Stocks across all {n_legs} legs — {len(common)} found")
        if common:
            # Per-leg rank maps (for quartile)
            leg_rank_maps = []
            for li, pid in enumerate(win["phase_ids"]):
                ph = returns_df[returns_df["phase_id"] == pid].dropna(subset=[_sort_by])
                top = ph.nlargest(_top_n, _sort_by).reset_index(drop=True)
                rmap = {t: (i + 1, len(top)) for i, t in enumerate(top["ticker"])}
                leg_rank_maps.append(rmap)

            common_rows = []
            for tkr in common:
                row_ = {}
                fracs = []
                for li, pid in enumerate(win["phase_ids"]):
                    ph = returns_df[returns_df["phase_id"] == pid]
                    r  = ph[ph["ticker"] == tkr]
                    if not r.empty:
                        row_[f"Leg {li+1} Alpha"] = round(float(r["alpha"].iloc[0] or 0), 2)
                        row_[f"Leg {li+1} Ret"]   = round(float(r["return_pct"].iloc[0] or 0), 2)
                    rk = leg_rank_maps[li].get(tkr)
                    if rk:
                        row_[f"Leg {li+1} Q"] = quartile_label(rk[0], rk[1])
                        fracs.append(rk[0] / rk[1])
                overall_q = quartile_label(int(round(np.mean(fracs) * 100)), 100) if fracs else "—"
                common_rows.append({"Ticker": tkr, "Quartile": overall_q, **row_})
            common_df = pd.DataFrame(common_rows)
            # Add avg alpha
            alpha_cols = [c for c in common_df.columns if "Alpha" in c]
            if alpha_cols:
                common_df["Avg Alpha"] = common_df[alpha_cols].mean(axis=1).round(2)
                common_df = common_df.sort_values("Avg Alpha", ascending=False)
            st.dataframe(common_df, use_container_width=True)
        else:
            st.info("No stocks are common across all legs in this window.")

        # ── THREE-LEG common candidates (pattern legs + entry segment) ────────
        _seg_txt = ("pattern legs <b>AND</b> the entry segment (Fall exit / next-Rise entry → "
                    "NIFTY entry-trigger)") if _include_seg else "<b>pattern legs only</b> (entry segment toggle is OFF)"
        _section("Common Candidates (used to BUY / SELL this window)")
        _card(
            "These are the stocks the engine actually trades for this window: the common stocks across "
            f"{_seg_txt}. Each carries a <b>Quartile</b> (Q1 = top 25% … Q4 = below 25%) so you can see "
            "which quartile the chosen stock came from. Candidates are computed <b>per window</b>."
        )
        if not candidates_df.empty:
            win_cand = candidates_df[candidates_df["window_idx"] == win["window_idx"]].copy()
            if not win_cand.empty:
                lead_cols = [c for c in ["common_rank", "ticker", "quartile", "mean_alpha", "mean_return",
                                          "selected_for_trade", "traded"] if c in win_cand.columns]
                q_cols    = [c for c in win_cand.columns if c.endswith("| Q")]
                leg_cols  = [c for c in win_cand.columns if c.endswith("| alpha") or c.endswith("| rank")]
                show = win_cand[lead_cols + q_cols + leg_cols].sort_values("common_rank")
                show = show.rename(columns=lambda c: c.replace("_", " ").title())

                def _hl_traded(row):
                    try:
                        if str(row.get("Traded")) == "True":
                            return ["background-color:#C6EFCE;color:#1A6B3C"] * len(row)
                    except Exception:
                        pass
                    return [""] * len(row)

                st.dataframe(show.style.apply(_hl_traded, axis=1), use_container_width=True, height=320)
            else:
                st.info("No common candidates for this window (or it was reshuffled).")
        else:
            st.info("No candidate data available.")

        # ── FULL ranked log: every stock valid across all legs of this window ──
        _section("All Considered Stocks — Full Ranking (this window)")
        _card(
            "Every stock that is valid across <b>all legs</b> of this window, ranked. "
            "A stock with entry Aux2=0 in any leg has no return there, so it drops out of "
            "the window entirely. Columns show each leg's rank &amp; <b>Quartile</b> plus the "
            "overall <b>Window Quartile</b>. Rows we are <b>buying</b> are highlighted green."
        )
        if not window_ranks_df.empty:
            win_rk = window_ranks_df[window_ranks_df["window_idx"] == win["window_idx"]].copy()
            if not win_rk.empty:
                lead = [c for c in ["window_rank", "ticker", "window_quartile", "mean_alpha", "buying"]
                        if c in win_rk.columns]
                qrank = [c for c in win_rk.columns if c.endswith("| Q") or c.endswith("| rank")]
                show_rk = win_rk[lead + qrank].sort_values("window_rank")
                show_rk = show_rk.rename(columns=lambda c: c.replace("_", " ").title())

                def _hl_buy(row):
                    try:
                        if str(row.get("Buying")) == "True":
                            return ["background-color:#C6EFCE;color:#1A6B3C;font-weight:bold"] * len(row)
                    except Exception:
                        pass
                    return [""] * len(row)

                st.caption(f"{len(win_rk)} stocks considered · "
                           f"{int(win_rk['buying'].sum())} selected to buy (highlighted)")
                st.dataframe(show_rk.style.apply(_hl_buy, axis=1), use_container_width=True, height=420)
            else:
                st.info("No considered stocks for this window (or it was reshuffled / has no buy phase).")
        else:
            st.info("No ranking data available.")

        # Frequency summary
        _section(f"Stock Frequency Across All {len(all_windows)} Windows  (reference only)")
        st.caption("ℹ️ This aggregate table is informational. Trading candidates are now chosen "
                   "**per window** from the three-leg common set above — not from this table.")
        if not freq_ordered.empty:
            disp_freq = freq_ordered.head(40)[
                [c for c in ["freq_rank", "ticker", "full_window_count", "full_window_pct",
                              "any_leg_count", "avg_alpha", "avg_return"] if c in freq_ordered.columns]
            ].copy()
            disp_freq.columns = [c.replace("_", " ").title() for c in disp_freq.columns]
            st.dataframe(disp_freq, use_container_width=True, height=350)

        # Persistence score
        if _pers_on and not persist_df.empty:
            _section("Persistence Score Ranking")
            disp_p = persist_df.head(30)[
                [c for c in ["ticker", "persistence_score", "full_window_pct",
                              "mean_alpha", "std_alpha", "consistency_score",
                              "trend_score", "n_appearances"] if c in persist_df.columns]
            ]
            disp_p.columns = [c.replace("_", " ").title() for c in disp_p.columns]
            st.dataframe(disp_p, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADE RESULTS
    # ════════════════════════════════════════════════════════════════════════

    with tab_trades:
        if per_trade_df.empty:
            st.warning("No trades generated. Check pattern / window data.")
        else:
            # KPIs
            all_r  = per_trade_df["return_pct"].dropna()
            all_a  = per_trade_df["alpha"].dropna()
            n_t    = len(per_trade_df)
            n_w    = int((all_a > 0).sum())
            wr     = round(n_w / len(all_a) * 100, 1) if len(all_a) else 0
            avg_r  = round(all_r.mean(), 2) if not all_r.empty else 0
            avg_a  = round(all_a.mean(), 2) if not all_a.empty else 0
            max_dd = round(all_r.min(), 2) if not all_r.empty else 0
            hit_c  = int(per_trade_df.get("entry_threshold_hit", pd.Series(dtype=bool)).sum()) \
                     if "entry_threshold_hit" in per_trade_df.columns else 0
            n_wins_unique = len(per_trade_df["window_idx"].unique())
            hit_rate = round(hit_c / n_wins_unique * 100, 1) if n_wins_unique > 0 else 0

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Total Trades",       n_t)
            k2.metric("Avg Return %",       f"{avg_r:+.2f}%",  delta_color=_color(avg_r))
            k3.metric("Avg Alpha %",        f"{avg_a:+.2f}%",  delta_color=_color(avg_a))
            k4.metric("Win Rate (Alpha>0)", f"{wr:.1f}%")
            k5.metric("Max Drawdown",       f"{max_dd:.2f}%",  delta_color="inverse")
            k6.metric("NIFTY Trig Hit Rate",f"{hit_rate:.1f}%")

            # ── Window-status / reshuffle overview ────────────────────────────
            if not window_status_df.empty:
                n_resh   = int(window_status_df["reshuffled"].sum())
                n_traded = int((window_status_df["n_trades"] > 0).sum())
                n_total  = len(window_status_df)
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Windows Traded",   n_traded)
                sc2.metric("Windows Reshuffled", n_resh,
                           help="Skipped because the Fall leg drawdown exceeded the reshuffle threshold.")
                sc3.metric("Total Windows",    n_total)

                if _reshuffle_on and n_resh > 0:
                    reshuffled = window_status_df[window_status_df["reshuffled"]]
                    with st.expander(f"🔄 {n_resh} window(s) reshuffled — see reasons", expanded=False):
                        for _, rr in reshuffled.iterrows():
                            st.markdown(f"<div class='card' style='border-color:#FF4B6E'>"
                                        f"<b style='color:#FF4B6E'>Window {int(rr['window'])}</b> — "
                                        f"<small>{rr['reason']}</small></div>", unsafe_allow_html=True)

            # ══════════════════════════════════════════════════════════════════
            # FEATURE B — ₹1,00,000 Investment Analysis
            # ══════════════════════════════════════════════════════════════════
            _section("💰 Investment Analysis — ₹1,00,000 Initial Capital")

            _n_wins_ia = int(per_trade_df["window_idx"].nunique())
            if _n_wins_ia < 2:
                st.warning("Not enough trade windows to compute investment analysis "
                           "(need at least 2 windows).")
            else:
                with st.expander("⚙️ Investment Analysis Settings", expanded=True):
                    ia1, ia2 = st.columns(2)
                    with ia1:
                        ia_capital = st.number_input(
                            "Initial Capital (₹)", min_value=10_000, value=100_000, step=5_000,
                            help="Starting capital for the simulated portfolio.")
                        ia_mode = st.radio(
                            "Capital Allocation Mode",
                            ["Equal weight across all stocks in window",
                             "Full capital into each stock independently"],
                            index=0,
                            help="Equal weight → split per-window capital across that window's stocks. "
                                 "Independent → single-stock what-if (full capital per stock).")
                    with ia2:
                        ia_reinvest = st.checkbox("Reinvest profits between windows?", value=False)
                        ia_yearwise = st.checkbox("Show year-wise breakdown?", value=True)

                _alloc_mode = "equal" if ia_mode.startswith("Equal") else "independent"

                # Persist for the Excel exporter
                st.session_state["ia_initial_capital"] = float(ia_capital)
                st.session_state["ia_alloc_mode"]      = _alloc_mode
                st.session_state["ia_reinvest"]        = bool(ia_reinvest)
                st.session_state["ia_yearwise"]        = bool(ia_yearwise)

                _ia = compute_investment_analysis(
                    per_trade_df, initial_capital=float(ia_capital),
                    alloc_mode=_alloc_mode, reinvest=bool(ia_reinvest),
                )

                if _ia is None:
                    st.warning("Not enough trade windows to compute investment analysis.")
                else:
                    m = _ia["metrics"]

                    # ── B3: metric cards ──────────────────────────────────────
                    r1 = st.columns(4)
                    r1[0].metric("Final Equity", fmt_inr_full(m["final_equity"]))
                    r1[1].metric("Total Profit / Loss", fmt_inr_full(m["total_pl"]),
                                 delta=f"{m['total_pl_pct']:+.1f}%")
                    r1[2].metric("CAGR", _fmt(m["cagr"], 2, "%") if not np.isnan(m["cagr"]) else "N/A")
                    r1[3].metric("Max Drawdown", f"{m['mdd_pct']:.1f}%", delta_color="inverse")

                    r2 = st.columns(4)
                    r2[0].metric("CAR / MDD (Calmar)",
                                 "N/A" if np.isnan(m["calmar"]) else f"{m['calmar']:.2f}")
                    r2[1].metric("Sharpe Ratio",
                                 "N/A" if np.isnan(m["sharpe"]) else f"{m['sharpe']:.2f}")
                    r2[2].metric("Win Rate", f"{m['win_rate']:.1f}%")
                    r2[3].metric("Alpha Win Rate",
                                 "N/A" if np.isnan(m["alpha_win_rate"]) else f"{m['alpha_win_rate']:.1f}%")

                    r3 = st.columns(4)
                    r3[0].metric("Total Trades", m["n_trades"])
                    r3[1].metric("Avg Profit / Trade", fmt_inr_full(m["avg_profit_trade"]))
                    r3[2].metric("Avg Holding Days",
                                 "N/A" if np.isnan(m["avg_holding_days"]) else f"{m['avg_holding_days']:.0f} days")
                    _bw = m["best_window"]
                    r3[3].metric("Best Window",
                                 fmt_inr(_bw["profit"]) if _bw else "—",
                                 help=(f"Window {_bw['window']}"
                                       + (f" · {_bw['date']:%d-%b-%Y}" if _bw and pd.notna(_bw['date']) else "")) if _bw else None)

                    if m["worst_window"]:
                        _ww = m["worst_window"]
                        st.caption(f"Worst Window: **{fmt_inr(_ww['profit'])}** "
                                   f"(Window {_ww['window']}"
                                   + (f" · {_ww['date']:%d-%b-%Y}" if pd.notna(_ww['date']) else "") + ")")

                    # ── B5: equity curve ──────────────────────────────────────
                    eq = _ia["equity_curve"].copy()
                    eq = eq.dropna(subset=["exit_date"]).sort_values("exit_date")
                    final_eq = float(eq["equity_inr"].iloc[-1]) if not eq.empty else float(ia_capital)
                    up = final_eq >= float(ia_capital)
                    line_col = "#00C896" if up else "#FF4B6E"

                    fig_ec = go.Figure()
                    fig_ec.add_trace(go.Scatter(
                        x=eq["exit_date"], y=eq["equity_inr"], mode="lines+markers",
                        line=dict(color=line_col, width=2.5), marker=dict(size=6),
                        name="Equity (₹)",
                        hovertemplate="%{x|%d-%b-%Y}<br>₹%{y:,.0f}<extra></extra>"))
                    fig_ec.add_hline(y=float(ia_capital), line_dash="dash", line_color="#8B9DB5",
                                     annotation_text=f"Initial {fmt_inr(ia_capital)}",
                                     annotation_position="bottom right")
                    if not eq.empty:
                        hi = eq.loc[eq["equity_inr"].idxmax()]
                        lo = eq.loc[eq["equity_inr"].idxmin()]
                        fig_ec.add_annotation(x=hi["exit_date"], y=hi["equity_inr"],
                                              text=f"High {fmt_inr(hi['equity_inr'])}",
                                              showarrow=True, arrowhead=2, font=dict(color="#00C896"))
                        fig_ec.add_annotation(x=lo["exit_date"], y=lo["equity_inr"],
                                              text=f"Low {fmt_inr(lo['equity_inr'])}",
                                              showarrow=True, arrowhead=2, font=dict(color="#FF4B6E"))
                        fig_ec.add_annotation(x=eq["exit_date"].iloc[-1], y=final_eq,
                                              text=f"Final {fmt_inr(final_eq)}",
                                              showarrow=True, arrowhead=2, font=dict(color=line_col))
                    fig_ec.update_layout(
                        title=f"Equity Curve — ₹{float(ia_capital):,.0f} starting capital",
                        height=340, margin=dict(t=46, b=20, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#C9C9C9", xaxis_title="Window Exit Date", yaxis_title="Equity (₹)",
                        xaxis=dict(color="#8B9DB5"), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                        showlegend=False,
                        title_font=dict(color="#C9C9C9", size=14))
                    st.plotly_chart(fig_ec, use_container_width=True)

                    # ── B4 + B6: year-wise table and per-year chart ───────────
                    yw = _ia["yearwise"]
                    if ia_yearwise and not yw.empty:
                        _section("Year-wise Breakdown")
                        disp = yw.copy()
                        disp["Invested(₹)"]          = disp["Invested"].map(fmt_inr_full)
                        disp["Profit(₹)"]            = disp["Profit"].map(fmt_inr_full)
                        disp["Cumulative Equity(₹)"] = disp["Cumulative Equity"].map(fmt_inr_full)
                        disp["Return%"]   = disp["Return%"].map(lambda v: f"{v:+.2f}%")
                        disp["Avg Alpha"] = disp["Avg Alpha"].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
                        disp["Win Rate"]  = disp["Win Rate"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
                        disp["Year"]      = disp["Year"].astype(str)
                        _profit_sign = yw["Profit"].tolist()
                        show = disp[["Year", "Windows", "Trades", "Invested(₹)", "Profit(₹)",
                                     "Return%", "Avg Alpha", "Win Rate", "Cumulative Equity(₹)"]]

                        def _hl_profit(col):
                            return [f"color:{'#00C896' if p >= 0 else '#FF4B6E'};font-weight:600"
                                    for p in _profit_sign]
                        styler = show.style.apply(_hl_profit, subset=["Profit(₹)"], axis=0)
                        st.dataframe(styler, use_container_width=True, hide_index=True)

                        # B6: per-year profit bars + cumulative equity line
                        ywx = yw[yw["Year"] != "TOTAL"].copy()
                        if not ywx.empty:
                            ywx["Year"] = ywx["Year"].astype(int).astype(str)
                            fig_py = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_py.add_trace(go.Bar(
                                x=ywx["Year"], y=ywx["Profit"],
                                marker_color=["#00C896" if p >= 0 else "#FF4B6E" for p in ywx["Profit"]],
                                name="Annual Profit (₹)",
                                text=[fmt_inr(p) for p in ywx["Profit"]], textposition="outside"),
                                secondary_y=False)
                            fig_py.add_trace(go.Scatter(
                                x=ywx["Year"], y=ywx["Cumulative Equity"], mode="lines+markers",
                                line=dict(color="#FFD700", width=2), marker=dict(size=7),
                                name="Cumulative Equity (₹)"),
                                secondary_y=True)
                            fig_py.update_layout(
                                height=320, margin=dict(t=30, b=20, l=10, r=10),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color="#C9C9C9", xaxis_title="Year",
                                xaxis=dict(color="#8B9DB5"),
                                legend=dict(font=dict(color="#C9C9C9"), orientation="h", y=1.15))
                            fig_py.update_yaxes(title_text="Annual Profit (₹)", color="#8B9DB5",
                                                gridcolor="#2A2A4A", secondary_y=False)
                            fig_py.update_yaxes(title_text="Cumulative Equity (₹)", color="#FFD700",
                                                showgrid=False, secondary_y=True)
                            st.plotly_chart(fig_py, use_container_width=True)

            # ── Per-window performance bar ────────────────────────────────────
            _section("Per-Window Performance")
            win_perf = (per_trade_df.groupby("window_idx")
                        .agg(avg_return=("return_pct", "mean"),
                             avg_alpha=("alpha", "mean"),
                             n_trades=("ticker", "count"))
                        .reset_index())
            win_perf["window"] = win_perf["window_idx"] + 1
            fig_wp = go.Figure()
            fig_wp.add_trace(go.Bar(
                x=win_perf["window"], y=win_perf["avg_return"].round(2),
                marker_color=["#00C896" if v >= 0 else "#FF4B6E" for v in win_perf["avg_return"]],
                name="Avg Return %",
                text=[f"{v:+.1f}%" for v in win_perf["avg_return"]], textposition="outside",
            ))
            fig_wp.add_trace(go.Scatter(
                x=win_perf["window"], y=win_perf["avg_alpha"].round(2),
                mode="lines+markers", name="Avg Alpha %",
                line=dict(color="#FFD700", width=2), marker=dict(size=7),
            ))
            fig_wp.update_layout(
                height=300, margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C9C9C9", xaxis_title="Window", yaxis_title="%",
                xaxis=dict(color="#8B9DB5", dtick=1), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                legend=dict(font=dict(color="#C9C9C9"), orientation="h", y=1.12),
            )
            st.plotly_chart(fig_wp, use_container_width=True)

            # ── Cumulative equity curve + win/loss donut ──────────────────────
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                _section("Cumulative Return (sequential trades)")
                seq = per_trade_df.sort_values(["window_idx"]).reset_index(drop=True)
                eq = (1 + seq["return_pct"].fillna(0) / 100).cumprod()
                cum_pct = (eq - 1) * 100
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=list(range(1, len(cum_pct) + 1)), y=cum_pct.round(2),
                    mode="lines", fill="tozeroy", name="Cumulative %",
                    line=dict(color="#6C63FF", width=2),
                    fillcolor="rgba(108,99,255,0.15)",
                ))
                fig_eq.update_layout(
                    height=280, margin=dict(t=10, b=20, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#C9C9C9", xaxis_title="Trade #", yaxis_title="Cumulative %",
                    xaxis=dict(color="#8B9DB5"), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                )
                st.plotly_chart(fig_eq, use_container_width=True)
            with cc2:
                _section("Win / Loss Split")
                n_win  = int((all_r > 0).sum())
                n_loss = int((all_r <= 0).sum())
                fig_wl = go.Figure(go.Pie(
                    labels=["Winning", "Losing"], values=[n_win, n_loss], hole=0.5,
                    marker=dict(colors=["#00C896", "#FF4B6E"]), textinfo="label+percent",
                ))
                fig_wl.update_layout(
                    height=280, margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#C9C9C9", showlegend=False,
                )
                st.plotly_chart(fig_wl, use_container_width=True)

            # ── Top Common Stocks — entry/exit P&L ────────────────────────────
            _section("Top Common Stocks — Entry, Exit, Trade-Phase Quartile & P&L")
            pnl = per_trade_df.copy()
            # Attach quartile from the TRADE-PHASE ranking (buy→sell return rank)
            if not trade_phase_ranks_df.empty and "quartile" in trade_phase_ranks_df.columns:
                qmap = {(int(r["window_idx"]), r["ticker"]): r["quartile"]
                        for _, r in trade_phase_ranks_df.iterrows()}
            else:
                qmap = {}
            pnl["quartile"] = pnl.apply(
                lambda r: qmap.get((int(r["window_idx"]), r["ticker"]), "—"), axis=1)
            for dc in ["entry_date", "exit_date"]:
                if dc in pnl.columns:
                    pnl[dc] = pd.to_datetime(pnl[dc], errors="coerce")
            pnl_disp = (pnl[["window_idx", "ticker", "quartile", "entry_date", "entry_price",
                              "exit_date", "exit_price", "return_pct", "alpha"]]
                        .sort_values("alpha", ascending=False).head(25).copy())
            pnl_disp["window_idx"] = pnl_disp["window_idx"] + 1
            pnl_disp["P&L"] = pnl_disp["return_pct"].apply(lambda v: "PROFIT" if v > 0 else "LOSS")
            for dc in ["entry_date", "exit_date"]:
                pnl_disp[dc] = pnl_disp[dc].dt.strftime("%d-%b-%Y")
            pnl_disp.columns = ["Window", "Ticker", "Quartile", "Entry Date", "Entry Price",
                                "Exit Date", "Exit Price", "Return %", "Alpha %", "P&L"]

            def _hl_pnl(row):
                col = "#C6EFCE" if row["P&L"] == "PROFIT" else "#FFC7CE"
                fg  = "#1A6B3C" if row["P&L"] == "PROFIT" else "#9C0006"
                return [f"background-color:{col};color:{fg}" if c in ("Return %", "Alpha %", "P&L") else "" for c in row.index]

            st.dataframe(
                pnl_disp.style.apply(_hl_pnl, axis=1)
                        .format({"Return %": "{:+.2f}", "Alpha %": "{:+.2f}",
                                 "Entry Price": "{:.2f}", "Exit Price": "{:.2f}"}),
                use_container_width=True, height=360,
            )

            _section("Stock Summary (aggregated stats of executed per-window trades)")
            if not summary_df.empty:
                disp_s = summary_df.copy()
                for c in ["avg_return", "avg_alpha", "win_rate", "max_drawdown", "annualised_return"]:
                    if c in disp_s.columns:
                        disp_s[c] = disp_s[c].apply(lambda v: f"{v:+.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—")

                def _hl_alpha(row):
                    try:
                        v = float(str(row.get("avg_alpha", "0")).replace("+", ""))
                        if v > 0:
                            return ["background-color:#C6EFCE;color:#1A6B3C" if c == "avg_alpha" else "" for c in row.index]
                        elif v < 0:
                            return ["background-color:#FFC7CE;color:#9C0006" if c == "avg_alpha" else "" for c in row.index]
                    except Exception:
                        pass
                    return [""] * len(row.index)

                st.dataframe(
                    disp_s.rename(columns=lambda c: c.replace("_", " ").title())
                          .style.apply(_hl_alpha, axis=1),
                    use_container_width=True, height=350,
                )

            _section("Full Trade Log")
            disp_t = per_trade_df.copy()
            for dc in ["entry_date", "exit_date", "buy_phase_start", "buy_phase_end",
                       "sell_phase_start", "sell_phase_end", "entry_triggered_on_date"]:
                if dc in disp_t.columns:
                    disp_t[dc] = pd.to_datetime(disp_t[dc], errors="coerce").dt.strftime("%d-%b-%Y")

            key_cols = [
                "window_idx", "buy_phase", "buy_phase_start", "buy_phase_end",
                "nifty_buy_base", "nifty_entry_trigger_level", "nifty_at_entry",
                "entry_triggered_on_date", "entry_threshold_hit",
                "ticker", "entry_date", "exit_date",
                "entry_price", "exit_price", "return_pct", "nifty_return", "alpha", "days_held"
            ]
            avail_cols = [c for c in key_cols if c in disp_t.columns]
            disp_t = disp_t[avail_cols].copy()
            disp_t["window_idx"] = disp_t["window_idx"] + 1
            disp_t.columns = [c.replace("_", " ").title() for c in disp_t.columns]

            st.dataframe(disp_t, use_container_width=True, height=450)

            # Return distribution chart
            _section("Return Distribution")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=per_trade_df["return_pct"].dropna(),
                nbinsx=30,
                marker_color=["#00C896" if x >= 0 else "#FF4B6E"
                              for x in per_trade_df["return_pct"].dropna()],
                name="Return %",
            ))
            fig_hist.update_layout(
                height=280, margin=dict(t=10, b=30, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#C9C9C9", xaxis_title="Return %", yaxis_title="Count",
                xaxis=dict(color="#8B9DB5"), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — NIFTY ENTRY VISUALISATION
    # ════════════════════════════════════════════════════════════════════════

    with tab_vis:
        if per_trade_df.empty or nifty_df.empty:
            st.warning("Run the analysis first.")
        else:
            _section("NIFTY Entry Trigger — Window Visualisation")
            _card(
                f"For each window, this chart shows the NIFTY path during the <b>buy phase</b>. "
                f"The <span style='color:#FFD700'><b>dashed gold line</b></span> is the trigger level "
                f"(NIFTY Base + {_entry_thr:.1f}%). "
                f"The <span style='color:#00C896'><b>green marker</b></span> is the actual buy trigger date. "
                f"Stocks are bought on that day (with Aux2 forward-walk if needed)."
            )

            # Window selector
            unique_wins = per_trade_df.drop_duplicates("window_idx").sort_values("window_idx")
            win_labels  = [
                f"Window {int(r['window_idx'])+1}  ({pd.Timestamp(r['buy_phase_start']).strftime('%d-%b-%Y')} — {pd.Timestamp(r['buy_phase_end']).strftime('%d-%b-%Y')})"
                for _, r in unique_wins.iterrows()
            ]
            sel_vis = st.selectbox("Select Window", win_labels, key="vis_win")
            sel_vis_idx = win_labels.index(sel_vis)
            trade_row   = unique_wins.iloc[sel_vis_idx]

            buy_start   = pd.Timestamp(trade_row["buy_phase_start"])
            buy_end     = pd.Timestamp(trade_row["buy_phase_end"])
            n_base      = float(trade_row.get("nifty_buy_base") or 0)
            trig_lvl    = float(trade_row.get("nifty_entry_trigger_level") or 0)
            entry_date  = trade_row.get("entry_triggered_on_date")
            hit         = bool(trade_row.get("entry_threshold_hit", False))

            # Also get sell phase for context
            sell_start  = pd.Timestamp(trade_row.get("sell_phase_start") or buy_end)
            sell_end    = pd.Timestamp(trade_row.get("sell_phase_end") or buy_end)
            n_sell_base = float(trade_row.get("nifty_sell_base") or 0)
            sell_trig   = float(trade_row.get("nifty_exit_trigger_level") or 0)
            exit_date   = trade_row.get("exit_triggered_on_date")

            # Extend NIFTY window to include both buy and sell phase
            vis_end     = sell_end if sell_end > buy_end else buy_end
            phase_nifty = nifty_df[
                (nifty_df.index >= buy_start - pd.Timedelta(days=5)) &
                (nifty_df.index <= vis_end + pd.Timedelta(days=5))
            ]

            if not phase_nifty.empty:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    subplot_titles=["NIFTY Path — Buy Phase → Sell Phase", "% Change from Buy Phase Start"],
                    vertical_spacing=0.08,
                )

                # ── Main NIFTY line ──────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=phase_nifty.index, y=phase_nifty["close"],
                    mode="lines", name="NIFTY Close",
                    line=dict(color="#6C63FF", width=2),
                ), row=1, col=1)

                # ── Buy phase background ─────────────────────────────────────
                fig.add_vrect(
                    x0=buy_start, x1=buy_end,
                    fillcolor="rgba(0,200,150,0.08)", line_width=0,
                    row=1, col=1,
                    annotation_text=f"Buy Phase ({trade_row.get('buy_phase','')})",
                    annotation_position="top left",
                    annotation_font=dict(color="#00C896", size=11),
                )

                # ── Sell phase background ────────────────────────────────────
                if sell_end > buy_end:
                    fig.add_vrect(
                        x0=sell_start, x1=sell_end,
                        fillcolor="rgba(255,75,110,0.07)", line_width=0,
                        row=1, col=1,
                        annotation_text=f"Sell Phase ({trade_row.get('sell_phase','')})",
                        annotation_position="top right",
                        annotation_font=dict(color="#FF4B6E", size=11),
                    )

                # ── NIFTY base level ─────────────────────────────────────────
                if n_base:
                    fig.add_hline(
                        y=n_base, line_dash="dot", line_color="#8B9DB5", line_width=1,
                        annotation_text=f"Base: {n_base:.0f}",
                        annotation_font=dict(color="#8B9DB5"),
                        row=1, col=1,
                    )

                # ── Entry trigger level ──────────────────────────────────────
                if trig_lvl:
                    fig.add_hline(
                        y=trig_lvl, line_dash="dash", line_color="#FFD700", line_width=2,
                        annotation_text=f"Entry Trigger: {trig_lvl:.0f} (+{_entry_thr:.1f}%)",
                        annotation_font=dict(color="#FFD700", size=11),
                        row=1, col=1,
                    )

                # ── Exit trigger level ───────────────────────────────────────
                if sell_trig and n_sell_base:
                    fig.add_hline(
                        y=sell_trig, line_dash="dash", line_color="#FF4B6E", line_width=1.5,
                        annotation_text=f"Exit Trigger: {sell_trig:.0f} (-{_exit_thr:.1f}%)",
                        annotation_font=dict(color="#FF4B6E", size=11),
                        row=1, col=1,
                    )

                # ── Entry triggered marker ───────────────────────────────────
                if entry_date is not None:
                    try:
                        ed = pd.Timestamp(entry_date)
                        n_rows = phase_nifty[phase_nifty.index >= ed]
                        if not n_rows.empty:
                            ey = float(n_rows["close"].iloc[0])
                            fig.add_trace(go.Scatter(
                                x=[ed], y=[ey],
                                mode="markers+text",
                                marker=dict(size=14, color="#00C896", symbol="triangle-up",
                                            line=dict(color="#FFFFFF", width=2)),
                                text=["BUY"],
                                textposition="top center",
                                textfont=dict(color="#00C896", size=11, family="Arial Black"),
                                name="Buy Triggered",
                            ), row=1, col=1)
                    except Exception:
                        pass

                # ── Exit triggered marker ────────────────────────────────────
                if exit_date is not None:
                    try:
                        xd = pd.Timestamp(exit_date)
                        n_rows = phase_nifty[phase_nifty.index <= xd]
                        if not n_rows.empty:
                            xy = float(n_rows["close"].iloc[-1])
                            fig.add_trace(go.Scatter(
                                x=[xd], y=[xy],
                                mode="markers+text",
                                marker=dict(size=14, color="#FF4B6E", symbol="triangle-down",
                                            line=dict(color="#FFFFFF", width=2)),
                                text=["SELL"],
                                textposition="bottom center",
                                textfont=dict(color="#FF4B6E", size=11, family="Arial Black"),
                                name="Sell Triggered",
                            ), row=1, col=1)
                    except Exception:
                        pass

                # ── % change subplot ─────────────────────────────────────────
                buy_start_nifty_rows = phase_nifty[phase_nifty.index >= buy_start]
                if not buy_start_nifty_rows.empty:
                    ref = float(buy_start_nifty_rows["close"].iloc[0])
                    pct_change = ((phase_nifty["close"] - ref) / ref * 100).round(2)
                    fig.add_trace(go.Bar(
                        x=pct_change.index, y=pct_change.values,
                        marker_color=["#00C896" if v >= 0 else "#FF4B6E" for v in pct_change.values],
                        name="% from phase start",
                    ), row=2, col=1)

                    # Threshold lines on subplot
                    if _entry_thr:
                        fig.add_hline(y=_entry_thr, line_dash="dash", line_color="#FFD700",
                                      annotation_text=f"+{_entry_thr:.1f}%", row=2, col=1)
                    if _exit_thr:
                        fig.add_hline(y=-_exit_thr, line_dash="dash", line_color="#FF4B6E",
                                      annotation_text=f"-{_exit_thr:.1f}%", row=2, col=1)

                fig.update_layout(
                    height=550,
                    margin=dict(t=40, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#C9C9C9",
                    xaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                    xaxis2=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                    yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                    yaxis2=dict(color="#8B9DB5", gridcolor="#2A2A4A", title="% Change"),
                    legend=dict(font=dict(color="#C9C9C9"), bgcolor="rgba(0,0,0,0)"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Trigger info box
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("NIFTY Base",       f"{n_base:,.2f}" if n_base else "—")
                col_b.metric(f"Entry Trigger (+{_entry_thr:.1f}%)", f"{trig_lvl:,.2f}" if trig_lvl else "—")
                col_c.metric("Threshold Hit?",   "✓ YES" if hit else "— NO (fallback)")
                if entry_date:
                    col_d.metric("Buy Triggered On",
                                 pd.Timestamp(entry_date).strftime("%d-%b-%Y") if entry_date else "—")

                # Show stocks bought on this window
                _section(f"Stocks Bought — Window {int(trade_row['window_idx'])+1}")
                win_trades = per_trade_df[per_trade_df["window_idx"] == trade_row["window_idx"]].copy()
                if not win_trades.empty:
                    for dc in ["entry_date", "exit_date"]:
                        if dc in win_trades.columns:
                            win_trades[dc] = pd.to_datetime(win_trades[dc], errors="coerce").dt.strftime("%d-%b-%Y")

                    show_cols = [c for c in ["ticker", "entry_date", "exit_date",
                                              "entry_price", "exit_price", "return_pct",
                                              "nifty_return", "alpha", "days_held"] if c in win_trades.columns]
                    wt = win_trades[show_cols].copy()
                    wt.columns = [c.replace("_", " ").title() for c in wt.columns]

                    def _hl_ret_row(row):
                        try:
                            ret = float(str(row.get("Return Pct", "0")).replace("+", ""))
                            if ret > 0:
                                return ["background-color:#C6EFCE;color:#1A6B3C" if "Return" in c or "Alpha" in c else "" for c in row.index]
                            elif ret < 0:
                                return ["background-color:#FFC7CE;color:#9C0006" if "Return" in c or "Alpha" in c else "" for c in row.index]
                        except Exception:
                            pass
                        return [""] * len(row.index)

                    st.dataframe(
                        wt.style.apply(_hl_ret_row, axis=1)
                                .format({c: "{:+.2f}" for c in wt.columns if "Pct" in c or "Alpha" in c or "Return" in c and wt[c].dtype in [float]}),
                        use_container_width=True,
                    )
            else:
                st.info("No NIFTY data available for this window's date range.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — EXPORT
    # ════════════════════════════════════════════════════════════════════════

    with tab_export:
        _section("Download Excel Report")
        _card(
            "The Excel workbook contains a full, per-window analysis:<br>"
            "① <b>Portfolio Summary</b> — strategy config (incl. reshuffle) + overall metrics<br>"
            "② <b>Window Status</b> — every window: traded / reshuffled, fall-leg %, reason, # trades<br>"
            "③ <b>Window Candidates</b> — per-window common stocks (pattern-leg selection) + Performance Summary by Period<br>"
            "④ <b>Window Stock Ranks</b> — every stock valid across all legs, ranked per window; bought stocks highlighted<br>"
            "⑤ <b>Trade Phase Rankings ★ NEW</b> — ALL eligible stocks ranked by actual buy→sell return per window; traded stocks highlighted; <b>Quartile from this ranking</b><br>"
            "⑥ <b>Yearwise Summary</b> — period summary + per-year breakdown with <b>Trade-Phase Quartile</b> breakdown (Q1..Q4)<br>"
            "⑦ <b>Common Stocks P&amp;L</b> — each traded stock: entry buy, exit sell, <b>Trade-Phase Quartile</b>, profit/loss<br>"
            "⑧ <b>Trade Log</b> — every individual trade with full NIFTY trigger details<br>"
            "⑨ <b>Stock Summary</b> — per-ticker aggregated statistics<br>"
            "⑩ <b>NIFTY Entry Visualisation</b> — day-by-day NIFTY path per window<br>"
            "⑪ <b>Window Analysis</b> — per-leg top stocks + pattern-leg Quartile + common flag<br>"
            "⑫ <b>Phase Schedule</b> — full phase listing"
        )

        cfg_dict = {
            "pattern"           : _pattern,
            "leg_count"         : len(_pattern),
            "top_n"             : _top_n,
            "top_k_common"      : _top_k,
            "entry_threshold_pct": _entry_thr,
            "exit_threshold_pct" : _exit_thr,
            "persistence_enabled": _pers_on,
            "sort_by"           : _sort_by,
            "reshuffle_enabled" : _reshuffle_on,
            "reshuffle_threshold": _reshuffle_thr,
            "include_entry_segment": _include_seg,
            "lowvol_enabled"    : _lowvol_on,
            "lowvol_pct"        : _lowvol_pct,
            "winvol_mode"       : _winvol_mode,
            "winvol_pct"        : _winvol_pct,
        }

        if not per_trade_df.empty or (not window_status_df.empty and bool(window_status_df["reshuffled"].any())):
            if st.button("Generate Excel File", type="primary"):
                with st.spinner("Building Excel workbook..."):
                    try:
                        xl_bytes = generate_excel(
                            per_trade_df = per_trade_df,
                            summary_df   = summary_df,
                            phases       = phases,
                            returns_df   = returns_df,
                            nifty_df     = nifty_df,
                            windows      = all_windows,
                            freq_df      = freq_ordered,
                            config       = cfg_dict,
                            top_n        = _top_n,
                            sort_by      = _sort_by,
                            window_status_df     = window_status_df,
                            candidates_df        = candidates_df,
                            window_ranks_df      = window_ranks_df,
                            trade_phase_ranks_df = trade_phase_ranks_df,
                            initial_capital      = float(st.session_state.get("ia_initial_capital", 100_000)),
                            alloc_mode           = st.session_state.get("ia_alloc_mode", "equal"),
                            reinvest             = bool(st.session_state.get("ia_reinvest", False)),
                        )
                        st.download_button(
                            label="⬇️ Download Excel Report",
                            data=xl_bytes,
                            file_name=f"En.{_pattern[0]}_TopN-{_top_n}_TopC-{_top_k}_{_entry_thr:g}%Buy_{_exit_thr:g}%Sell.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                        )
                        st.success("✓ Excel report ready!")
                    except Exception as e:
                        st.error(f"Excel generation failed: {e}")

        else:
            st.info("Run the analysis first to generate the Excel export.")

        # CSV download of trade log always available
        if not per_trade_df.empty or not window_status_df.empty:
            st.divider()
            _section("Quick CSV Downloads")
            c1, c2, c3 = st.columns(3)
            with c1:
                if not per_trade_df.empty:
                    st.download_button(
                        "⬇️ Trade Log CSV",
                        per_trade_df.to_csv(index=False).encode("utf-8"),
                        f"trade_log_{'_'.join(_pattern)}.csv",
                        "text/csv",
                    )
            with c2:
                if not window_status_df.empty:
                    st.download_button(
                        "⬇️ Window Status CSV",
                        window_status_df.to_csv(index=False).encode("utf-8"),
                        f"window_status_{'_'.join(_pattern)}.csv",
                        "text/csv",
                    )
            with c3:
                if not candidates_df.empty:
                    st.download_button(
                        "⬇️ Window Candidates CSV",
                        candidates_df.to_csv(index=False).encode("utf-8"),
                        f"window_candidates_{'_'.join(_pattern)}.csv",
                        "text/csv",
                    )
            if not window_ranks_df.empty:
                st.download_button(
                    "⬇️ Window Stock Ranks CSV (all considered stocks)",
                    window_ranks_df.to_csv(index=False).encode("utf-8"),
                    f"window_stock_ranks_{'_'.join(_pattern)}.csv",
                    "text/csv",
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — ML PORTFOLIO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ml:
        _section("🤖 Machine-Learning Trade Filter & Portfolio")
        st.markdown(
            "<div class='card'>A gradient-boosting model learns — from <b>pre-entry information only</b> "
            "(technical momentum, volatility, RSI, NIFTY context, window volatility) — which trades are likely "
            "to be profitable. Windows are split <b>chronologically</b> (earlier windows train, later windows test) "
            "so results are <b>out-of-sample</b> and free of look-ahead. The search then picks the model settings "
            "and a probability threshold that maximise the out-of-sample <b>CAR/MDD</b>, and builds a portfolio "
            "from only the approved trades.</div>", unsafe_allow_html=True)

        if per_trade_df.empty or per_trade_df["window_idx"].nunique() < 4:
            st.warning("Need at least ~4 trade windows (and ~12 trades) to train the model. "
                       "Loosen the strategy or widen the date range, then re-run the analysis.")
        else:
            with st.expander("⚙️ Model Settings", expanded=True):
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    ml_target = st.radio("Predict", ["Profitable trade (return > 0)", "Beats NIFTY (alpha > 0)"],
                                         index=0)
                    ml_target = "alpha" if ml_target.startswith("Beats") else "profit"
                with mc2:
                    ml_capital = st.number_input("Initial Capital (₹)", 10_000, value=100_000, step=5_000, key="ml_cap")
                    ml_reinvest = st.checkbox("Reinvest profits between windows?", value=False, key="ml_re")
                with mc3:
                    ml_train = st.slider("Train window fraction", 0.4, 0.8, 0.6, 0.05)
                    ml_search = st.radio("Search depth", ["Quick", "Thorough"], index=0,
                                         help="Thorough explores a larger grid of model settings (slower).")

            if st.button("🚀 Train Model & Build ML Portfolio", type="primary"):
                grid = ({"n_estimators": [80, 150], "learning_rate": [0.05, 0.10], "max_depth": [2, 3]}
                        if ml_search == "Quick" else
                        {"n_estimators": [80, 150, 250], "learning_rate": [0.03, 0.06, 0.10],
                         "max_depth": [2, 3, 4]})
                with st.spinner("Engineering features, training gradient-boosting model, searching settings..."):
                    try:
                        st.session_state["ml_result"] = run_ml(
                            per_trade_df, stock_dict, nifty_df, target=ml_target,
                            initial_capital=float(ml_capital), reinvest=bool(ml_reinvest),
                            grid=grid, train_frac=float(ml_train), valid_frac=0.2)
                        st.session_state["ml_capital_used"] = float(ml_capital)
                    except Exception as e:
                        st.session_state["ml_result"] = {"ok": False, "msg": f"Training failed: {e}"}

            _mlr = st.session_state.get("ml_result")
            if _mlr is None:
                st.info("Set your options above and click **Train Model & Build ML Portfolio**.")
            elif not _mlr.get("ok"):
                st.warning(_mlr.get("msg", "Model could not be trained."))
            else:
                sp = _mlr["split"]; b = _mlr["best"]; tm = _mlr["test_metrics"]
                _capml = st.session_state.get("ml_capital_used", 100_000.0)
                _f2 = lambda x: "N/A" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.2f}"
                _fp = lambda x: "N/A" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.1f}%"

                if _mlr.get("sel_is_test"):
                    st.caption("⚠️ Few windows available — settings were selected on the test slice itself, "
                               "so out-of-sample figures are optimistic. Add more trade windows for a clean split.")

                st.markdown(f"<div class='card'><b>Engine:</b> {_mlr.get('engine','XGBoost')} &nbsp;·&nbsp; "
                            f"Split by window — "
                            f"<b>{sp['train_w']}</b> train · <b>{sp['valid_w']}</b> validation · "
                            f"<b>{sp['test_w']}</b> test &nbsp;(of {sp['n_windows']} windows). "
                            f"Trained on {_mlr['n_train_trades']} trades, tested on {_mlr['n_test_trades']}.</div>",
                            unsafe_allow_html=True)

                # ── Model performance ─────────────────────────────────────────
                _section("Model Performance — out-of-sample (test windows)")
                p1 = st.columns(4)
                p1[0].metric("Accuracy", _fp(tm["accuracy"]*100 if tm["accuracy"]==tm["accuracy"] else float('nan')))
                p1[1].metric("Precision", _fp(tm["precision"]*100 if tm["precision"]==tm["precision"] else float('nan')),
                             help="Of trades the model approved, how many were actually profitable.")
                p1[2].metric("Recall", _fp(tm["recall"]*100 if tm["recall"]==tm["recall"] else float('nan')))
                p1[3].metric("AUC", _f2(tm["auc"]), help="Ranking quality; 0.5 = coin-flip, 1.0 = perfect.")
                st.caption(f"Confusion matrix (test): "
                           f"✅ true-profit {tm['tp']} · ❌ false-approve {tm['fp']} · "
                           f"missed-winners {tm['fn']} · correctly-avoided {tm['tn']}")

                # ── Best settings ─────────────────────────────────────────────
                _section("Best Settings Found")
                bs = st.columns(4)
                bs[0].metric("Trees", b["n_estimators"])
                bs[1].metric("Learning Rate", f"{b['learning_rate']:.2f}")
                bs[2].metric("Tree Depth", b["max_depth"])
                bs[3].metric("Prob. Threshold", f"{b['threshold']:.2f}",
                             help="A trade is taken only if the model's profit-probability is at least this.")

                # ── ML portfolio vs baseline (test) ───────────────────────────
                _section("ML-Filtered Portfolio vs. Take-Every-Trade — out-of-sample")
                ml = _mlr["ml_portfolio"]; base = _mlr["baseline_portfolio"]
                comp = pd.DataFrame({
                    "Metric": ["Trades taken", "Win rate", "Final equity", "Total return",
                               "CAGR", "Max drawdown", "CAR / MDD (Calmar)"],
                    "Take every trade": [base["n"], _fp(base["win_rate"]), fmt_inr_full(base["final"]),
                                          _fp(base["profit_pct"]), _fp(base["cagr"]),
                                          _fp(base["mdd"]), _f2(base["calmar"])],
                    "ML-filtered": [ml["n"], _fp(ml["win_rate"]), fmt_inr_full(ml["final"]),
                                     _fp(ml["profit_pct"]), _fp(ml["cagr"]),
                                     _fp(ml["mdd"]), _f2(ml["calmar"])],
                })
                st.dataframe(comp.astype(str), use_container_width=True, hide_index=True)

                # equity overlay (test)
                ia_ml = ml.get("ia"); ia_base = base.get("ia")
                if ia_base is not None or ia_ml is not None:
                    figm = go.Figure()
                    if ia_base is not None:
                        eb = ia_base["equity_curve"].dropna(subset=["exit_date"]).sort_values("exit_date")
                        figm.add_trace(go.Scatter(x=eb["exit_date"], y=eb["equity_inr"], mode="lines+markers",
                                       name="Take every trade", line=dict(color="#8B9DB5", width=2, dash="dot")))
                    if ia_ml is not None:
                        em = ia_ml["equity_curve"].dropna(subset=["exit_date"]).sort_values("exit_date")
                        figm.add_trace(go.Scatter(x=em["exit_date"], y=em["equity_inr"], mode="lines+markers",
                                       name="ML-filtered", line=dict(color="#00C896", width=2.6)))
                    figm.add_hline(y=float(_capml), line_dash="dash", line_color="#8B9DB5")
                    figm.update_layout(height=340, margin=dict(t=24, b=20, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9C9C9",
                        title="Out-of-sample equity — ML-filtered vs. take-every-trade",
                        xaxis_title="Window exit date", yaxis_title="Equity (₹)",
                        xaxis=dict(color="#8B9DB5"), yaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"),
                        legend=dict(font=dict(color="#C9C9C9"), orientation="h", y=1.14),
                        title_font=dict(color="#C9C9C9", size=14))
                    st.plotly_chart(figm, use_container_width=True)

                # ── Feature importances ───────────────────────────────────────
                _section("What the model paid attention to")
                imp = _mlr["importances"][:12][::-1]
                fig_imp = go.Figure(go.Bar(
                    x=[v for _, v in imp], y=[f for f, _ in imp], orientation="h",
                    marker_color="#7b78ff"))
                fig_imp.update_layout(height=360, margin=dict(t=10, b=20, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#C9C9C9",
                    xaxis_title="Relative importance",
                    xaxis=dict(color="#8B9DB5", gridcolor="#2A2A4A"), yaxis=dict(color="#C9C9C9"))
                st.plotly_chart(fig_imp, use_container_width=True)

                # ── Full-period ML portfolio + year-wise ──────────────────────
                full = _mlr["full_ml_portfolio"]; ia_full = full.get("ia")
                _section("Full-Period ML-Filtered Portfolio (all windows the model approves)")
                fc = st.columns(4)
                fc[0].metric("Final Equity", fmt_inr_full(full["final"]))
                fc[1].metric("CAGR", _fp(full["cagr"]))
                fc[2].metric("Max Drawdown", _fp(full["mdd"]), delta_color="inverse")
                fc[3].metric("CAR / MDD", _f2(full["calmar"]))
                if ia_full is not None and not ia_full["yearwise"].empty:
                    yw = ia_full["yearwise"].copy()
                    yw["Invested(₹)"] = yw["Invested"].map(fmt_inr_full)
                    yw["Profit(₹)"]   = yw["Profit"].map(fmt_inr_full)
                    yw["Equity(₹)"]   = yw["Cumulative Equity"].map(fmt_inr_full)
                    yw["Return%"]     = yw["Return%"].map(lambda v: f"{v:+.2f}%")
                    yw["Win Rate"]    = yw["Win Rate"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
                    yw["Year"]        = yw["Year"].astype(str)
                    st.dataframe(yw[["Year", "Windows", "Trades", "Invested(₹)", "Profit(₹)",
                                     "Return%", "Win Rate", "Equity(₹)"]],
                                 use_container_width=True, hide_index=True)

                # ── Settings leaderboard ──────────────────────────────────────
                with st.expander("🔬 Settings search leaderboard (selection slice)", expanded=False):
                    lb = _mlr["leaderboard"].copy()
                    for c in ("sel_auc", "sel_acc", "sel_calmar", "sel_cagr", "sel_mdd",
                              "sel_profit_pct", "sel_win_rate"):
                        if c in lb.columns:
                            lb[c] = lb[c].map(lambda v: "" if (v is None or (isinstance(v, float) and np.isnan(v)))
                                              else f"{v:.2f}")
                    st.dataframe(lb, use_container_width=True, hide_index=True)

                st.caption(f"Model: {_mlr.get('engine', 'XGBoost')} · binary log-loss · subsampled · "
                           "class-imbalance weighted. Features use only information available at entry. "
                           "Research tool — not investment advice.")


else:
    # No analysis run yet
    with tab_windows:
        st.info("Configure settings in the sidebar and click **🚀 Run Analysis**.")
    with tab_trades:
        st.info("Configure settings in the sidebar and click **🚀 Run Analysis**.")
    with tab_vis:
        st.info("Configure settings in the sidebar and click **🚀 Run Analysis**.")
    with tab_export:
        st.info("Configure settings in the sidebar and click **🚀 Run Analysis**.")
    with tab_ml:
        st.info("Run the analysis first — the ML model trains on the trades it produces.")
