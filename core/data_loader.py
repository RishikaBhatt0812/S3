"""
S³ Core — Data Loader
=====================
Loads Data.xlsx (stock universe) and Dates.xlsx (phases + NIFTY).
Uses calamine engine if available, falls back to openpyxl.
All data is cached by file hash.
"""
from __future__ import annotations

import hashlib
import io
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


def _best_engine() -> str:
    try:
        import python_calamine  # noqa
        return "calamine"
    except ImportError:
        return "openpyxl"


_EXCEL_ENGINE = _best_engine()


def _file_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@st.cache_data(show_spinner=False)
def _read_all_sheets(file_bytes: bytes, _fh: str) -> dict[str, pd.DataFrame]:
    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine=_EXCEL_ENGINE)
    except Exception:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
    return {str(k): v for k, v in sheets.items()}


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner=False)
def load_phases(file_bytes: bytes) -> pd.DataFrame:
    """Load phase schedule from Dates.xlsx → phase_id, trade, entry_date, exit_date."""
    fh = _file_hash(file_bytes)
    all_sheets = _read_all_sheets(file_bytes, fh)

    # Pick the right sheet
    sheet = None
    for s in all_sheets:
        sl = s.lower()
        if any(x in sl for x in ("phase", "trade", "2008", "nifty_phase")):
            sheet = s
            break
    if sheet is None:
        sheet = next(iter(all_sheets))

    df = all_sheets[sheet].copy()
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "trade":
            col_map[c] = "trade"
        elif "entry" in cl and "date" in cl:
            col_map[c] = "entry_date"
        elif "exit" in cl and "date" in cl:
            col_map[c] = "exit_date"
        elif "day" in cl:
            col_map[c] = "days"
    df = df.rename(columns=col_map)

    df["entry_date"] = _parse_date(df["entry_date"]).dt.normalize()
    df["exit_date"]  = _parse_date(df["exit_date"]).dt.normalize()
    df = df.dropna(subset=["entry_date", "exit_date", "trade"])
    df["trade"] = df["trade"].astype(str).str.strip().str.capitalize()
    df = df[df["trade"].isin(["Rise", "Fall"])].reset_index(drop=True)
    df["phase_id"] = range(len(df))
    df["days"] = (df["exit_date"] - df["entry_date"]).dt.days
    return df[["phase_id", "trade", "entry_date", "exit_date", "days"]]


@st.cache_data(show_spinner=False)
def load_nifty(file_bytes: bytes) -> pd.DataFrame:
    """Load NIFTY daily closes from Dates.xlsx → date-indexed, 'close' column."""
    fh = _file_hash(file_bytes)
    all_sheets = _read_all_sheets(file_bytes, fh)

    sheet = None
    for s in all_sheets:
        if s.upper() in ("NF", "NIFTY", "NIFTY50", "NIFTY 50"):
            sheet = s
            break
    if sheet is None and len(all_sheets) > 1:
        sheet = list(all_sheets.keys())[1]
    if sheet is None:
        return pd.DataFrame()

    df = all_sheets[sheet].copy()
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if "date" in cl or "time" in cl:
            col_map[c] = "date"
        elif cl == "close":
            col_map[c] = "close"
    df = df.rename(columns=col_map)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["date"]  = _parse_date(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df[["close"]]


def _clean_stock(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("date/time", "date", "datetime"):
            col_map[c] = "date"
        elif cl == "close":
            col_map[c] = "close"
        elif cl == "aux2":
            col_map[c] = "aux2"
    df = df.rename(columns=col_map)
    if not {"date", "close", "aux2"}.issubset(df.columns):
        return None
    df = df[["date", "close", "aux2"]].copy()
    df["date"]  = _parse_date(df["date"]).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["aux2"]  = pd.to_numeric(df["aux2"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0].sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df.empty:
        return None
    df["ticker"] = ticker
    return df


@st.cache_data(show_spinner=False)
def load_all_stocks(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    """Load all sheets from Data.xlsx → {ticker: date-indexed DataFrame}."""
    fh = _file_hash(file_bytes)
    all_sheets = _read_all_sheets(file_bytes, fh)
    out = {}
    for sheet, raw in all_sheets.items():
        try:
            cleaned = _clean_stock(raw, sheet)
            if cleaned is not None:
                out[sheet] = cleaned.set_index("date")
        except Exception:
            continue
    return out


def validate_files(dates_bytes: bytes, data_bytes: bytes) -> dict:
    res = {"phases_ok": False, "nifty_ok": False, "stocks_ok": False,
           "n_phases": 0, "n_stocks": 0, "nifty_rows": 0, "date_range": None, "errors": []}
    try:
        p = load_phases(dates_bytes)
        res["phases_ok"] = not p.empty
        res["n_phases"] = len(p)
        if not p.empty:
            start = p["entry_date"].min().strftime("%b %Y")
            end   = p["exit_date"].max().strftime("%b %Y")
            res["date_range"] = f"{start} → {end}"
    except Exception as e:
        res["errors"].append(f"Phases: {e}")
    try:
        n = load_nifty(dates_bytes)
        res["nifty_ok"] = not n.empty
        res["nifty_rows"] = len(n)
    except Exception as e:
        res["errors"].append(f"NIFTY: {e}")
    try:
        s = load_all_stocks(data_bytes)
        res["stocks_ok"] = len(s) > 0
        res["n_stocks"] = len(s)
    except Exception as e:
        res["errors"].append(f"Stocks: {e}")
    return res
