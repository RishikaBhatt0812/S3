"""
S³ Core — ML Trade Filter
=========================
A self-contained gradient-boosting classifier (pure numpy — no scikit-learn,
no new dependencies) that learns, from the strategy's own executed trades,
which trades are likely to be PROFITABLE, and then builds an ML-filtered
portfolio from the approved trades.

Design notes (built the way a quant would):
  • Features are PRE-ENTRY ONLY — nothing that is known only after the trade
    closes (no exit price, realised return, alpha, days-held, NIFTY exit …).
    Using outcome columns as features would leak the answer.
  • The split is TIME-AWARE and done by WINDOW, never by row: earlier windows
    train the model, later windows test it. This mimics live deployment and
    prevents same-window leakage between train and test.
  • "Best settings" are chosen on a VALIDATION slice and only then reported on
    a held-out TEST slice, so the reported numbers are out-of-sample.
  • The score that drives the search is the out-of-sample CAR/MDD (Calmar) of
    the resulting portfolio, with profitable-trade rate as a tie-breaker.

Everything here is numpy/pandas only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.investment_analysis import compute_investment_analysis

# XGBoost is the primary model. The pure-numpy gradient booster below is kept
# only as a fallback so the ML tab never hard-crashes if xgboost is missing.
try:
    import xgboost as xgb  # noqa
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ─────────────────────────────────────────────────────────────────────────────
# Technical feature engineering  (PRE-ENTRY information only)
# ─────────────────────────────────────────────────────────────────────────────

def _trailing(stock: pd.DataFrame, before: pd.Timestamp, n: int) -> np.ndarray:
    """Last n closes strictly before `before`."""
    if stock is None:
        return np.array([])
    sub = stock[stock.index < before]
    if sub.empty:
        return np.array([])
    return sub["close"].tail(n).to_numpy(dtype=float)


def _ret(arr: np.ndarray) -> float:
    if arr.size < 2 or arr[0] <= 0:
        return 0.0
    return float((arr[-1] / arr[0] - 1.0) * 100.0)


def _vol(arr: np.ndarray) -> float:
    if arr.size < 3:
        return 0.0
    dr = np.diff(arr) / arr[:-1]
    return float(np.std(dr) * 100.0)


def _ma_dist(price: float, arr: np.ndarray) -> float:
    if arr.size < 2 or price <= 0:
        return 0.0
    ma = float(np.mean(arr))
    return float((price / ma - 1.0) * 100.0) if ma > 0 else 0.0


def _rsi(arr: np.ndarray, period: int = 14) -> float:
    if arr.size < period + 1:
        return 50.0
    d = np.diff(arr[-(period + 1):])
    gain = np.mean(np.clip(d, 0, None))
    loss = np.mean(np.clip(-d, 0, None))
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100.0 - 100.0 / (1.0 + rs))


FEATURE_NAMES = [
    "entry_thr_pct", "exit_thr_pct", "nifty_entry_premium", "entry_latency_days",
    "buy_phase_len_days", "month_sin", "month_cos", "window_vol",
    "stk_mom_21", "stk_mom_63", "stk_vol_21", "stk_dist_ma50", "stk_dist_ma200",
    "stk_rsi14", "nifty_mom_21", "nifty_vol_21", "rel_strength_63",
]


def engineer_features(per_trade_df: pd.DataFrame, stock_dict: dict,
                      nifty_df: pd.DataFrame, target: str = "profit"):
    """
    Build the pre-entry feature matrix.

    Returns (X_df, y, meta_df) or (None, None, None) if unusable.
        X_df  : columns == FEATURE_NAMES
        y     : 1 = profitable (per `target`), 0 otherwise
        meta  : window_idx, ticker, entry_date, exit_date, return_pct, alpha, days_held
    """
    if per_trade_df is None or per_trade_df.empty:
        return None, None, None
    df = per_trade_df.copy()
    for c in ("entry_date", "exit_date", "pattern_start", "buy_phase_start", "buy_phase_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["entry_date", "return_pct"])
    if df.empty:
        return None, None, None

    nf = nifty_df if (nifty_df is not None and not nifty_df.empty) else None

    rows, meta_rows, y = [], [], []
    for _, r in df.iterrows():
        edate = r["entry_date"]
        tkr = str(r.get("ticker", ""))
        sdf = stock_dict.get(tkr) if stock_dict else None

        base = float(r.get("nifty_buy_base") or 0) or np.nan
        at_entry = float(r.get("nifty_at_entry") or 0) or np.nan
        nifty_entry_premium = ((at_entry - base) / base * 100.0) if (base and base > 0 and not np.isnan(at_entry)) else 0.0

        p_start = r.get("pattern_start")
        entry_latency = (edate - p_start).days if pd.notna(p_start) else 0
        bp_s, bp_e = r.get("buy_phase_start"), r.get("buy_phase_end")
        buy_len = (bp_e - bp_s).days if pd.notna(bp_s) and pd.notna(bp_e) else 0

        mon = edate.month
        month_sin = np.sin(2 * np.pi * mon / 12.0)
        month_cos = np.cos(2 * np.pi * mon / 12.0)

        # window volatility (pattern start → entry)
        wv = 0.0
        if sdf is not None and pd.notna(p_start):
            sub = sdf[(sdf.index >= p_start) & (sdf.index <= edate)]
            if len(sub) >= 3:
                wv = _vol(sub["close"].to_numpy(dtype=float))

        a21 = _trailing(sdf, edate, 21)
        a63 = _trailing(sdf, edate, 63)
        a50 = _trailing(sdf, edate, 50)
        a200 = _trailing(sdf, edate, 200)
        entry_px = float(r.get("entry_price") or 0)
        stk_mom_21 = _ret(a21)
        stk_mom_63 = _ret(a63)
        stk_vol_21 = _vol(a21)
        stk_dist_ma50 = _ma_dist(entry_px, a50)
        stk_dist_ma200 = _ma_dist(entry_px, a200)
        stk_rsi14 = _rsi(a63 if a63.size else a21, 14)

        if nf is not None:
            n21 = _trailing(nf, edate, 21)
            n63 = _trailing(nf, edate, 63)
            nifty_mom_21 = _ret(n21)
            nifty_vol_21 = _vol(n21)
            rel_strength_63 = stk_mom_63 - _ret(n63)
        else:
            nifty_mom_21 = nifty_vol_21 = rel_strength_63 = 0.0

        rows.append([
            float(r.get("entry_threshold_pct") or 0),
            float(r.get("exit_threshold_pct") or 0),
            nifty_entry_premium, float(entry_latency), float(buy_len),
            month_sin, month_cos, wv,
            stk_mom_21, stk_mom_63, stk_vol_21, stk_dist_ma50, stk_dist_ma200,
            stk_rsi14, nifty_mom_21, nifty_vol_21, rel_strength_63,
        ])

        ret = float(r["return_pct"])
        alpha = float(r.get("alpha") or 0)
        label = (alpha > 0) if target == "alpha" else (ret > 0)
        y.append(1 if label else 0)
        meta_rows.append({
            "window_idx": int(r.get("window_idx", 0)),
            "ticker": tkr, "entry_date": edate, "exit_date": r.get("exit_date"),
            "return_pct": ret, "alpha": alpha,
            "days_held": float(r.get("days_held") or np.nan),
        })

    X = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0.0)
    return X, np.array(y, dtype=int), pd.DataFrame(meta_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient-boosted regression trees  (numpy)
# ─────────────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")
    def __init__(self):
        self.feat = -1; self.thr = 0.0; self.left = None; self.right = None; self.value = 0.0


class _Tree:
    """Depth-limited CART regressor minimising squared error."""
    def __init__(self, max_depth=3, min_leaf=4, max_thr=24, rng=None):
        self.max_depth = max_depth; self.min_leaf = min_leaf
        self.max_thr = max_thr; self.rng = rng or np.random.default_rng(0)
        self.root = None; self.importances = None

    def fit(self, X, g, n_features):
        self.importances = np.zeros(n_features)
        self.root = self._build(X, g, 0)
        return self

    def _build(self, X, g, depth):
        node = _Node()
        node.value = float(np.mean(g)) if g.size else 0.0
        if depth >= self.max_depth or g.size < 2 * self.min_leaf or np.allclose(g, g[0]):
            return node
        best = self._best_split(X, g)
        if best is None:
            return node
        feat, thr, gain, mask = best
        if mask.sum() < self.min_leaf or (~mask).sum() < self.min_leaf:
            return node
        node.feat, node.thr = feat, thr
        self.importances[feat] += gain
        node.left = self._build(X[mask], g[mask], depth + 1)
        node.right = self._build(X[~mask], g[~mask], depth + 1)
        return node

    def _best_split(self, X, g):
        n, m = X.shape
        sse_parent = float(np.sum((g - g.mean()) ** 2))
        best_gain, best = 0.0, None
        for f in range(m):
            col = X[:, f]
            uniq = np.unique(col)
            if uniq.size < 2:
                continue
            if uniq.size > self.max_thr:
                qs = np.linspace(0, 100, self.max_thr)
                cand = np.unique(np.percentile(col, qs))
            else:
                cand = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in cand:
                mask = col <= thr
                nl = mask.sum()
                if nl < self.min_leaf or n - nl < self.min_leaf:
                    continue
                gl, gr = g[mask], g[~mask]
                sse = float(np.sum((gl - gl.mean()) ** 2) + np.sum((gr - gr.mean()) ** 2))
                gain = sse_parent - sse
                if gain > best_gain:
                    best_gain, best = gain, (f, float(thr), gain, mask)
        return best

    def predict(self, X):
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            node = self.root
            while node.feat >= 0:
                node = node.left if X[i, node.feat] <= node.thr else node.right
            out[i] = node.value
        return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GradientBoostClassifier:
    """Binary log-loss gradient boosting on shallow regression trees."""
    def __init__(self, n_estimators=120, learning_rate=0.08, max_depth=3,
                 min_leaf=4, subsample=0.85, seed=42):
        self.n_estimators = n_estimators; self.learning_rate = learning_rate
        self.max_depth = max_depth; self.min_leaf = min_leaf
        self.subsample = subsample; self.seed = seed
        self.trees = []; self.F0 = 0.0; self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        n, m = X.shape
        rng = np.random.default_rng(self.seed)
        p = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        self.F0 = float(np.log(p / (1 - p)))
        F = np.full(n, self.F0)
        imp = np.zeros(m)
        self.trees = []
        for _ in range(self.n_estimators):
            prob = _sigmoid(F)
            grad = y - prob                      # negative gradient of log-loss
            if self.subsample < 1.0:
                k = max(2 * self.min_leaf, int(self.subsample * n))
                idx = rng.choice(n, size=k, replace=False)
            else:
                idx = np.arange(n)
            t = _Tree(self.max_depth, self.min_leaf, rng=rng).fit(X[idx], grad[idx], m)
            F += self.learning_rate * t.predict(X)
            self.trees.append(t)
            imp += t.importances
        s = imp.sum()
        self.feature_importances_ = imp / s if s > 0 else imp
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.F0)
        for t in self.trees:
            F += self.learning_rate * t.predict(X)
        return F

    def predict_proba(self, X):
        return _sigmoid(self.decision_function(X))


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost adapter + model factory
# ─────────────────────────────────────────────────────────────────────────────

class _XGBModel:
    """Thin wrapper giving XGBClassifier the same interface as the numpy model:
    .fit(X, y) · .predict_proba(X) → 1-D positive-class probability ·
    .feature_importances_."""
    def __init__(self, n_estimators=120, learning_rate=0.08, max_depth=3,
                 subsample=0.85, seed=42, scale_pos_weight=1.0):
        self.clf = xgb.XGBClassifier(
            n_estimators=int(n_estimators), learning_rate=float(learning_rate),
            max_depth=int(max_depth), subsample=float(subsample),
            colsample_bytree=0.85, reg_lambda=1.0, min_child_weight=2.0, gamma=0.0,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=2, random_state=int(seed),
            scale_pos_weight=float(scale_pos_weight), verbosity=0)
        self.feature_importances_ = None

    def fit(self, X, y):
        self.clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        self.feature_importances_ = np.asarray(self.clf.feature_importances_, dtype=float)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X, dtype=float))[:, 1]


def ENGINE_NAME() -> str:
    return "XGBoost" if _HAS_XGB else "built-in gradient boosting (numpy)"


def _new_model(n_estimators, learning_rate, max_depth, seed, scale_pos_weight=1.0):
    """Return a fitted-interface model — XGBoost when available, else numpy GBM."""
    if _HAS_XGB:
        return _XGBModel(n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, seed=seed, scale_pos_weight=scale_pos_weight)
    return GradientBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                   max_depth=max_depth, seed=seed)


def _pos_weight(y) -> float:
    """neg/pos ratio (clamped) to counter class imbalance in 'profitable' labels."""
    y = np.asarray(y)
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return float(np.clip(neg / pos, 0.25, 4.0))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    r_pos = ranks[y == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def classification_metrics(y, p, thr=0.5) -> dict:
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0) else float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": _auc(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# Time-aware split (by window)
# ─────────────────────────────────────────────────────────────────────────────

def time_window_split(meta: pd.DataFrame, train=0.6, valid=0.2):
    """Chronological split BY WINDOW → boolean masks (train, valid, test)."""
    order = (meta.assign(_r=range(len(meta)))
                 .sort_values("entry_date")["window_idx"])
    wins = list(dict.fromkeys(order.tolist()))   # unique windows, chronological
    nw = len(wins)
    n_tr = max(1, int(round(train * nw)))
    n_va = int(round(valid * nw))
    if n_tr + n_va >= nw:               # ensure at least 1 test window
        n_va = max(0, nw - n_tr - 1)
    tr_w = set(wins[:n_tr]); va_w = set(wins[n_tr:n_tr + n_va]); te_w = set(wins[n_tr + n_va:])
    w = meta["window_idx"]
    return (w.isin(tr_w).to_numpy(), w.isin(va_w).to_numpy(), w.isin(te_w).to_numpy(),
            {"n_windows": nw, "train_w": len(tr_w), "valid_w": len(va_w), "test_w": len(te_w)})


def _portfolio_score(meta_sub: pd.DataFrame, initial=100_000.0, reinvest=False) -> dict:
    """Run investment analysis on a meta subset that mimics per_trade_df."""
    if meta_sub is None or meta_sub.empty:
        return {"calmar": float("nan"), "cagr": float("nan"), "mdd": float("nan"),
                "final": initial, "profit_pct": float("nan"), "n": 0, "win_rate": float("nan")}
    ptd = meta_sub.rename(columns={}).copy()
    ia = compute_investment_analysis(ptd, initial_capital=initial,
                                     alloc_mode="equal", reinvest=reinvest)
    if ia is None:
        wr = float((meta_sub["return_pct"] > 0).mean() * 100.0)
        return {"calmar": float("nan"), "cagr": float("nan"), "mdd": float("nan"),
                "final": initial, "profit_pct": float("nan"),
                "n": len(meta_sub), "win_rate": wr, "ia": None}
    m = ia["metrics"]
    return {"calmar": m["calmar"], "cagr": m["cagr"], "mdd": m["mdd_pct"],
            "final": m["final_equity"], "profit_pct": m["total_pl_pct"],
            "n": m["n_trades"], "win_rate": m["win_rate"], "ia": ia}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — train, search best settings, build ML portfolio
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GRID = {
    "n_estimators":  [80, 150],
    "learning_rate": [0.05, 0.10],
    "max_depth":     [2, 3],
}
DEFAULT_THRESHOLDS = [0.45, 0.50, 0.55, 0.60, 0.65]


def run_ml(per_trade_df: pd.DataFrame, stock_dict: dict, nifty_df: pd.DataFrame,
           target: str = "profit", initial_capital: float = 100_000.0,
           reinvest: bool = False, grid: dict | None = None,
           thresholds=None, train_frac: float = 0.6, valid_frac: float = 0.2,
           seed: int = 42) -> dict:
    """
    Full pipeline. Returns a dict consumed by the Streamlit ML tab.
    Keys: ok, msg, split, best, leaderboard, test_metrics, importances,
          ml_portfolio (ia), baseline_portfolio (ia), full_ml_portfolio (ia),
          approved_mask, feature_names, proba (test).
    """
    grid = grid or DEFAULT_GRID
    thresholds = thresholds or DEFAULT_THRESHOLDS

    X, y, meta = engineer_features(per_trade_df, stock_dict, nifty_df, target=target)
    if X is None or len(X) < 12:
        return {"ok": False, "msg": "Need at least ~12 executed trades to train a model."}
    if len(np.unique(y)) < 2:
        return {"ok": False, "msg": "All trades have the same outcome — nothing to learn. "
                                    "Loosen the strategy so it produces both winners and losers."}

    tr, va, te, split = time_window_split(meta, train_frac, valid_frac)
    if tr.sum() < 6 or te.sum() < 2:
        return {"ok": False, "msg": f"Not enough windows for a time-aware split "
                                    f"(have {split['n_windows']}). Need more trade windows."}
    # If no validation windows, select on test (with caveat) by reusing test as valid.
    sel = va if va.sum() >= 1 else te
    sel_is_test = va.sum() < 1

    Xv = X.to_numpy(dtype=float)
    spw = _pos_weight(y[tr])

    # ── Grid + threshold search on the selection slice ────────────────────────
    leaderboard = []
    best = None
    combos = [(ne, lr, md) for ne in grid["n_estimators"]
              for lr in grid["learning_rate"] for md in grid["max_depth"]]
    for (ne, lr, md) in combos:
        model = _new_model(ne, lr, md, seed, scale_pos_weight=spw).fit(Xv[tr], y[tr])
        p_sel = model.predict_proba(Xv[sel])
        sel_auc = _auc(y[sel], p_sel)
        for thr in thresholds:
            approve = p_sel >= thr
            sub = meta[sel].iloc[approve]
            sc = _portfolio_score(sub, initial_capital, reinvest)
            cm = classification_metrics(y[sel], p_sel, thr)
            # objective: out-of-sample Calmar, then profitable %, then coverage
            calmar = sc["calmar"]
            obj = (calmar if (calmar is not None and not np.isnan(calmar)) else -1e9)
            row = {"n_estimators": ne, "learning_rate": lr, "max_depth": md,
                   "threshold": thr, "sel_auc": sel_auc, "sel_acc": cm["accuracy"],
                   "trades_kept": int(approve.sum()), "sel_calmar": calmar,
                   "sel_cagr": sc["cagr"], "sel_mdd": sc["mdd"],
                   "sel_profit_pct": sc["profit_pct"], "sel_win_rate": sc["win_rate"],
                   "_obj": obj, "_winrate": sc["win_rate"] if not np.isnan(sc["win_rate"]) else -1}
            leaderboard.append(row)
            if best is None or (row["_obj"], row["_winrate"]) > (best["_obj"], best["_winrate"]):
                best = row

    if best is None:
        return {"ok": False, "msg": "Search produced no usable configuration."}

    # ── Refit best on TRAIN(+VALID) and report on TEST (held-out) ─────────────
    fit_mask = tr | (va if not sel_is_test else np.zeros_like(tr))
    final_model = _new_model(
        best["n_estimators"], best["learning_rate"], best["max_depth"],
        seed, scale_pos_weight=_pos_weight(y[fit_mask])).fit(Xv[fit_mask], y[fit_mask])

    p_test = final_model.predict_proba(Xv[te])
    thr = best["threshold"]
    test_metrics = classification_metrics(y[te], p_test, thr)

    approved = p_test >= thr
    ml_sub = meta[te].iloc[approved]
    base_sub = meta[te]
    ml_port = _portfolio_score(ml_sub, initial_capital, reinvest)
    base_port = _portfolio_score(base_sub, initial_capital, reinvest)

    # full-period ML-filtered portfolio (in+out of sample) for an overall view
    p_all = final_model.predict_proba(Xv)
    full_sub = meta.iloc[p_all >= thr]
    full_port = _portfolio_score(full_sub, initial_capital, reinvest)

    imp = sorted(zip(FEATURE_NAMES, final_model.feature_importances_),
                 key=lambda kv: kv[1], reverse=True)

    lb_df = pd.DataFrame(leaderboard).drop(columns=["_obj", "_winrate"]) \
              .sort_values(["sel_calmar", "sel_profit_pct"], ascending=False, na_position="last") \
              .reset_index(drop=True)

    return {
        "ok": True, "msg": "", "split": split, "sel_is_test": sel_is_test,
        "engine": ENGINE_NAME(),
        "best": {k: best[k] for k in ("n_estimators", "learning_rate", "max_depth",
                                      "threshold", "sel_auc", "sel_calmar",
                                      "sel_cagr", "sel_mdd", "sel_profit_pct",
                                      "sel_win_rate", "trades_kept")},
        "leaderboard": lb_df, "test_metrics": test_metrics,
        "importances": imp, "feature_names": FEATURE_NAMES,
        "ml_portfolio": ml_port, "baseline_portfolio": base_port,
        "full_ml_portfolio": full_port,
        "n_test_trades": int(te.sum()), "n_train_trades": int(tr.sum()),
        "test_proba": p_test, "target": target,
    }
