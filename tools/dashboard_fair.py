#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

is_demo = os.environ.get("RUN_MODE", "Normal") == "Demography"
if is_demo:
    OUT_DIR = f"D:/Bob_Skripsi_Do Not Delete/Analysis/Demography/{sys.argv[1]}_percent"
else:
    OUT_DIR = f"D:/Bob_Skripsi_Do Not Delete/Analysis/Normal/{sys.argv[1]}_percent"


METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
PARETO_COLORS = {
    "Ye_ApxI": "#16a34a", 
    "Ye_multi": "#d97706", 
    "New_True_inf": "#2563eb", 
    "New_Max": "#dc2626"
}

# Mapping for the side-by-side comparison chart
METHOD_DISPLAY_NAMES = {
    "Ye_ApxI": "Metode Terdahulu<br>(Single Env)",
    "Ye_multi": "Metode Terdahulu<br>(Multi Env)",
    "New_True_inf": "Metode yang<br>Diusulkan",
    "New_Max": "Metode yang<br>Diusulkan (Max)"
}

TOP_SELECTION_METRIC = "base_retain_Hit"

# =====================================================================
# CUSTOM UTILITY SCORE (Used when mode == 'fair')
# =====================================================================
def custom_utility_score(forget_drop_rel, retain_drop_rel):
    scaling_factor = 5.0
    if retain_drop_rel <= 0:
        penalty = 0
    else:
        penalty = (retain_drop_rel ** 2) / scaling_factor
    score = forget_drop_rel - penalty
    return score
# =====================================================================


def parse_args():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python dashboard.py <forget_percentage> [retain_drop_threshold] [K] [mode] [num_top_models]\n"
            "Modes: standard (default) or fair\n"
            "Example: python dashboard.py 20 0.01 10 standard 5"
        )
    forget_percentage = int(sys.argv[1])
    retain_drop_threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.01
    k_val = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    mode = sys.argv[4].lower() if len(sys.argv) >= 5 else "standard"
    num_top_models = int(sys.argv[5]) if len(sys.argv) >= 6 else None
    
    if mode not in ["standard", "fair"]:
        mode = "standard"
        
    return forget_percentage, retain_drop_threshold, k_val, mode, num_top_models


def pct(x):
    if pd.isna(x): return "N/A"
    return f"{100.0 * float(x):.2f}%"

def pp(x):
    if pd.isna(x): return "N/A"
    return f"{100.0 * float(x):.2f} pp"

def normalize_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def signed_log10(x):
    if pd.isna(x): return np.nan
    x = float(x)
    return np.sign(x) * np.log10(1.0 + abs(x))

def signed_log_label():
    return "sign(x) · log10(1 + |x|)"

def find_loss_log_path(results_base: Path):
    for name in ["unlearning_loss_log.csv", "unlearn_loss_log.csv", "loss_log.csv"]:
        p = results_base / name
        if p.exists(): return p
    return None

def load_inputs(results_base: Path):
    results_path = results_base / "tuning_full_results.csv"
    train_results_path = results_base / "train_phase_results.csv"
    loss_log_path = find_loss_log_path(results_base)
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")
    results_df = pd.read_csv(results_path)
    train_df = pd.read_csv(train_results_path) if train_results_path.exists() else pd.DataFrame()
    loss_df = pd.read_csv(loss_log_path) if loss_log_path is not None else pd.DataFrame()
    return results_df, train_df, loss_df, results_path, loss_log_path

def prepare_results(df: pd.DataFrame):
    numeric_cols = [
        "K", "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain",
        "unlearn_time_s", "train_time_s",
        "base_retain_Hit", "base_retain_NDCG",
        "base_forget_Hit", "base_forget_NDCG",
        "retain_Hit", "retain_NDCG",
        "forget_Hit", "forget_NDCG",
        "loss_forget_final", "loss_retain_final", "loss_total_final",
        "fq_hit", "fq_ndcg",
    ]
    df = normalize_numeric(df.copy(), numeric_cols)
    if "method" not in df.columns:
        raise ValueError("tuning_full_results.csv is missing the 'method' column.")
    df = df[df["method"].isin(METHODS)].copy()
    
    df["retain_drop_hit"]  = df["base_retain_Hit"]  - df["retain_Hit"]
    df["forget_drop_hit"]  = df["base_forget_Hit"]  - df["forget_Hit"]
    df["retain_drop_ndcg"] = df["base_retain_NDCG"] - df["retain_NDCG"]
    df["forget_drop_ndcg"] = df["base_forget_NDCG"] - df["forget_NDCG"]
    
    df["retain_drop_hit_rel"] = np.where(df["base_retain_Hit"] > 0, (df["retain_drop_hit"] / df["base_retain_Hit"]) * 100, 0)
    df["forget_drop_hit_rel"] = np.where(df["base_forget_Hit"] > 0, (df["forget_drop_hit"] / df["base_forget_Hit"]) * 100, 0)
    df["retain_drop_ndcg_rel"] = np.where(df["base_retain_NDCG"] > 0, (df["retain_drop_ndcg"] / df["base_retain_NDCG"]) * 100, 0)
    df["forget_drop_ndcg_rel"] = np.where(df["base_forget_NDCG"] > 0, (df["forget_drop_ndcg"] / df["base_forget_NDCG"]) * 100, 0)
    return df

def build_aggregate_data(df: pd.DataFrame, k_val: int) -> dict:
    df_k = df[(df["K"] == k_val)].copy()
    agg = {}
    for method in METHODS:
        mdf = df_k[df_k["method"] == method]
        if mdf.empty: continue
        agg[method] = {"n": int(len(mdf))}
        for col in ["retain_drop_hit_rel", "forget_drop_hit_rel"]:
            vals = mdf[col].dropna() if col in mdf.columns else pd.Series(dtype=float)
            if len(vals) == 0:
                agg[method][col] = {"mean": None, "median": None, "q1": None, "q3": None}
            else:
                agg[method][col] = {
                    "mean":   round(float(vals.mean()), 3),
                    "median": round(float(vals.median()), 3),
                    "q1":     round(float(vals.quantile(0.25)), 3),
                    "q3":     round(float(vals.quantile(0.75)), 3),
                }
    return agg

def select_one_per_method(df: pd.DataFrame, k_val: int, retain_drop_threshold: float, mode: str):
    df_k = df[df["K"] == k_val].copy()
    df_k["utility_score"] = df_k.apply(
        lambda r: custom_utility_score(r["forget_drop_hit_rel"], r["retain_drop_hit_rel"]), axis=1
    )
    selected_rows, selection_notes = [], []
    
    for method in METHODS:
        mdf = df_k[df_k["method"] == method].copy()
        if mdf.empty:
            selection_notes.append(f"{method}: no rows found at K={k_val}.")
            continue
            
        if mode == "fair":
            mdf = mdf.sort_values(by="utility_score", ascending=False, na_position="last")
            row = mdf.iloc[0].copy()
            row["selection_note"] = f"Selected via fair Utility Score: {row['utility_score']:.2f}. Hard threshold bypassed."
            selected_rows.append(row)
            selection_notes.append(f"{method}: {row['selection_note']}")
        else: 
            constrained = mdf[mdf["retain_drop_hit"] < retain_drop_threshold].copy()
            if constrained.empty:
                selection_notes.append(f"{method}: no run satisfied retain_drop < {retain_drop_threshold} at K={k_val}.")
                continue
            constrained = constrained.sort_values(by="forget_drop_hit", ascending=False, na_position="last")
            row = constrained.iloc[0].copy()
            row["selection_note"] = f"Selected from runs with retain_drop &lt; {retain_drop_threshold}, then maximized forget_drop."
            selected_rows.append(row)
            selection_notes.append(f"{method}: Selected standard hard-threshold.")
            
    if not selected_rows:
        return pd.DataFrame(), selection_notes
    return pd.DataFrame(selected_rows).reset_index(drop=True), selection_notes


def metric_flags_extended(before, after, rel_drop, higher_is_better=True):
    if pd.isna(before) or pd.isna(after):
        return "neutral", "N/A", "N/A", "•"
    delta = float(after) - float(before)
    if abs(delta) < 1e-12:
        return "neutral", "0.00 pp", "0.00% rel.", "•"
    good = (delta > 0) if higher_is_better else (delta < 0)
    css = "good" if good else "bad"
    arrow = "▲" if delta > 0 else "▼"
    pp_str = f"{abs(delta)*100:.2f} pp"
    rel_str = f"{abs(rel_drop):.2f}% rel."
    return css, pp_str, rel_str, arrow


def build_aggregate_section(agg: dict, k_val: int) -> str:
    cards = []
    for method in METHODS:
        if method not in agg:
            cards.append(f'<div class="agg-card"><div class="agg-method-name">{method}</div><div class="agg-n">No data</div></div>')
            continue
        d = agg[method]
        n = d["n"]

        def sv(col, stat):
            v = d.get(col, {}).get(stat)
            return str(v) if v is not None else "null"
        def fv(col, stat="mean"):
            v = d.get(col, {}).get(stat)
            return f"{v:.2f}%" if v is not None else "N/A"

        cards.append(f"""
        <div class="agg-card">
            <div class="agg-method-name">{method}</div>
            <div class="agg-n">{n} runs &nbsp;·&nbsp; K={k_val}</div>
            <div class="agg-divider"></div>
            <div class="agg-metric-row">
                <span class="agg-label">Rel Retain Drop</span>
                <span class="agg-stat retain-col" data-mean="{sv('retain_drop_hit_rel','mean')}" data-median="{sv('retain_drop_hit_rel','median')}" data-q1="{sv('retain_drop_hit_rel','q1')}" data-q3="{sv('retain_drop_hit_rel','q3')}">{fv('retain_drop_hit_rel')}</span>
            </div>
            <div class="agg-metric-row">
                <span class="agg-label">Rel Forget Drop</span>
                <span class="agg-stat forget-col" data-mean="{sv('forget_drop_hit_rel','mean')}" data-median="{sv('forget_drop_hit_rel','median')}" data-q1="{sv('forget_drop_hit_rel','q1')}" data-q3="{sv('forget_drop_hit_rel','q3')}">{fv('forget_drop_hit_rel')}</span>
            </div>
        </div>""")

    return f"""
    <div class="section">
        <div class="header-row">
            <h2>Aggregate Relative Performance <span class="sub-label">Filtered subset runs &nbsp;·&nbsp; K={k_val}</span></h2>
            <div class="toggle-group">
                <button class="tgl active" data-stat="mean" onclick="updateAgg('mean')">Mean</button>
                <button class="tgl" data-stat="median" onclick="updateAgg('median')">Median</button>
                <button class="tgl" data-stat="q1" onclick="updateAgg('q1')">Q1 (25th)</button>
                <button class="tgl" data-stat="q3" onclick="updateAgg('q3')">Q3 (75th)</button>
            </div>
        </div>
        <div class="agg-grid">{''.join(cards)}</div>
    </div>"""


def build_method_rows_html(selected_df: pd.DataFrame, k_val: int) -> str:
    rows = []
    for _, r in selected_df.iterrows():
        r_css, r_pp, r_rel, r_a = metric_flags_extended(r["base_retain_Hit"],  r["retain_Hit"], r["retain_drop_hit_rel"], True)
        f_css, f_pp, f_rel, f_a = metric_flags_extended(r["base_forget_Hit"],  r["forget_Hit"], r["forget_drop_hit_rel"], False)
        rn_css,rn_pp,rn_rel,rn_a= metric_flags_extended(r["base_retain_NDCG"], r["retain_NDCG"], r["retain_drop_ndcg_rel"], True)
        fn_css,fn_pp,fn_rel,fn_a= metric_flags_extended(r["base_forget_NDCG"], r["forget_NDCG"], r["forget_drop_ndcg_rel"], False)
        
        rows.append(f"""
        <div class="method-row">
            <div class="method-title-block">
                <div class="method-name">{r['method']} <span style="font-size:16px; color:#64748b; font-weight:600;">| Score: {r['utility_score']:.2f}</span></div>
                <div class="method-sub">
                    λ = {r.get('lambda_retain', np.nan)} &nbsp;·&nbsp;
                    u_lr = {r.get('unlearn_lr', np.nan)} &nbsp;·&nbsp;
                    u_iters = {int(r['unlearn_iters']) if pd.notna(r['unlearn_iters']) else 'N/A'} &nbsp;·&nbsp;
                    train_lr = {r.get('train_lr', np.nan)} &nbsp;·&nbsp;
                    γ = {r.get('gamma', np.nan)} &nbsp;·&nbsp;
                    hidden = {int(r['hidden_dim']) if pd.notna(r['hidden_dim']) else 'N/A'} &nbsp;·&nbsp;
                    train_bs = {int(r['train_batch']) if pd.notna(r['train_batch']) else 'N/A'}
                </div>
                <div class="method-note">{r['selection_note']}</div>
            </div>
            <div class="metrics-row">
                <div class="metric-card big">
                    <div class="metric-label">Retain Hit@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_retain_Hit'])} → {pct(r['retain_Hit'])}</div>
                    <div class="metric-delta {r_css} metric-toggle-target" data-pp="{r_a} {r_pp}" data-rel="{r_a} {r_rel}">{r_a} {r_pp}</div>
                </div>
                <div class="metric-card big">
                    <div class="metric-label">Forget Hit@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_forget_Hit'])} → {pct(r['forget_Hit'])}</div>
                    <div class="metric-delta {f_css} metric-toggle-target" data-pp="{f_a} {f_pp}" data-rel="{f_a} {f_rel}">{f_a} {f_pp}</div>
                </div>
                <div class="metric-card small">
                    <div class="metric-label">Retain NDCG@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_retain_NDCG'])} → {pct(r['retain_NDCG'])}</div>
                    <div class="metric-delta {rn_css} metric-toggle-target" data-pp="{rn_a} {rn_pp}" data-rel="{rn_a} {rn_rel}">{rn_a} {rn_pp}</div>
                </div>
                <div class="metric-card small">
                    <div class="metric-label">Forget NDCG@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_forget_NDCG'])} → {pct(r['forget_NDCG'])}</div>
                    <div class="metric-delta {fn_css} metric-toggle-target" data-pp="{fn_a} {fn_pp}" data-rel="{fn_a} {fn_rel}">{fn_a} {fn_pp}</div>
                </div>
            </div>
        </div>""")
    return "\n".join(rows)

def build_summary_table(selected_df: pd.DataFrame, k_val: int) -> str:
    rows = []
    for _, r in selected_df.iterrows():
        r_css, r_pp, _, r_a = metric_flags_extended(r["base_retain_Hit"], r["retain_Hit"], 0, True)
        f_css, f_pp, _, f_a = metric_flags_extended(r["base_forget_Hit"], r["forget_Hit"], 0, False)
        
        r_rel_color = "#991b1b" if r['retain_drop_hit_rel'] > 0 else "#065f46"
        f_rel_color = "#065f46" if r['forget_drop_hit_rel'] > 0 else "#991b1b"
        
        rows.append(f"""
        <tr>
            <td><b>{r['method']}</b></td>
            <td>{r['utility_score']:.2f}</td>
            <td>{pct(r['base_retain_Hit'])}</td>
            <td>{pct(r['retain_Hit'])}</td>
            <td><span class="{r_css}">{r_a} {r_pp}</span></td>
            <td>{pct(r['base_forget_Hit'])}</td>
            <td>{pct(r['forget_Hit'])}</td>
            <td><span class="{f_css}">{f_a} {f_pp}</span></td>
            <td style="color: {r_rel_color}; font-weight: 600;">{r['retain_drop_hit_rel']:.2f}%</td>
            <td style="color: {f_rel_color}; font-weight: 600;">{r['forget_drop_hit_rel']:.2f}%</td>
            <td>{r.get('lambda_retain', np.nan)}</td>
            <td>{r.get('unlearn_lr', np.nan)}</td>
        </tr>""")
    return f"""
    <table class="styled-table">
        <thead><tr>
            <th>Method</th><th>Score</th>
            <th>Base Retain@{k_val}</th><th>After Retain</th><th>Retain Δ</th>
            <th>Base Forget@{k_val}</th><th>After Forget</th><th>Forget Δ</th>
            <th>Rel Retain Drop</th><th>Rel Forget Drop</th>
            <th>λ</th><th>u_lr</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_pareto_figure(df: pd.DataFrame, k_val: int):
    df_k = df[df["K"] == k_val].copy()
    if df_k.empty: return None
        
    fig = go.Figure()
    
    for method in METHODS:
        mdf = df_k[df_k["method"] == method].dropna(subset=["retain_drop_hit_rel", "forget_drop_hit_rel"])
        if mdf.empty: continue
            
        fig.add_trace(go.Scatter(
            x=mdf["retain_drop_hit_rel"],
            y=mdf["forget_drop_hit_rel"],
            mode="markers",
            name=f"{method} (Data)",
            marker=dict(color=PARETO_COLORS.get(method, "#64748b"), opacity=0.115, size=6),
            showlegend=False,
            hovertemplate="Rel Retain Drop: %{x:.2f}%<br>Rel Forget Drop: %{y:.2f}%<extra></extra>"
        ))
        
        mdf_sorted = mdf.sort_values("retain_drop_hit_rel")
        pareto_front = []
        max_f = -np.inf
        for _, r in mdf_sorted.iterrows():
            if r["forget_drop_hit_rel"] > max_f:
                pareto_front.append(r)
                max_f = r["forget_drop_hit_rel"]
                
        if pareto_front:
            pdf = pd.DataFrame(pareto_front)
            fig.add_trace(go.Scatter(
                x=pdf["retain_drop_hit_rel"],
                y=pdf["forget_drop_hit_rel"],
                mode="lines+markers",
                name=f"{method} (Line)",
                line=dict(color=PARETO_COLORS.get(method, "#64748b"), width=3),
                marker=dict(size=8, symbol="diamond"),
                showlegend=False,
                hovertemplate="Rel Retain Drop: %{x:.2f}%<br>Rel Forget Drop: %{y:.2f}%<extra></extra>"
            ))
            
    fig.update_layout(
        title="Pareto Frontier: Retain Drop vs Forget Drop (Relative %)",
        xaxis_title="Relative Retain Drop % (Further Right is Better / Lower Drop)",
        yaxis_title="Relative Forget Drop % (Higher is Better ↑)",
        template="plotly_white",
        height=550,
        margin=dict(l=50, r=20, t=60, b=50),
        showlegend=False,
        font=dict(family="Inter, system-ui, sans-serif"),
        autosize=True
    )
    fig.update_xaxes(autorange="reversed")
    return fig

# =====================================================================
# NEW COMPARISON BAR CHART GENERATOR
# =====================================================================
def build_comparison_figure(selected_df: pd.DataFrame, k_val: int, forget_pct: int):
    if selected_df.empty: return None

    # Retrieve data
    methods_raw = selected_df["method"].tolist()
    # Apply Indonesian mapping for display
    display_methods = [METHOD_DISPLAY_NAMES.get(m, m) for m in methods_raw]
    
    rb = (selected_df["base_retain_Hit"] * 100).round(2).tolist()
    ra = (selected_df["retain_Hit"] * 100).round(2).tolist()
    fb = (selected_df["base_forget_Hit"] * 100).round(2).tolist()
    fa = (selected_df["forget_Hit"] * 100).round(2).tolist()

    r_drop = [round(b - a, 2) for b, a in zip(rb, ra)]
    f_drop = [round(b - a, 2) for b, a in zip(fb, fa)]

    fig = go.Figure()

    # Styling Constants
    c_base  = "#94a3b8"  # Slate 400
    c_ret   = "#3b82f6"  # Blue 500 (Retain)
    c_for   = "#f43f5e"  # Rose 500 (Forget)

    # Retain Before
    fig.add_trace(go.Bar(
        name='Retain Before', x=display_methods, y=rb,
        marker_color=c_base, offsetgroup=0,
        text=[f"Before:<br>{v}%" for v in rb], textposition='inside',
        insidetextanchor='middle'
    ))
    
    # Retain After
    fig.add_trace(go.Bar(
        name='Retain After', x=display_methods, y=ra,
        marker_color=c_ret, offsetgroup=1,
        text=[f"After:<br><b>{v}%</b><br>(-{d} pp)" for v, d in zip(ra, r_drop)], textposition='outside',
    ))

    # Forget Before
    fig.add_trace(go.Bar(
        name='Forget Before', x=display_methods, y=fb,
        marker_color="#cbd5e1", offsetgroup=2,
        text=[f"Before:<br>{v}%" for v in fb], textposition='inside',
        insidetextanchor='middle'
    ))

    # Forget After
    fig.add_trace(go.Bar(
        name='Forget After', x=display_methods, y=fa,
        marker_color=c_for, offsetgroup=3,
        text=[f"After:<br><b>{v}%</b><br>(-{d} pp)" for v, d in zip(fa, f_drop)], textposition='outside',
    ))

    # Layout Customization
    max_y = max(max(rb) if rb else [0], max(fb) if fb else [0]) + 15
    fig.update_layout(
        title=f"Side-by-Side Performance Comparison (Hit@{k_val}) — {forget_pct}% Split",
        template="plotly_white",
        font_family="Inter, system-ui, sans-serif",
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        bargap=0.2, 
        bargroupgap=0.08,
        height=650
    )
    fig.update_yaxes(title_text=f"Hit@{k_val} Score (%)", range=[0, max_y])
    
    return fig


def build_loss_figure(loss_df: pd.DataFrame, selected_df: pd.DataFrame):
    if loss_df.empty: return None

    expected_numeric = [
        "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain", "iter",
        "loss_forget", "loss_retain", "loss_total"
    ]
    existing_numeric = [c for c in expected_numeric if c in loss_df.columns]
    loss_df = normalize_numeric(loss_df.copy(), existing_numeric)

    rows = len(selected_df)
    if rows == 0: return None

    MATCH_STRATEGIES = [
        ["method", "train_lr", "gamma", "hidden_dim", "train_batch",
         "unlearn_lr", "unlearn_iters", "lambda_retain"],
        ["method", "train_lr", "gamma", "hidden_dim", "train_batch",
         "unlearn_lr", "unlearn_iters"],
        ["method", "train_lr", "gamma", "hidden_dim", "train_batch",
         "unlearn_lr"],
        ["method", "train_lr", "gamma", "hidden_dim", "train_batch"],
        ["method", "train_lr", "gamma", "hidden_dim"],
        ["method"],
    ]

    STRATEGY_LABELS = [
        "exact match",
        "closest (lambda relaxed)",
        "closest (iters relaxed)",
        "closest (unlearn_lr + iters relaxed)",
        "closest (train_batch relaxed)",
        "closest (method only — no matching run found)",
    ]

    def try_match(ldf, s, cols):
        mask = pd.Series(True, index=ldf.index)
        for col in cols:
            if col not in ldf.columns:
                continue
            val = s.get(col, np.nan)
            if col == "method":
                mask &= (ldf["method"] == val)
            elif pd.isna(val):
                mask &= ldf[col].isna()
            else:
                mask &= (
                    ldf[col].fillna(-999).astype(float).round(5)
                    == round(float(val), 5)
                )
        return ldf[mask].copy().sort_values("iter")

    subplot_titles = []
    matched_subs   = []

    for _, s in selected_df.iterrows():
        method = s.get("method", "???")
        sub, strategy_label = pd.DataFrame(), ""

        for strategy_cols, label in zip(MATCH_STRATEGIES, STRATEGY_LABELS):
            candidate = try_match(loss_df, s, strategy_cols)
            if not candidate.empty:
                sub            = candidate
                strategy_label = label
                break

        matched_subs.append((sub, strategy_label))
        title = f"{method} — signed-log loss"
        if strategy_label and strategy_label != "exact match":
            title += f"  ⚠ {strategy_label}"
        subplot_titles.append(title)

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=False,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    color_map = {
        "loss_forget": "#dc2626",
        "loss_retain": "#2563eb",
        "loss_total":  "#111827",
    }

    for i, ((_, s), (sub, strategy_label)) in enumerate(
        zip(selected_df.iterrows(), matched_subs), start=1
    ):
        method = s.get("method", "???")

        if sub.empty:
            continue

        for ln in ["loss_forget", "loss_retain", "loss_total"]:
            if ln not in sub.columns:
                continue
            sub[f"{ln}_sl"] = sub[ln].apply(signed_log10)
            fig.add_trace(go.Scatter(
                x=sub["iter"],
                y=sub[f"{ln}_sl"],
                mode="lines+markers",
                name=ln,
                line=dict(color=color_map.get(ln, "#000"), width=2),
                marker=dict(size=5),
                showlegend=(i == 1),
                customdata=np.stack(
                    [sub[ln].fillna(np.nan), sub[f"{ln}_sl"].fillna(np.nan)],
                    axis=1,
                ),
                hovertemplate=(
                    "iter=%{x}<br>raw=%{customdata[0]:.6g}"
                    "<br>signed_log=%{customdata[1]:.4f}<extra></extra>"
                ),
            ), row=i, col=1)

        fig.update_xaxes(title_text="Unlearning iteration", row=i, col=1)
        fig.update_yaxes(title_text=signed_log_label(),      row=i, col=1)

    fig.update_layout(
        title="Loss curves by selected run",
        template="plotly_white",
        height=max(380, 300 * rows),
        margin=dict(l=50, r=40, t=90, b=40),
        font=dict(family="Inter, system-ui, sans-serif"),
        autosize=True,
    )
    return fig

def build_html(
    forget_percentage, retain_drop_threshold, k_val, mode, num_top_models,
    selected_df, selection_notes, agg,
    results_path, loss_log_path, pareto_fig, loss_fig, comp_fig
) -> str:
    method_rows  = build_method_rows_html(selected_df, k_val)
    table_html   = build_summary_table(selected_df, k_val)
    agg_section  = build_aggregate_section(agg, k_val)
    notes_html   = "".join(f"<li>{n}</li>" for n in selection_notes)
    
    pareto_div = plot(pareto_fig, output_type="div", include_plotlyjs="cdn") if pareto_fig is not None else '<div class="empty-box">No data available.</div>'
    loss_div   = plot(loss_fig, output_type="div", include_plotlyjs=False) if loss_fig is not None else '<div class="empty-box">No loss-log matching current runs found.</div>'
    comp_div   = plot(comp_fig, output_type="div", include_plotlyjs=False) if comp_fig is not None else '<div class="empty-box">No comparison data available.</div>'

    pareto_checkboxes = []
    for method in METHODS:
        pareto_checkboxes.append(f"""
        <div class="pareto-sidebar-group">
            <div class="psg-title"><span style="color:{PARETO_COLORS.get(method, '#000')}">●</span> {method}</div>
            <div class="psg-toggles">
                <label><input type="checkbox" checked onchange="toggleTrace('{method} (Data)', this.checked)"> Data</label>
                <label><input type="checkbox" checked onchange="toggleTrace('{method} (Line)', this.checked)"> Line</label>
            </div>
        </div>
        """)
    sidebar_controls_html = f"""<div class="pareto-sidebar-card"><h3>Controls</h3>{''.join(pareto_checkboxes)}</div>"""

    mode_display = 'Standard (Hard Threshold)' if mode == 'standard' else 'Fair (Custom Utility Score)'
    filter_text = f"Top {num_top_models} Base Models (by {TOP_SELECTION_METRIC} at K={k_val})" if num_top_models else "All Base Models (Unfiltered)"

    math_block = r"""
    <div class="math-container" style="text-align: center; font-size: 1.1em; margin: 20px 0; color: #0f172a;">
        $$ \text{Score} = \Delta_{\text{forget}} - \frac{(\max(0, \Delta_{\text{retain}}))^2}{\alpha} $$
    </div>
    """

    math_desc = r"""
    <p style="font-size: 14px; color: #475569; line-height: 1.8;">
        Where:<br>
        • \( \Delta_{\text{forget}} \) is the relative drop in forget performance (%).<br>
        • \( \Delta_{\text{retain}} \) is the relative drop in retain performance (%).<br>
        • \( \alpha \) is the scaling factor (currently set to 10.0), which non-linearly penalizes catastrophic retention degradation.
    </p>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Unlearning Dashboard — {forget_percentage}%</title>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        *, *::before, *::after {{ box-sizing: border-box; }}
        body {{ font-family: 'Inter', system-ui, sans-serif; font-size: 15px; background: #f1f5f9; color: #0f172a; margin: 0; padding: 40px 0 72px; }}
        .container {{ max-width: 1440px; margin: 0 auto; padding: 0 64px; }}
        h1 {{ font-size: 32px; font-weight: 800; letter-spacing: -0.6px; margin: 0 0 6px; }}
        h2 {{ font-size: 20px; font-weight: 700; margin: 0 0 14px; letter-spacing: -0.2px; }}
        .page-subtitle {{ color: #64748b; font-size: 15px; margin: 0 0 30px; }}
        .section, .meta {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,.04); }}
        .meta {{ font-size: 14px; line-height: 2.1; }}
        .meta b {{ color: #334155; }}
        .header-row {{ display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }}
        .header-row h2 {{ margin: 0; }}
        .sub-label {{ font-size: 13px; font-weight: 500; color: #94a3b8; margin-left: 8px; }}
        .toggle-group {{ display: inline-flex; background: #f1f5f9; border-radius: 10px; padding: 3px; gap: 2px; }}
        .tgl, .tgl-best {{ border: none; background: transparent; padding: 7px 18px; border-radius: 8px; font-family: inherit; font-size: 13px; font-weight: 600; color: #64748b; cursor: pointer; transition: all .15s; }}
        .tgl:hover, .tgl-best:hover {{ background: #e2e8f0; color: #1e293b; }}
        .tgl.active, .tgl-best.active {{ background: #fff; color: #1e293b; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
        .agg-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
        .agg-card {{ border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; background: #f8fafc; }}
        .agg-method-name {{ font-size: 16px; font-weight: 800; margin-bottom: 3px; }}
        .agg-n {{ font-size: 12px; color: #94a3b8; margin-bottom: 14px; }}
        .agg-divider {{ border-top: 1px solid #e2e8f0; margin-bottom: 12px; }}
        .agg-metric-row {{ display: flex; justify-content: space-between; align-items: center; padding: 6px 0; }}
        .agg-label {{ font-size: 13px; color: #64748b; }}
        .agg-stat {{ font-size: 16px; font-weight: 700; }}
        .retain-col {{ color: #b45309; }}
        .forget-col {{ color: #1d4ed8; }}
        .method-row {{ border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 14px; background: #fff; }}
        .method-title-block {{ margin-bottom: 14px; }}
        .method-name {{ font-size: 23px; font-weight: 800; margin-bottom: 5px; letter-spacing: -0.4px; }}
        .method-sub {{ color: #475569; font-size: 13.5px; line-height: 1.7; margin-bottom: 5px; }}
        .method-note {{ color: #94a3b8; font-size: 12.5px; }}
        .metrics-row {{ display: grid; grid-template-columns: 1.15fr 1.15fr 1fr 1fr; gap: 12px; }}
        .metric-card {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; background: #f8fafc; }}
        .metric-card.big {{ background: #eff6ff; border-color: #bfdbfe; }}
        .metric-label {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }}
        .metric-value-line {{ font-size: 22px; font-weight: 800; line-height: 1.2; margin-bottom: 10px; letter-spacing: -0.5px; }}
        .metric-delta {{ display: inline-block; padding: 5px 13px; border-radius: 999px; font-size: 13px; font-weight: 700; transition: all 0.2s ease; }}
        .good {{ color: #065f46; background: #d1fae5; }}
        .bad {{ color: #991b1b; background: #fee2e2; }}
        .neutral {{ color: #374151; background: #e5e7eb; }}
        .styled-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .styled-table th, .styled-table td {{ border-bottom: 1px solid #e2e8f0; padding: 11px 10px; text-align: left; vertical-align: middle; }}
        .styled-table th {{ background: #f8fafc; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: #475569; }}
        
        /* Flexbox Layout Fix for Pareto Section */
        .pareto-layout {{ display: flex; gap: 24px; align-items: stretch; width: 100%; }}
        .pareto-graph-col {{ flex-grow: 1; min-width: 0; position: relative; }}
        
        /* Sidebar Checkbox Card Styling */
        .pareto-sidebar-card {{
            width: 220px;
            flex-shrink: 0;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 16px 20px;
            align-self: flex-start;
        }}
        .pareto-sidebar-card h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin: 0 0 16px 0; }}
        .pareto-sidebar-group {{ margin-bottom: 16px; border-bottom: 1px solid #e2e8f0; padding-bottom: 14px; }}
        .pareto-sidebar-group:last-child {{ margin-bottom: 0; border-bottom: none; padding-bottom: 0; }}
        .psg-title {{ font-weight: 700; color: #334155; font-size: 14px; margin-bottom: 8px; }}
        .psg-toggles {{ display: flex; gap: 14px; }}
        .psg-toggles label {{ cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; font-size: 13.5px; color: #475569; }}
        
        ul {{ margin: 8px 0 0; padding-left: 20px; }}
        li {{ margin-bottom: 5px; font-size: 14px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Unlearning Dashboard</h1>
    <p class="page-subtitle">
        {forget_percentage}% forget split &nbsp;·&nbsp; K = {k_val} &nbsp;·&nbsp; Mode = {mode_display}
    </p>

    <div class="meta">
        <div><b>Folder:</b> {results_path.parent}</div>
        <div><b>Selection Logic ({mode.upper()}):</b> {'Hard cutoff (retain drop < ' + str(retain_drop_threshold) + ')' if mode == 'standard' else 'Ranked strictly by Custom Utility Score (threshold ignored).'}</div>
        <div><b>Base Model Filter:</b> <span style="color:#0f172a; font-weight:600;">{filter_text}</span></div>
    </div>

    {agg_section}
    
    <div class="section">
        <h2>Pareto Frontier</h2>
        <div class="pareto-layout">
            <div class="pareto-graph-col">
                <div id="pareto-graph-wrapper" style="width: 100%; height: 100%;">{pareto_div}</div>
            </div>
            {sidebar_controls_html}
        </div>
    </div>

    <div class="section">
        <div class="header-row">
            <h2>Best Run per Method</h2>
            <div class="toggle-group">
                <button class="tgl-best active" data-mode="pp" onclick="updateBestRunMode('pp')">Absolute (pp)</button>
                <button class="tgl-best" data-mode="rel" onclick="updateBestRunMode('rel')">Relative (%)</button>
            </div>
        </div>
        {method_rows}
    </div>

    <div class="section">
        <h2>Summary Table</h2>
        {table_html}
    </div>

    <div class="section">
        <h2>Before/After Performance Comparison</h2>
        <div style="width: 100%; overflow: hidden;">{comp_div}</div>
    </div>

    <div class="section">
        <h2>Loss Curves</h2>
        <div style="width: 100%; overflow: hidden;">{loss_div}</div>
    </div>
    
    <div class="section">
        <h2>Utility Score Formula (Fair Mode)</h2>
        <p class="section-desc">
            In <b>fair</b> mode, runs are ranked using a custom utility score that balances forgetting efficacy with a penalty for degrading retained knowledge. Both metrics are computed as relative percentage drops to ensure a level playing field across different base models.
        </p>
        {math_block}
        {math_desc}
    </div>
</div>
<script>
// Aggregation Toggles
function updateAgg(stat) {{
    document.querySelectorAll('.tgl').forEach(b => {{ b.classList.toggle('active', b.dataset.stat === stat); }});
    document.querySelectorAll('.agg-stat').forEach(el => {{
        const v = el.dataset[stat];
        el.textContent = (v === 'null' || v === undefined) ? 'N/A' : v + '%';
    }});
}}

// Best Run pp vs % Toggle
function updateBestRunMode(mode) {{
    document.querySelectorAll('.tgl-best').forEach(b => {{ b.classList.toggle('active', b.dataset.mode === mode); }});
    document.querySelectorAll('.metric-toggle-target').forEach(el => {{
        el.textContent = el.dataset[mode];
    }});
}}

// Pareto Checkbox Plotly integration
function toggleTrace(traceName, isVisible) {{
    const gd = document.querySelector('#pareto-graph-wrapper .js-plotly-plot');
    if(!gd) return;
    
    const update = {{ visible: isVisible ? true : false }};
    const traceIndices = [];
    gd.data.forEach((trace, i) => {{
        if(trace.name === traceName) {{
            traceIndices.push(i);
        }}
    }});
    
    if(traceIndices.length > 0) {{
        Plotly.restyle(gd, update, traceIndices);
    }}
}}

// Resize Plotly charts to ensure they fit correctly inside flex containers
function resizePlots() {{
    const paretoGraph = document.querySelector('#pareto-graph-wrapper .js-plotly-plot');
    if(paretoGraph) Plotly.Plots.resize(paretoGraph);
    
    const otherGraphs = document.querySelectorAll('.section .js-plotly-plot');
    otherGraphs.forEach(g => {{
        if (g !== paretoGraph) Plotly.Plots.resize(g);
    }});
}}

// 1. Listen for standard window resizes
window.addEventListener('resize', resizePlots);

// 2. Force a single resize calculation immediately after DOM is ready
setTimeout(resizePlots, 100);
</script>
</body>
</html>"""


def main():
    forget_percentage, retain_drop_threshold, k_val, mode, num_top_models = parse_args()

    is_demo = os.environ.get("RUN_MODE", "Normal") == "Demography"
    if is_demo:
        results_base = Path(f"D:/Bob_Skripsi_Do Not Delete/results_demography/{forget_percentage}_percent")
        out_dir = Path(f"D:/Bob_Skripsi_Do Not Delete/Analysis/Demography/{forget_percentage}_percent")
    else:
        results_base = Path(f"C:/Bob/results/{forget_percentage}_percent") if forget_percentage in [1, 20] else Path(f"D:/Bob_Skripsi_Do Not Delete/results/{forget_percentage}_percent")
        out_dir = Path(f"D:/Bob_Skripsi_Do Not Delete/Analysis/Normal/{forget_percentage}_percent")

    if not results_base.exists():
        raise FileNotFoundError(f"Results folder not found: {results_base}")

    results_df, train_df, loss_df, results_path, loss_log_path = load_inputs(results_base)
    results_df = prepare_results(results_df)

    # Filter base models based on num_top_models if specified
    if num_top_models is not None and not train_df.empty:
        # Standardize train_df to match results_df types for exact merging
        train_numeric_cols = ["K", "train_lr", "gamma", "hidden_dim", "train_batch", TOP_SELECTION_METRIC]
        train_df = normalize_numeric(train_df, train_numeric_cols)
        
        rank_df = (
            train_df[train_df["K"] == k_val]
            .sort_values(
                [TOP_SELECTION_METRIC, "train_lr", "gamma", "hidden_dim", "train_batch"],
                ascending=[False, True, True, True, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )
        top_train_models = rank_df.head(num_top_models)
        
        # Merge to enforce strict filtering
        keys = ["train_lr", "gamma", "hidden_dim", "train_batch"]
        base_configs_df = top_train_models[keys].drop_duplicates()
        
        results_df = results_df.merge(base_configs_df, on=keys, how="inner")
        print(f"Filtered tuning results to top {num_top_models} base models at K={k_val} (Remaining runs: {len(results_df)}).")

    agg = build_aggregate_data(results_df, k_val)

    selected_df, selection_notes = select_one_per_method(
        results_df,
        k_val=k_val,
        retain_drop_threshold=retain_drop_threshold,
        mode=mode
    )

    if selected_df.empty:
        raise ValueError(f"No valid runs could be processed for K={k_val}.")

    pareto_fig = build_pareto_figure(results_df, k_val)
    loss_fig = build_loss_figure(loss_df, selected_df)
    
    # Generate the new comparison figure using the dynamically filtered data
    comp_fig = build_comparison_figure(selected_df, k_val, forget_percentage)

    html = build_html(
        forget_percentage=forget_percentage,
        retain_drop_threshold=retain_drop_threshold,
        k_val=k_val,
        mode=mode,
        num_top_models=num_top_models,
        selected_df=selected_df,
        selection_notes=selection_notes,
        agg=agg,
        results_path=results_path,
        loss_log_path=loss_log_path,
        pareto_fig=pareto_fig,
        loss_fig=loss_fig,
        comp_fig=comp_fig, # Pass it to the HTML builder
    )

    out_path = Path(OUT_DIR) / f"dashboard_fair_{retain_drop_threshold}_{mode}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {out_path} (Mode: {mode})")


if __name__ == "__main__":
    main()