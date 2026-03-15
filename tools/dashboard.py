#!/usr/bin/env python
# coding: utf-8

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot


METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]


def parse_args():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python dashboard.py <forget_percentage> [retain_drop_threshold] [K]\n"
            "Example: python dashboard.py 1\n"
            "Example: python dashboard.py 20 0.01 10"
        )
    forget_percentage = int(sys.argv[1])
    retain_drop_threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.01
    k_val = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    return forget_percentage, retain_drop_threshold, k_val


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{100.0 * float(x):.2f}%"


def pp(x):
    if pd.isna(x):
        return "N/A"
    return f"{100.0 * float(x):.2f} pp"


def normalize_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def signed_log10(x):
    if pd.isna(x):
        return np.nan
    x = float(x)
    return np.sign(x) * np.log10(1.0 + abs(x))


def signed_log_label():
    return "sign(x) · log10(1 + |x|)"


def find_loss_log_path(results_base: Path):
    for name in ["unlearning_loss_log.csv", "unlearn_loss_log.csv", "loss_log.csv"]:
        p = results_base / name
        if p.exists():
            return p
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
    return df


def build_aggregate_data(df: pd.DataFrame, k_val: int) -> dict:
    df_k = df[(df["K"] == k_val)].copy()
    agg = {}
    for method in METHODS:
        mdf = df_k[df_k["method"] == method]
        if mdf.empty:
            continue
        agg[method] = {"n": int(len(mdf))}
        for col in ["retain_drop_hit", "forget_drop_hit"]:
            vals = mdf[col].dropna() if col in mdf.columns else pd.Series(dtype=float)
            if len(vals) == 0:
                agg[method][col] = {"mean": None, "median": None, "q1": None, "q3": None}
            else:
                agg[method][col] = {
                    "mean":   round(float(vals.mean())            * 100, 3),
                    "median": round(float(vals.median())          * 100, 3),
                    "q1":     round(float(vals.quantile(0.25))    * 100, 3),
                    "q3":     round(float(vals.quantile(0.75))    * 100, 3),
                }
    return agg


def select_one_per_method(df: pd.DataFrame, k_val: int, retain_drop_threshold: float):
    df_k = df[df["K"] == k_val].copy()
    selected_rows, selection_notes = [], []
    thresh_display = f"{retain_drop_threshold * 100:.2f}% pp"
    for method in METHODS:
        mdf = df_k[df_k["method"] == method].copy()
        if mdf.empty:
            selection_notes.append(f"{method}: no rows found at K={k_val}.")
            continue
        constrained = mdf[mdf["retain_drop_hit"] < retain_drop_threshold].copy()
        if constrained.empty:
            selection_notes.append(
                f"{method}: no run satisfied retain_drop < {thresh_display} at K={k_val}."
            )
            continue
        # constrained['forget_quality_new'] = constrained['retain_Hit'] - constrained['base_retain_Hit'] - (constrained['forget_Hit'] - constrained['base_forget_Hit'])
        constrained = constrained.sort_values(
            by="forget_drop_hit",
            ascending=False,
            na_position="last",
        )
        row = constrained.iloc[0].copy()
        row["selection_note"] = (
            f"Selected from runs with retain_drop &lt; {thresh_display}, "
            f"then maximized forget_drop."
        )
        selected_rows.append(row)
        selection_notes.append(
            f"{method}: selected from runs with retain_drop < {thresh_display}, "
            f"then maximized forget_drop."
        )
    if not selected_rows:
        return pd.DataFrame(), selection_notes
    return pd.DataFrame(selected_rows).reset_index(drop=True), selection_notes


def metric_flag(before, after, higher_is_better=True):
    if pd.isna(before) or pd.isna(after):
        return "neutral", "N/A", "•"
    delta = float(after) - float(before)
    if abs(delta) < 1e-12:
        return "neutral", "0.00 pp", "•"
    good = (delta > 0) if higher_is_better else (delta < 0)
    return ("good" if good else "bad"), pp(abs(delta)), ("▲" if delta > 0 else "▼")


def build_aggregate_section(agg: dict, k_val: int) -> str:
    cards = []
    for method in METHODS:
        if method not in agg:
            cards.append(
                f'<div class="agg-card"><div class="agg-method-name">{method}</div>'
                f'<div class="agg-n">No data</div></div>'
            )
            continue
        d = agg[method]
        n = d["n"]

        def sv(col, stat):
            v = d.get(col, {}).get(stat)
            return str(v) if v is not None else "null"

        def fv(col, stat="mean"):
            v = d.get(col, {}).get(stat)
            return f"{v:.2f} pp" if v is not None else "N/A"

        cards.append(f"""
        <div class="agg-card">
            <div class="agg-method-name">{method}</div>
            <div class="agg-n">{n} runs &nbsp;·&nbsp; K={k_val}</div>
            <div class="agg-divider"></div>
            <div class="agg-metric-row">
                <span class="agg-label">Retain Drop</span>
                <span class="agg-stat retain-col"
                    data-mean="{sv('retain_drop_hit','mean')}"
                    data-median="{sv('retain_drop_hit','median')}"
                    data-q1="{sv('retain_drop_hit','q1')}"
                    data-q3="{sv('retain_drop_hit','q3')}"
                >{fv('retain_drop_hit')}</span>
            </div>
            <div class="agg-metric-row">
                <span class="agg-label">Forget Drop</span>
                <span class="agg-stat forget-col"
                    data-mean="{sv('forget_drop_hit','mean')}"
                    data-median="{sv('forget_drop_hit','median')}"
                    data-q1="{sv('forget_drop_hit','q1')}"
                    data-q3="{sv('forget_drop_hit','q3')}"
                >{fv('forget_drop_hit')}</span>
            </div>
        </div>""")

    return f"""
    <div class="section">
        <div class="header-row">
            <h2>Aggregate Performance <span class="sub-label">all runs &nbsp;·&nbsp; K={k_val}</span></h2>
            <div class="toggle-group">
                <button class="tgl active" data-stat="mean"   onclick="updateAgg('mean')">Mean</button>
                <button class="tgl"        data-stat="median" onclick="updateAgg('median')">Median</button>
                <button class="tgl"        data-stat="q1"     onclick="updateAgg('q1')">Q1 (25th)</button>
                <button class="tgl"        data-stat="q3"     onclick="updateAgg('q3')">Q3 (75th)</button>
            </div>
        </div>
        <p class="section-desc">Distribution of retain_drop and forget_drop across all hyperparameter combinations.</p>
        <div class="agg-grid">{''.join(cards)}</div>
    </div>"""


def build_method_rows_html(selected_df: pd.DataFrame, k_val: int) -> str:
    rows = []
    for _, r in selected_df.iterrows():
        r_css, r_d, r_a = metric_flag(r["base_retain_Hit"],  r["retain_Hit"],  True)
        f_css, f_d, f_a = metric_flag(r["base_forget_Hit"],  r["forget_Hit"],  False)
        rn_css,rn_d,rn_a= metric_flag(r["base_retain_NDCG"], r["retain_NDCG"], True)
        fn_css,fn_d,fn_a= metric_flag(r["base_forget_NDCG"], r["forget_NDCG"], False)
        rows.append(f"""
        <div class="method-row">
            <div class="method-title-block">
                <div class="method-name">{r['method']}</div>
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
                    <div class="metric-delta {r_css}">{r_a} {r_d}</div>
                </div>
                <div class="metric-card big">
                    <div class="metric-label">Forget Hit@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_forget_Hit'])} → {pct(r['forget_Hit'])}</div>
                    <div class="metric-delta {f_css}">{f_a} {f_d}</div>
                </div>
                <div class="metric-card small">
                    <div class="metric-label">Retain NDCG@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_retain_NDCG'])} → {pct(r['retain_NDCG'])}</div>
                    <div class="metric-delta {rn_css}">{rn_a} {rn_d}</div>
                </div>
                <div class="metric-card small">
                    <div class="metric-label">Forget NDCG@{k_val}</div>
                    <div class="metric-value-line">{pct(r['base_forget_NDCG'])} → {pct(r['forget_NDCG'])}</div>
                    <div class="metric-delta {fn_css}">{fn_a} {fn_d}</div>
                </div>
            </div>
        </div>""")
    return "\n".join(rows)


def build_summary_table(selected_df: pd.DataFrame, k_val: int) -> str:
    rows = []
    for _, r in selected_df.iterrows():
        r_css, r_d, r_a = metric_flag(r["base_retain_Hit"], r["retain_Hit"], True)
        f_css, f_d, f_a = metric_flag(r["base_forget_Hit"], r["forget_Hit"], False)
        rows.append(f"""
        <tr>
            <td><b>{r['method']}</b></td>
            <td>{pct(r['base_retain_Hit'])}</td>
            <td>{pct(r['retain_Hit'])}</td>
            <td><span class="{r_css}">{r_a} {r_d}</span></td>
            <td>{pct(r['base_forget_Hit'])}</td>
            <td>{pct(r['forget_Hit'])}</td>
            <td><span class="{f_css}">{f_a} {f_d}</span></td>
            <td>{pp(r['retain_drop_hit'])}</td>
            <td>{pp(r['forget_drop_hit'])}</td>
            <td>{r.get('lambda_retain', np.nan)}</td>
            <td>{r.get('unlearn_lr', np.nan)}</td>
            <td>{int(r['unlearn_iters']) if pd.notna(r['unlearn_iters']) else 'N/A'}</td>
            <td>{r.get('loss_total_final', np.nan):.6g}</td>
        </tr>""")
    return f"""
    <table class="styled-table">
        <thead><tr>
            <th>Method</th>
            <th>Base Retain Hit@{k_val}</th><th>After Retain Hit@{k_val}</th><th>Retain Δ</th>
            <th>Base Forget Hit@{k_val}</th><th>After Forget Hit@{k_val}</th><th>Forget Δ</th>
            <th>Retain Drop</th><th>Forget Drop</th>
            <th>λ</th><th>u_lr</th><th>u_iters</th><th>Final Loss</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>"""


def build_loss_figure(loss_df: pd.DataFrame, selected_df: pd.DataFrame):
    if loss_df.empty:
        return None
    loss_df = normalize_numeric(loss_df.copy(), [
        "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain", "iter",
        "loss_forget", "loss_retain", "loss_total",
    ])
    rows = len(selected_df)
    if rows == 0:
        return None
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=False,
        subplot_titles=[f"{m} — signed-log loss" for m in selected_df["method"].tolist()],
        vertical_spacing=0.08,
    )
    color_map = {"loss_forget": "#dc2626", "loss_retain": "#2563eb", "loss_total": "#111827"}
    for i, (_, s) in enumerate(selected_df.iterrows(), start=1):
        sub = loss_df[
            (loss_df["method"] == s["method"]) &
            (loss_df["train_lr"].round(12)     == float(s["train_lr"])) &
            (loss_df["gamma"].round(12)         == float(s["gamma"])) &
            (loss_df["hidden_dim"]              == float(s["hidden_dim"])) &
            (loss_df["train_batch"]             == float(s["train_batch"])) &
            (loss_df["unlearn_lr"].round(12)    == float(s["unlearn_lr"])) &
            (loss_df["unlearn_iters"]           == float(s["unlearn_iters"])) &
            (loss_df["lambda_retain"].round(12) == float(s["lambda_retain"]))
        ].copy().sort_values("iter")
        if sub.empty:
            continue
        for ln in ["loss_forget", "loss_retain", "loss_total"]:
            if ln not in sub.columns:
                continue
            sub[f"{ln}_sl"] = sub[ln].apply(signed_log10)
            fig.add_trace(go.Scatter(
                x=sub["iter"], y=sub[f"{ln}_sl"],
                mode="lines+markers", name=ln,
                line=dict(color=color_map[ln], width=2), marker=dict(size=5),
                showlegend=(i == 1),
                customdata=np.stack([sub[ln].fillna(np.nan), sub[f"{ln}_sl"].fillna(np.nan)], axis=1),
                hovertemplate="iter=%{x}<br>raw=%{customdata[0]:.6g}<br>signed_log=%{customdata[1]:.4f}<extra></extra>",
            ), row=i, col=1)
        fig.update_xaxes(title_text="Unlearning iteration", row=i, col=1)
        fig.update_yaxes(title_text=signed_log_label(), row=i, col=1)
    fig.update_layout(
        title="Loss curves by selected run", template="plotly_white",
        height=max(380, 300 * rows), margin=dict(l=50, r=40, t=90, b=40),
        font=dict(family="Inter, system-ui, sans-serif"),
    )
    return fig


def build_html(
    forget_percentage, retain_drop_threshold, k_val,
    selected_df, selection_notes, agg,
    results_path, loss_log_path, loss_fig,
) -> str:
    thresh_display = f"{retain_drop_threshold * 100:.2f}% pp"
    method_rows  = build_method_rows_html(selected_df, k_val)
    table_html   = build_summary_table(selected_df, k_val)
    agg_section  = build_aggregate_section(agg, k_val)
    notes_html   = "".join(f"<li>{n}</li>" for n in selection_notes)
    loss_div     = (
        plot(loss_fig, output_type="div", include_plotlyjs="cdn")
        if loss_fig is not None
        else '<div class="empty-box">No loss-log CSV found — loss curves skipped.</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>Unlearning Dashboard — {forget_percentage}%</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after {{ box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 15px;
            line-height: 1.6;
            background: #f1f5f9;
            color: #0f172a;
            margin: 0;
            padding: 40px 0 72px;
        }}
        .container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 0 64px;
        }}
        h1 {{
            font-size: 32px;
            font-weight: 800;
            letter-spacing: -0.6px;
            margin: 0 0 6px;
        }}
        h2 {{
            font-size: 20px;
            font-weight: 700;
            margin: 0 0 14px;
            letter-spacing: -0.2px;
        }}
        .page-subtitle {{
            color: #64748b;
            font-size: 15px;
            margin: 0 0 30px;
        }}
        /* sections */
        .section, .meta {{
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 22px;
            box-shadow: 0 1px 4px rgba(0,0,0,.04);
        }}
        .meta {{ font-size: 14px; line-height: 2.1; }}
        .meta b {{ color: #334155; }}
        .header-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 10px;
        }}
        .header-row h2 {{ margin: 0; }}
        .sub-label {{
            font-size: 13px;
            font-weight: 500;
            color: #94a3b8;
            margin-left: 8px;
        }}
        .section-desc {{
            color: #64748b;
            font-size: 13.5px;
            margin: 0 0 18px;
        }}
        /* toggle */
        .toggle-group {{
            display: inline-flex;
            background: #f1f5f9;
            border-radius: 10px;
            padding: 3px;
            gap: 2px;
        }}
        .tgl {{
            border: none;
            background: transparent;
            padding: 7px 18px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 13px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            transition: all .15s;
        }}
        .tgl:hover {{ background: #e2e8f0; color: #1e293b; }}
        .tgl.active {{
            background: #fff;
            color: #1e293b;
            box-shadow: 0 1px 4px rgba(0,0,0,.12);
        }}
        /* aggregate */
        .agg-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
        }}
        .agg-card {{
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px;
            background: #f8fafc;
        }}
        .agg-method-name {{
            font-size: 16px;
            font-weight: 800;
            margin-bottom: 3px;
        }}
        .agg-n {{
            font-size: 12px;
            color: #94a3b8;
            margin-bottom: 14px;
        }}
        .agg-divider {{ border-top: 1px solid #e2e8f0; margin-bottom: 12px; }}
        .agg-metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
        }}
        .agg-label {{ font-size: 13px; color: #64748b; }}
        .agg-stat   {{ font-size: 16px; font-weight: 700; }}
        .retain-col {{ color: #b45309; }}
        .forget-col {{ color: #1d4ed8; }}
        /* method rows */
        .method-row {{
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 14px;
            background: #fff;
        }}
        .method-title-block {{ margin-bottom: 14px; }}
        .method-name {{
            font-size: 23px;
            font-weight: 800;
            margin-bottom: 5px;
            letter-spacing: -0.4px;
        }}
        .method-sub {{
            color: #475569;
            font-size: 13.5px;
            line-height: 1.7;
            margin-bottom: 5px;
        }}
        .method-note {{ color: #94a3b8; font-size: 12.5px; }}
        .metrics-row {{
            display: grid;
            grid-template-columns: 1.15fr 1.15fr 1fr 1fr;
            gap: 12px;
        }}
        .metric-card {{
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px;
            background: #f8fafc;
        }}
        .metric-card.big {{ background: #eff6ff; border-color: #bfdbfe; }}
        .metric-label {{
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        .metric-value-line {{
            font-size: 22px;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}
        .metric-delta {{
            display: inline-block;
            padding: 5px 13px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
        }}
        .good    {{ color: #065f46; background: #d1fae5; }}
        .bad     {{ color: #991b1b; background: #fee2e2; }}
        .neutral {{ color: #374151; background: #e5e7eb; }}
        /* table */
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .styled-table th,
        .styled-table td {{
            border-bottom: 1px solid #e2e8f0;
            padding: 11px 10px;
            text-align: left;
            vertical-align: top;
        }}
        .styled-table th {{
            background: #f8fafc;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #475569;
            position: sticky;
            top: 0;
        }}
        .styled-table tr:hover td {{ background: #f8fafc; }}
        /* misc */
        .empty-box {{
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 12px;
            padding: 28px;
            color: #64748b;
            text-align: center;
            font-size: 14px;
        }}
        ul {{ margin: 8px 0 0; padding-left: 20px; }}
        li {{ margin-bottom: 5px; font-size: 14px; }}
        .small-note {{ color: #64748b; font-size: 13px; margin: 0 0 14px; }}
        @media (max-width: 1200px) {{
            .agg-grid    {{ grid-template-columns: repeat(2, 1fr); }}
            .metrics-row {{ grid-template-columns: 1fr 1fr; }}
        }}
        @media (max-width: 700px) {{
            .container   {{ padding: 0 20px; }}
            .agg-grid    {{ grid-template-columns: 1fr; }}
            .metrics-row {{ grid-template-columns: 1fr; }}
            .metric-value-line {{ font-size: 18px; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Unlearning Dashboard</h1>
    <p class="page-subtitle">
        {forget_percentage}% forget split &nbsp;·&nbsp; K = {k_val} &nbsp;·&nbsp; threshold = {thresh_display}
    </p>

    <div class="meta">
        <div><b>Folder:</b> {results_path.parent}</div>
        <div><b>Main CSV:</b> {results_path.name}</div>
        <div><b>Loss log:</b> {loss_log_path.name if loss_log_path is not None else 'Not found'}</div>
        <div><b>Selection rule:</b> For each method at K = {k_val}, keep only runs with
            retain_drop &lt; {thresh_display}, then pick the run with maximum forget_drop.</div>
    </div>

    {agg_section}

    <div class="section">
        <h2>Selection Notes</h2>
        <ul>{notes_html}</ul>
    </div>

    <div class="section">
        <h2>Best Run per Method</h2>
        {method_rows}
    </div>

    <div class="section">
        <h2>Summary Table</h2>
        {table_html}
    </div>

    <div class="section">
        <h2>Loss Curves</h2>
        <p class="small-note">
            Signed-log scale: {signed_log_label()}.
            Keeps very large negative Appendix-I losses visible while preserving sign and trend.
        </p>
        {loss_div}
    </div>
</div>

<script>
function updateAgg(stat) {{
    // update toggle buttons
    document.querySelectorAll('.tgl').forEach(b => {{
        b.classList.toggle('active', b.dataset.stat === stat);
    }});
    // update all agg-stat spans
    document.querySelectorAll('.agg-stat').forEach(el => {{
        const v = el.dataset[stat];
        el.textContent = (v === 'null' || v === undefined) ? 'N/A' : v + ' pp';
    }});
}}
</script>
</body>
</html>"""


def main():
    forget_percentage, retain_drop_threshold, k_val = parse_args()

    results_base = Path(f"C:/Bob/results/{forget_percentage}_percent")
    if not results_base.exists():
        raise FileNotFoundError(f"Results folder not found: {results_base}")

    results_df, train_df, loss_df, results_path, loss_log_path = load_inputs(results_base)
    results_df = prepare_results(results_df)

    agg = build_aggregate_data(results_df, k_val)

    selected_df, selection_notes = select_one_per_method(
        results_df,
        k_val=k_val,
        retain_drop_threshold=retain_drop_threshold,
    )

    if selected_df.empty:
        raise ValueError(
            f"No selected runs found. Either no rows exist for K={k_val}, "
            f"or no method has retain_drop < {retain_drop_threshold * 100:.2f}% pp."
        )

    loss_fig = build_loss_figure(loss_df, selected_df)

    html = build_html(
        forget_percentage=forget_percentage,
        retain_drop_threshold=retain_drop_threshold,
        k_val=k_val,
        selected_df=selected_df,
        selection_notes=selection_notes,
        agg=agg,
        results_path=results_path,
        loss_log_path=loss_log_path,
        loss_fig=loss_fig,
    )

    out_path = results_base / "dashboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {out_path}")


if __name__ == "__main__":
    main()

