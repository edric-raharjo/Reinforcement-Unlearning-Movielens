#!/usr/bin/env python
# coding: utf-8
# C:\Bob\tools\dashboard_analyze.py

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    raise ValueError(
        "Usage: python dashboard_analyze.py <forget_pct>\n"
        "Example: python dashboard_analyze.py 20\n"
    )

FORGET_PERCENTAGE = int(sys.argv[1])
RESULTS_BASE      = Path(f"C:/Bob/results/{FORGET_PERCENTAGE}_percent")
ANALYSIS_DIR      = RESULTS_BASE / "analysis"
OUTPUT_HTML       = ANALYSIS_DIR / "dashboard_analyze.html"

METHODS      = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
FLIP_LABELS  = ["flipped_harmful", "stable_correct", "stable_miss", "recovered"]

# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------
FLIP_COLORS = {
    "flipped_harmful": "#e74c3c",
    "stable_correct":  "#27ae60",
    "stable_miss":     "#f39c12",
    "recovered":       "#2980b9",
}
METHOD_COLORS = {
    "Ye_ApxI":      "#8e44ad",
    "Ye_multi":     "#16a085",
    "New_True_inf": "#d35400",
    "New_Max":      "#2c3e50",
}

FLIP_META = {
    "flipped_harmful": {
        "label": "Flipped Harmful",
        "icon":  "⚠️",
        "desc":  "Was a correct recommendation before unlearning, became incorrect after. "
                 "These are retain users directly harmed by the unlearning process.",
        "color": FLIP_COLORS["flipped_harmful"],
    },
    "stable_correct": {
        "label": "Stable Correct",
        "icon":  "✅",
        "desc":  "Correct both before and after unlearning. "
                 "The unlearning process did not affect these retain users.",
        "color": FLIP_COLORS["stable_correct"],
    },
    "stable_miss": {
        "label": "Stable Miss",
        "icon":  "❌",
        "desc":  "Incorrect both before and after unlearning. "
                 "The model was already failing these users — unlearning did not change that.",
        "color": FLIP_COLORS["stable_miss"],
    },
    "recovered": {
        "label": "Recovered",
        "icon":  "🔄",
        "desc":  "Was incorrect before unlearning, became correct after. "
                 "A rare positive side-effect of the unlearning process.",
        "color": FLIP_COLORS["recovered"],
    },
}

METRIC_META = {
    "overlap_2gram_pct": {
        "label": "2-gram Overlap %",
        "desc":  "Percentage of consecutive movie pairs (A→B) in the retain user's "
                 "watch-history prefix that also appear in at least one forget user's prefix. "
                 "Higher = more behaviourally similar to forgotten users at a coarse level.",
    },
    "overlap_3gram_pct": {
        "label": "3-gram Overlap %",
        "desc":  "Percentage of consecutive movie triples (A→B→C) in the retain user's "
                 "prefix that appear in any forget user's prefix. "
                 "Higher specificity than 2-gram — a real match here is meaningful.",
    },
    "max_single_forget_overlap_pct": {
        "label": "Max Single-User Overlap %",
        "desc":  "For the retain user, the highest n-gram overlap % with any single "
                 "forget user. Captures whether the retain user has a 'near-twin' in the "
                 "forget set, rather than just diffuse similarity spread across many users.",
    },
    "num_forget_users_sharing_ngram": {
        "label": "# Forget Users Sharing n-gram",
        "desc":  "How many distinct forget users share at least one n-gram with this "
                 "retain user. High breadth means the retain user overlaps with many "
                 "different forgotten users, not just one.",
    },
}

METHOD_META = {
    "Ye_ApxI": {
        "label": "Ye Appendix I",
        "desc":  "Fixed λ=1.0. Maximises forget-set Q-values (gradient ascent on forget "
                 "actions) while penalising deviation of retain-set Q-values from the "
                 "original model. Uses action-specific Q-values gathered by the buffer actions.",
    },
    "Ye_multi": {
        "label": "Ye Multi-Env",
        "desc":  "Fixed λ=1.0. True L∞ variant — uses the maximum Q-value across all "
                 "actions (not just the buffer action) for both forget and retain terms. "
                 "Models the worst-case policy deviation.",
    },
    "New_True_inf": {
        "label": "New True-Inf",
        "desc":  "λ swept. Same L∞ max-action structure as Ye_multi, but both forget and "
                 "retain terms are squared before averaging, making the loss more sensitive "
                 "to large deviations.",
    },
    "New_Max": {
        "label": "New Max",
        "desc":  "λ swept. Forget term uses raw max Q-value (not absolute), retain term "
                 "uses squared L∞ deviation. Allows the forget term to exploit sign — "
                 "drives Q-values down rather than just to zero.",
    },
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_method_data(method):
    d    = {}
    base = ANALYSIS_DIR / method

    summary_path = base / "flipped_vs_stable_summary.csv"
    eval_path    = base / "retain_eval_per_traj.csv"
    overlap_path = base / "ngram_overlap.csv"
    info_path    = base / "selected_run_info.txt"

    if not summary_path.exists():
        return None

    raw = pd.read_csv(summary_path, header=[0, 1], index_col=0)
    raw.index.name = "flip_label"
    d["summary"] = raw

    if eval_path.exists():
        d["eval"] = pd.read_csv(eval_path)

    if overlap_path.exists():
        d["overlap"] = pd.read_csv(overlap_path)

    if info_path.exists():
        info = {}
        for line in info_path.read_text().splitlines():
            if "=" in line and "===" not in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
        d["info"] = info

    return d

all_data = {m: load_method_data(m) for m in METHODS}
all_data = {m: v for m, v in all_data.items() if v is not None}

combined_path = ANALYSIS_DIR / "all_methods_summary.csv"
combined_df   = pd.read_csv(combined_path) if combined_path.exists() else pd.DataFrame()

# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def fig_to_json(fig):
    return fig.to_json()

# ── A. Stacked bar: outcome distribution, annotated with % ─────────────────
def make_flip_distribution_chart(all_data):
    methods = []
    counts  = {lbl: [] for lbl in FLIP_LABELS}

    for method, data in all_data.items():
        if "eval" not in data:
            continue
        vc = data["eval"]["flip_label"].value_counts()
        methods.append(METHOD_META.get(method, {}).get("label", method))
        for lbl in FLIP_LABELS:
            counts[lbl].append(int(vc.get(lbl, 0)))

    totals = [
        sum(counts[lbl][i] for lbl in FLIP_LABELS)
        for i in range(len(methods))
    ]

    fig = go.Figure()
    for lbl in FLIP_LABELS:
        # Only annotate segments that are large enough to read
        text_vals = [
            f"{counts[lbl][i]/totals[i]*100:.1f}%" if totals[i] > 0 and counts[lbl][i]/totals[i] >= 0.04 else ""
            for i in range(len(methods))
        ]
        fig.add_trace(go.Bar(
            name=FLIP_META[lbl]["label"],
            x=methods,
            y=counts[lbl],
            text=text_vals,
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=13, color="white", family="Inter, sans-serif"),
            marker_color=FLIP_COLORS[lbl],
        ))

    fig.update_layout(
        barmode="stack",
        title="Retain Trajectory Outcomes per Method",
        xaxis_title="Method",
        yaxis_title="# Retain Users",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
    )
    return fig_to_json(fig)

# ── B. Grouped bar: overlap by flip label & method, annotated with value ────
def make_overlap_comparison_chart(all_data, metric_col, metric_label):
    fig = go.Figure()

    for lbl in FLIP_LABELS:
        x_vals, y_means, y_stds = [], [], []

        for method, data in all_data.items():
            if "overlap" not in data:
                continue
            sub = data["overlap"][data["overlap"]["flip_label"] == lbl]
            if metric_col not in sub.columns or sub.empty:
                continue
            x_vals.append(METHOD_META.get(method, {}).get("label", method))
            y_means.append(round(sub[metric_col].mean(), 4))
            y_stds.append(round(sub[metric_col].std(), 4))

        if not x_vals:
            continue

        # Format annotation: use % suffix for pct cols, plain number otherwise
        is_pct = "pct" in metric_col
        text_vals = [
            f"{v:.2f}{'%' if is_pct else ''}" for v in y_means
        ]

        fig.add_trace(go.Bar(
            name=FLIP_META[lbl]["label"],
            x=x_vals,
            y=y_means,
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif"),
            marker_color=FLIP_COLORS[lbl],
            error_y=dict(type="data", array=y_stds, visible=True),
        ))

    fig.update_layout(
        barmode="group",
        title=f"{metric_label} by Flip Label & Method",
        xaxis_title="Method",
        yaxis_title=metric_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
        # Extra top margin so "outside" text isn't clipped
        margin=dict(t=120),
    )
    return fig_to_json(fig)

# ---------------------------------------------------------------------------
# Build charts JSON  (no scatter)
# ---------------------------------------------------------------------------
charts = {}
if HAS_PLOTLY and all_data:
    charts["flip_dist"]     = make_flip_distribution_chart(all_data)
    charts["overlap_2gram"] = make_overlap_comparison_chart(
        all_data, "overlap_2gram_pct", "2-gram Overlap %")
    charts["overlap_3gram"] = make_overlap_comparison_chart(
        all_data, "overlap_3gram_pct", "3-gram Overlap %")
    charts["max_single"]    = make_overlap_comparison_chart(
        all_data, "max_single_forget_overlap_pct", "Max Single-User Overlap %")

# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def summary_table_html(method, data):
    if "summary" not in data:
        return "<p>No summary data available.</p>"

    df   = data["summary"]
    cols = list(df.columns)

    rows_html = ""
    for lbl in FLIP_LABELS:
        if lbl not in df.index:
            continue
        meta  = FLIP_META[lbl]
        color = meta["color"]
        rows_html += f'<tr style="border-left: 4px solid {color};">'
        rows_html += (
            f'<td><span class="flip-badge" style="background:{color}22;color:{color};'
            f'border:1px solid {color};">'
            f'{meta["icon"]} {meta["label"]}</span></td>'
        )
        for col in cols:
            val = df.loc[lbl, col]
            rows_html += f"<td>{val:.4f}</td>" if pd.notna(val) else "<td>—</td>"
        rows_html += "</tr>"

    col_headers = "".join(f"<th>{c[0]}<br><small>{c[1]}</small></th>" for c in cols)
    return f"""
    <div class="table-wrapper">
      <table class="summary-table">
        <thead><tr><th>Outcome</th>{col_headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """

def selected_run_html(info):
    if not info:
        return ""
    fields = [
        ("train_lr",       "Training LR"),
        ("gamma",          "Gamma"),
        ("hidden_dim",     "Hidden Dim"),
        ("train_batch",    "Train Batch"),
        ("unlearn_lr",     "Unlearn LR"),
        ("unlearn_iters",  "Unlearn Iters"),
        ("lambda_retain",  "Lambda"),
        ("retain_drop",    "Retain Drop"),
        ("forget_drop",    "Forget Drop"),
        ("retain_Hit",     "Retain Hit@10"),
        ("forget_Hit",     "Forget Hit@10"),
        ("base_retain_Hit","Base Retain Hit@10"),
        ("base_forget_Hit","Base Forget Hit@10"),
    ]
    items = ""
    for key, label in fields:
        val          = info.get(key, "—")
        color_style  = ""
        if key == "retain_drop":
            try:
                v = float(val)
                color_style = f'style="color:{"#e74c3c" if v > 0.05 else "#27ae60"};font-weight:600;"'
            except: pass
        if key == "forget_drop":
            try:
                v = float(val)
                color_style = f'style="color:{"#27ae60" if v > 0.05 else "#e74c3c"};font-weight:600;"'
            except: pass
        items += (
            f'<div class="info-item">'
            f'<span class="info-label">{label}</span>'
            f'<span class="info-value" {color_style}>{val}</span>'
            f'</div>'
        )
    return f'<div class="info-grid">{items}</div>'

def method_tab_content(method, data):
    m_meta    = METHOD_META.get(method, {"label": method, "desc": ""})
    info_html = selected_run_html(data.get("info", {}))
    sum_html  = summary_table_html(method, data)

    eval_counts = ""
    if "eval" in data:
        vc    = data["eval"]["flip_label"].value_counts()
        total = len(data["eval"])
        badges = ""
        for lbl in FLIP_LABELS:
            cnt  = int(vc.get(lbl, 0))
            pct  = cnt / total * 100 if total > 0 else 0
            meta = FLIP_META[lbl]
            badges += (
                f'<div class="stat-card" style="border-top: 3px solid {meta["color"]};">'
                f'<div class="stat-icon">{meta["icon"]}</div>'
                f'<div class="stat-value" style="color:{meta["color"]};">{cnt}</div>'
                f'<div class="stat-pct">{pct:.1f}%</div>'
                f'<div class="stat-label">{meta["label"]}</div>'
                f'</div>'
            )
        eval_counts = f'<div class="stat-grid">{badges}</div>'

    # No scatter — removed entirely
    return f"""
    <div class="method-header" style="border-left: 5px solid {METHOD_COLORS.get(method,'#555')};">
      <h2>{m_meta['label']}</h2>
      <p class="method-desc">{m_meta['desc']}</p>
    </div>

    <h3 class="section-title">Selected Run Parameters</h3>
    {info_html}

    <h3 class="section-title">Outcome Distribution</h3>
    {eval_counts}

    <h3 class="section-title">Overlap Statistics by Outcome</h3>
    {sum_html}
    """

# ---------------------------------------------------------------------------
# Glossary  (C: moved to top)
# ---------------------------------------------------------------------------
def glossary_html():
    sections = []

    sections.append("<h2 class='glossary-title'>Outcome Labels</h2>")
    sections.append('<div class="glossary-grid">')
    for lbl, meta in FLIP_META.items():
        sections.append(f"""
        <div class="glossary-card" style="border-top:4px solid {meta['color']};">
          <div class="glossary-icon">{meta['icon']}</div>
          <div class="glossary-term" style="color:{meta['color']};">{meta['label']}</div>
          <div class="glossary-desc">{meta['desc']}</div>
        </div>
        """)
    sections.append("</div>")

    sections.append("<h2 class='glossary-title' style='margin-top:2rem;'>Overlap Metrics</h2>")
    sections.append('<div class="glossary-grid">')
    for key, meta in METRIC_META.items():
        sections.append(f"""
        <div class="glossary-card" style="border-top:4px solid #3498db;">
          <div class="glossary-term" style="color:#3498db;">{meta['label']}</div>
          <div class="glossary-desc">{meta['desc']}</div>
        </div>
        """)
    sections.append("</div>")

    sections.append("<h2 class='glossary-title' style='margin-top:2rem;'>Unlearning Methods</h2>")
    sections.append('<div class="glossary-grid">')
    for method, meta in METHOD_META.items():
        color = METHOD_COLORS.get(method, "#555")
        sections.append(f"""
        <div class="glossary-card" style="border-top:4px solid {color};">
          <div class="glossary-term" style="color:{color};">{meta['label']}</div>
          <div class="glossary-desc">{meta['desc']}</div>
        </div>
        """)
    sections.append("</div>")

    return "\n".join(sections)

# ---------------------------------------------------------------------------
# Cross-method charts section  (no scatter)
# ---------------------------------------------------------------------------
def cross_method_charts_html():
    chart_ids = [
        ("flip_dist",
         "Outcome Distribution Across Methods",
         "Stacked bars showing how many retain users fall into each outcome per method. "
         "Segments are annotated with their share of the total. "
         "A good method maximises stable_correct and minimises flipped_harmful."),
        ("overlap_2gram",
         "2-gram Overlap % by Outcome & Method",
         "Mean 2-gram overlap (± 1 std) for each outcome group, annotated with the mean value. "
         "If flipped_harmful were significantly higher than stable_correct, "
         "sequence similarity would explain the retain drop."),
        ("overlap_3gram",
         "3-gram Overlap % by Outcome & Method",
         "Same as above at 3-gram resolution — higher specificity, so a match is more meaningful."),
        ("max_single",
         "Max Single-User Overlap % by Outcome & Method",
         "The highest n-gram overlap any retain user has with a single forget user, "
         "averaged over each outcome group."),
    ]
    html = ""
    for cid, title, caption in chart_ids:
        if cid not in charts:
            continue
        html += f"""
        <div class="cross-chart-block">
          <h3 class="section-title">{title}</h3>
          <p class="chart-caption">{caption}</p>
          <div id="{cid}" class="chart-container"></div>
        </div>
        """
    return html

# ---------------------------------------------------------------------------
# Assemble tabs
# ---------------------------------------------------------------------------
tab_buttons  = ""
tab_contents = ""
for i, method in enumerate(METHODS):
    if method not in all_data:
        continue
    m_meta  = METHOD_META.get(method, {"label": method})
    active  = "active" if i == 0 else ""
    color   = METHOD_COLORS.get(method, "#555")
    tab_buttons += (
        f'<button class="tab-btn {active}" onclick="showTab(\'{method}\')" '
        f'id="btn_{method}" style="border-bottom-color:{color if active else "transparent"};">'
        f'{m_meta["label"]}</button>'
    )
    display = "block" if i == 0 else "none"
    tab_contents += (
        f'<div id="tab_{method}" class="tab-content" style="display:{display};">'
        f'{method_tab_content(method, all_data[method])}'
        f'</div>'
    )

charts_json = json.dumps(charts)

# ---------------------------------------------------------------------------
# Full HTML  — C: glossary is FIRST section after header
# ---------------------------------------------------------------------------
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sequence Overlap Analysis — {FORGET_PERCENTAGE}% Forget</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', 'Segoe UI', sans-serif;
      background: #f0f2f5;
      color: #1a1a2e;
      line-height: 1.6;
    }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
      color: #fff;
      padding: 2.5rem 3rem;
    }}
    .header h1 {{ font-size: 2rem; font-weight: 700; }}
    .header .subtitle {{ margin-top: .4rem; font-size: 1.05rem; opacity: .8; }}
    .header .meta-pills {{ display: flex; gap: .75rem; margin-top: 1.2rem; flex-wrap: wrap; }}
    .meta-pill {{
      background: rgba(255,255,255,.15);
      border: 1px solid rgba(255,255,255,.25);
      border-radius: 20px;
      padding: .3rem .9rem;
      font-size: .85rem;
    }}

    /* ── Layout ── */
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem 2.5rem; }}
    .section {{
      background: #fff; border-radius: 12px; padding: 2rem;
      margin-bottom: 2rem; box-shadow: 0 2px 12px rgba(0,0,0,.07);
    }}

    /* ── Tabs ── */
    .tab-nav {{
      display: flex; border-bottom: 2px solid #e0e0e0;
      background: #fff; border-radius: 12px 12px 0 0;
      overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .tab-btn {{
      flex: 1; padding: .9rem 1.2rem; border: none; background: transparent;
      cursor: pointer; font-size: .95rem; font-weight: 600; color: #666;
      border-bottom: 3px solid transparent; transition: all .2s; white-space: nowrap;
    }}
    .tab-btn:hover {{ background: #f5f5f5; color: #333; }}
    .tab-btn.active {{ color: #1a1a2e; background: #fff; }}
    .tab-content-wrapper {{
      background: #fff; border-radius: 0 0 12px 12px;
      padding: 2rem; box-shadow: 0 4px 12px rgba(0,0,0,.07);
      margin-bottom: 2rem;
    }}

    /* ── Method header ── */
    .method-header {{
      padding: 1.2rem 1.5rem; border-radius: 8px;
      background: #fafafa; margin-bottom: 1.5rem;
    }}
    .method-header h2 {{ font-size: 1.4rem; color: #1a1a2e; }}
    .method-desc {{ color: #555; margin-top: .4rem; font-size: .95rem; }}

    /* ── Section titles ── */
    .section-title {{
      font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
      margin: 1.8rem 0 .8rem; padding-bottom: .4rem;
      border-bottom: 2px solid #f0f2f5;
    }}

    /* ── Stat cards ── */
    .stat-grid {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
    .stat-card {{
      flex: 1; min-width: 130px; background: #fafafa;
      border-radius: 10px; padding: 1.2rem 1rem; text-align: center;
      box-shadow: 0 1px 6px rgba(0,0,0,.06);
    }}
    .stat-icon {{ font-size: 1.6rem; margin-bottom: .3rem; }}
    .stat-value {{ font-size: 2rem; font-weight: 800; }}
    .stat-pct {{ font-size: .9rem; color: #777; }}
    .stat-label {{ font-size: .8rem; color: #555; margin-top: .3rem; font-weight: 600; }}

    /* ── Info grid ── */
    .info-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: .8rem; margin-bottom: 1.5rem;
    }}
    .info-item {{
      background: #f7f8fa; border-radius: 8px; padding: .7rem 1rem;
      display: flex; flex-direction: column; gap: .2rem;
    }}
    .info-label {{ font-size: .75rem; color: #888; text-transform: uppercase; letter-spacing: .05em; }}
    .info-value {{ font-size: .95rem; font-weight: 600; color: #1a1a2e; }}

    /* ── Flip badge ── */
    .flip-badge {{
      display: inline-block; border-radius: 12px;
      padding: .2rem .7rem; font-size: .82rem; font-weight: 600; white-space: nowrap;
    }}

    /* ── Summary table ── */
    .table-wrapper {{ overflow-x: auto; }}
    .summary-table {{
      width: 100%; border-collapse: collapse; font-size: .88rem;
    }}
    .summary-table th {{
      background: #f0f2f5; padding: .6rem .8rem;
      text-align: center; font-weight: 700; color: #333;
      border-bottom: 2px solid #ddd;
    }}
    .summary-table td {{
      padding: .55rem .8rem; border-bottom: 1px solid #f0f0f0;
      text-align: center; color: #444;
    }}
    .summary-table tr:hover {{ background: #fafafa; }}
    .summary-table tr:last-child td {{ border-bottom: none; }}

    /* ── Charts ── */
    .chart-container {{ width: 100%; min-height: 420px; margin-bottom: 1rem; }}
    .chart-caption {{ color: #666; font-size: .9rem; margin-bottom: .8rem; font-style: italic; }}
    .cross-chart-block {{ margin-bottom: 2.5rem; }}

    /* ── Glossary ── */
    .glossary-title {{ font-size: 1.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 1rem; }}
    .glossary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1rem;
    }}
    .glossary-card {{
      background: #fafafa; border-radius: 10px; padding: 1.2rem;
      box-shadow: 0 1px 6px rgba(0,0,0,.06);
    }}
    .glossary-icon {{ font-size: 1.5rem; margin-bottom: .4rem; }}
    .glossary-term {{ font-size: 1rem; font-weight: 700; margin-bottom: .4rem; }}
    .glossary-desc {{ font-size: .88rem; color: #555; line-height: 1.5; }}

    /* ── Footer ── */
    .footer {{
      text-align: center; color: #aaa; font-size: .82rem;
      padding: 1.5rem; margin-top: 1rem;
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>🔍 Sequence Overlap Analysis</h1>
  <div class="subtitle">
    Investigating whether retain users harmed by unlearning are
    more behaviourally similar to the forget set
  </div>
  <div class="meta-pills">
    <span class="meta-pill">📁 Forget set: <strong>{FORGET_PERCENTAGE}%</strong></span>
    <span class="meta-pill">📊 Methods: <strong>{len(all_data)}</strong></span>
    <span class="meta-pill">📐 Evaluation K: <strong>10</strong></span>
    <span class="meta-pill">🧬 Prefix-only n-grams</span>
  </div>
</div>

<div class="container">

  <!-- ── C: Glossary FIRST ───────────────────────────────── -->
  <div class="section">
    <h2 style="font-size:1.3rem;margin-bottom:1.5rem;">📖 Definitions</h2>
    {glossary_html()}
  </div>

  <!-- ── Cross-method charts ─────────────────────────────── -->
  <div class="section">
    <h2 style="font-size:1.3rem;margin-bottom:1.5rem;">Cross-Method Comparison</h2>
    {cross_method_charts_html()}
  </div>

  <!-- ── Per-method tabs ─────────────────────────────────── -->
  <div class="tab-nav">
    {tab_buttons}
  </div>
  <div class="tab-content-wrapper">
    {tab_contents}
  </div>

</div>

<div class="footer">
  Generated by sequence_overlap.py &nbsp;|&nbsp;
  Forget {FORGET_PERCENTAGE}% &nbsp;|&nbsp;
  {len(all_data)} method(s) analysed
</div>

<script>
const CHARTS = {charts_json};

function renderChart(divId, chartKey) {{
  const el = document.getElementById(divId);
  if (!el || !CHARTS[chartKey]) return;
  const spec = JSON.parse(CHARTS[chartKey]);
  Plotly.newPlot(el, spec.data, spec.layout, {{responsive: true, displayModeBar: false}});
}}

// Render cross-method charts on load
['flip_dist','overlap_2gram','overlap_3gram','max_single'].forEach(k => renderChart(k, k));

// Tab switching
function showTab(method) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  const tab = document.getElementById('tab_' + method);
  const btn = document.getElementById('btn_' + method);
  if (tab) tab.style.display = 'block';
  if (btn) btn.classList.add('active');
  setTimeout(() => {{
    document.querySelectorAll('#tab_' + method + ' .chart-container').forEach(el => {{
      if (el.children.length > 0) Plotly.Plots.resize(el);
    }});
  }}, 50);
}}
</script>

</body>
</html>"""

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"\n✓ Dashboard saved → {OUTPUT_HTML}")
print(f"  Open in browser: file:///{str(OUTPUT_HTML).replace(chr(92), '/')}")
