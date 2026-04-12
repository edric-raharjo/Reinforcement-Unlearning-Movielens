#!/usr/bin/env python
# coding: utf-8

"""
lambda_dashboard.py

Interactive dashboard for analyzing the effect of lambda on forgetting quality.

Controls:
- forget percentage
- K in {1, 5, 10}
- retain-drop threshold

Plots:
1. Best forget quality by lambda after filtering by the selected forget percentage,
   K, and retain-drop threshold.
2. Average forget quality by lambda across all matching attempts.

The script preloads all available result rows into a single HTML file so the
browser can switch filters without re-reading any files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_ANALYSIS = Path("D:/Bob_Skripsi_Do Not Delete/Analysis")
BASE_RESULTS = Path("D:/Bob_Skripsi_Do Not Delete/results")
BASE_RESULTS_DEMO = Path("D:/Bob_Skripsi_Do Not Delete/results_demography")

MODE = os.environ.get("RUN_MODE", "Normal")
MODE_DIR = "Demography" if MODE == "Demography" else "Normal"

METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max", "Gradient_Ascent"]
METHOD_COLORS = {
    "Ye_ApxI": "#16a34a",
    "Ye_multi": "#d97706",
    "New_True_inf": "#2563eb",
    "New_Max": "#dc2626",
    "Gradient_Ascent": "#9333ea",
}

DEFAULT_PCTS = [1, 2, 3, 4, 5, 20]
KS = [1, 5, 10]
THRESHOLDS = [0, 1, 2, 5, 10]


def normalize_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def results_root() -> Path:
    if MODE == "Demography":
        return BASE_RESULTS_DEMO
    return BASE_RESULTS


def resolve_results_path(forget_pct: int) -> Path | None:
    roots = [results_root(), BASE_ANALYSIS, BASE_RESULTS, BASE_RESULTS_DEMO]
    seen: set[Path] = set()

    for root in roots:
        if root in seen:
            continue
        seen.add(root)

        candidates = [
            root / f"{forget_pct}_percent" / "tuning_full_results.csv",
            root / MODE_DIR / f"{forget_pct}_percent" / "tuning_full_results.csv",
            root / MODE_DIR / f"{forget_pct}_percent" / "analyze" / "overall" / "tuning_full_results.csv",
            root / f"{forget_pct}_percent" / "analyze" / "overall" / "tuning_full_results.csv",
            root / MODE_DIR / f"{forget_pct}_percent" / "analyze" / "diagnose" / "tuning_full_results.csv",
            root / f"{forget_pct}_percent" / "analyze" / "diagnose" / "tuning_full_results.csv",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("tuning_full_results.csv"):
            parts = {part.lower() for part in candidate.parts}
            if f"{forget_pct}_percent" in parts:
                has_demo = any("demography" in part for part in candidate.parts)
                if (MODE == "Demography" and has_demo) or (MODE != "Demography" and not has_demo):
                    return candidate

    return None


def load_results_for_pct(forget_pct: int) -> pd.DataFrame:
    results_path = resolve_results_path(forget_pct)
    if results_path is None:
        return pd.DataFrame()

    df = pd.read_csv(results_path)
    numeric_cols = [
        "K", "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain",
        "train_time_s", "unlearn_time_s",
        "base_retain_Hit", "base_retain_NDCG",
        "base_forget_Hit", "base_forget_NDCG",
        "retain_Hit", "retain_NDCG",
        "forget_Hit", "forget_NDCG",
        "loss_forget_final", "loss_retain_final", "loss_total_final",
    ]
    df = normalize_numeric(df.copy(), numeric_cols)

    if "method" not in df.columns:
        return pd.DataFrame()

    df = df[df["method"].isin(METHODS)].copy()
    if df.empty:
        return df

    df["forget_pct"] = int(forget_pct)
    df["retain_drop_pp"] = (df["base_retain_Hit"] - df["retain_Hit"]) * 100.0
    df["forget_drop_pp"] = (df["base_forget_Hit"] - df["forget_Hit"]) * 100.0
    df["forget_quality_pp"] = df["forget_drop_pp"] - df["retain_drop_pp"]

    keep_cols = [
        "forget_pct", "K", "method", "lambda_retain",
        "retain_drop_pp", "forget_drop_pp", "forget_quality_pp",
        "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters",
        "base_retain_Hit", "base_forget_Hit", "retain_Hit", "forget_Hit",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].copy()


def load_all_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen_pcts: list[int] = []

    for pct in DEFAULT_PCTS:
        df = load_results_for_pct(pct)
        if not df.empty:
            frames.append(df)
            seen_pcts.append(pct)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df["lambda_retain"] = pd.to_numeric(all_df["lambda_retain"], errors="coerce")
    all_df = all_df.dropna(subset=["lambda_retain", "retain_drop_pp", "forget_drop_pp", "forget_quality_pp"])
    all_df = all_df[all_df["method"].isin(METHODS)].copy()
    all_df["K"] = pd.to_numeric(all_df["K"], errors="coerce")
    all_df["forget_pct"] = pd.to_numeric(all_df["forget_pct"], errors="coerce")
    all_df = all_df.dropna(subset=["K", "forget_pct"])
    all_df["K"] = all_df["K"].astype(int)
    all_df["forget_pct"] = all_df["forget_pct"].astype(int)
    return all_df


def build_html(data_df: pd.DataFrame, available_pcts: list[int], output_path: Path, warning_message: str | None = None) -> str:
    records = data_df.to_dict(orient="records")
    payload = json.dumps(records, default=float)

    controls_pct = "".join(
        f'<option value="{pct}">{pct}%</option>' for pct in available_pcts
    )
    controls_k = "".join(f'<option value="{k}">{k}</option>' for k in KS)
    controls_thr = "".join(f'<option value="{thr}">{thr} pp</option>' for thr in THRESHOLDS)

    method_colors = json.dumps(METHOD_COLORS)
    methods = json.dumps(METHODS)
    pcts = json.dumps(available_pcts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Lambda Dashboard - {MODE_DIR}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --bg: #f1f5f9;
            --card: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --line: #e2e8f0;
            --accent: #2563eb;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 18%, #f8fafc 100%); color: var(--text); font-family: 'Inter', sans-serif; }}
        .container {{ max-width: 1500px; margin: 0 auto; padding: 36px 28px 72px; }}
        h1 {{ margin: 0; font-size: 34px; letter-spacing: -0.8px; font-weight: 800; }}
        .subtitle {{ margin: 8px 0 20px; color: var(--muted); line-height: 1.6; }}
        .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); padding: 22px; margin-bottom: 20px; }}
        .controls {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 14px; align-items: end; }}
        .control {{ display: flex; flex-direction: column; gap: 8px; }}
        label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 700; }}
        select {{ width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; font: inherit; background: #fff; color: var(--text); }}
        .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
        .stat {{ background: #f8fafc; border: 1px solid var(--line); border-radius: 16px; padding: 16px; }}
        .stat .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}
        .stat .v {{ margin-top: 8px; font-size: 20px; font-weight: 800; letter-spacing: -0.02em; }}
        .section-title {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }}
        .section-title h2 {{ margin: 0; font-size: 20px; font-weight: 800; letter-spacing: -0.4px; }}
        .section-title .hint {{ color: var(--muted); font-size: 13px; }}
        .plot {{ width: 100%; height: 520px; }}
        .mini {{ color: var(--muted); font-size: 14px; line-height: 1.7; margin: 0; }}
        @media (max-width: 960px) {{
            .controls, .summary {{ grid-template-columns: 1fr; }}
            .container {{ padding: 24px 14px 56px; }}
            h1 {{ font-size: 28px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="panel">
            <h1>Lambda Dashboard</h1>
            <p class="subtitle">
                Mode: <b>{MODE_DIR}</b> &middot; the dashboard preloads all matching rows once and updates the charts in-browser.
                Forget quality is defined as <b>forget_drop - retain_drop</b> in percentage points.
            </p>
            {f'<div style="margin:0 0 18px;padding:12px 14px;border-radius:12px;background:#fff7ed;border:1px solid #fdba74;color:#9a3412;font-weight:600;">{warning_message}</div>' if warning_message else ''}
            <div class="controls">
                <div class="control">
                    <label for="forgetPct">Forget percentage</label>
                    <select id="forgetPct">{controls_pct}</select>
                </div>
                <div class="control">
                    <label for="kValue">K</label>
                    <select id="kValue">{controls_k}</select>
                </div>
                <div class="control">
                    <label for="retainThreshold">Retain-drop threshold</label>
                    <select id="retainThreshold">{controls_thr}</select>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="summary">
                <div class="stat"><div class="k">Rows loaded</div><div class="v" id="rowsLoaded">0</div></div>
                <div class="stat"><div class="k">Matching rows</div><div class="v" id="rowsMatched">0</div></div>
                <div class="stat"><div class="k">Lambda values</div><div class="v" id="lambdaCount">0</div></div>
                <div class="stat"><div class="k">Best quality</div><div class="v" id="bestQuality">N/A</div></div>
            </div>
        </div>

        <div class="panel">
            <div class="section-title">
                <h2>Best Forget Quality by Lambda</h2>
                <div class="hint">For each method and lambda, the best forget-drop run after filtering is shown.</div>
            </div>
            <div id="bestPlot" class="plot"></div>
        </div>

        <div class="panel">
            <div class="section-title">
                <h2>Average Forget Quality by Lambda</h2>
                <div class="hint">Average across all matching attempts for each lambda.</div>
            </div>
            <div id="avgPlot" class="plot"></div>
        </div>

        <div class="panel">
            <p class="mini">
                Filter rule: retain_drop must be less than or equal to the selected threshold.
                The first chart chooses the row with the highest forget_drop within each method-lambda slice.
                The second chart averages forget quality across all remaining rows for each lambda.
            </p>
        </div>
    </div>

    <script>
        const METHODS = {methods};
        const COLORS = {method_colors};
        const DATA = {payload};
        const AVAILABLE_PCTS = {pcts};

        const forgetPctEl = document.getElementById('forgetPct');
        const kValueEl = document.getElementById('kValue');
        const retainThresholdEl = document.getElementById('retainThreshold');

        function getSelectedPct() {{ return Number(forgetPctEl.value); }}
        function getSelectedK() {{ return Number(kValueEl.value); }}
        function getSelectedThreshold() {{ return Number(retainThresholdEl.value); }}

        function formatMaybe(v) {{
            if (v === null || v === undefined || Number.isNaN(v)) return 'N/A';
            return Number(v).toFixed(2);
        }}

        function filterRows() {{
            const pct = getSelectedPct();
            const kVal = getSelectedK();
            const threshold = getSelectedThreshold();
            return DATA.filter(row =>
                Number(row.forget_pct) === pct &&
                Number(row.K) === kVal &&
                Number(row.retain_drop_pp) <= threshold
            );
        }}

        function buildBestByLambda(rows) {{
            const byMethod = new Map();

            METHODS.forEach(method => byMethod.set(method, new Map()));

            rows.forEach(row => {{
                const method = row.method;
                if (!byMethod.has(method)) return;
                const lambdaKey = String(row.lambda_retain);
                const current = byMethod.get(method).get(lambdaKey);
                if (!current || Number(row.forget_drop_pp) > Number(current.forget_drop_pp)) {{
                    byMethod.get(method).set(lambdaKey, row);
                }}
            }});

            const traces = [];
            METHODS.forEach(method => {{
                const entries = Array.from(byMethod.get(method).values())
                    .sort((a, b) => Number(a.lambda_retain) - Number(b.lambda_retain));
                if (!entries.length) return;

                traces.push({{
                    x: entries.map(row => Number(row.lambda_retain)),
                    y: entries.map(row => Number(row.forget_quality_pp)),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: method,
                    line: {{ color: COLORS[method] || '#64748b', width: 3 }},
                    marker: {{ color: COLORS[method] || '#64748b', size: 8 }},
                    customdata: entries.map(row => [row.forget_drop_pp, row.retain_drop_pp]),
                    hovertemplate:
                        'lambda=%{{x}}<br>' +
                        'forget quality=%{{y:.2f}} pp<br>' +
                        'forget drop=%{{customdata[0]:.2f}} pp<br>' +
                        'retain drop=%{{customdata[1]:.2f}} pp<extra></extra>',
                }});
            }});

            return traces;
        }}

        function buildAverageByLambda(rows) {{
            const groups = new Map();

            rows.forEach(row => {{
                const lambdaKey = String(row.lambda_retain);
                if (!groups.has(lambdaKey)) groups.set(lambdaKey, []);
                groups.get(lambdaKey).push(row);
            }});

            const sorted = Array.from(groups.entries())
                .map(([lambdaKey, items]) => {{
                    const lambdaValue = Number(lambdaKey);
                    const avgQuality = items.reduce((sum, row) => sum + Number(row.forget_quality_pp), 0) / items.length;
                    return {{
                        lambda: lambdaValue,
                        avgQuality,
                        count: items.length,
                    }};
                }})
                .sort((a, b) => a.lambda - b.lambda);

            return sorted;
        }}

        function updateSummary(rows, avgRows) {{
            document.getElementById('rowsLoaded').textContent = DATA.length.toLocaleString();
            document.getElementById('rowsMatched').textContent = rows.length.toLocaleString();
            document.getElementById('lambdaCount').textContent = avgRows.length.toLocaleString();

            if (!rows.length) {{
                document.getElementById('bestQuality').textContent = 'N/A';
                return;
            }}

            const best = rows.reduce((acc, row) => Number(row.forget_quality_pp) > Number(acc.forget_quality_pp) ? row : acc, rows[0]);
            document.getElementById('bestQuality').textContent = formatMaybe(best.forget_quality_pp) + ' pp';
        }}

        function renderPlots() {{
            const rows = filterRows();
            const bestTraces = buildBestByLambda(rows);
            const avgRows = buildAverageByLambda(rows);

            updateSummary(rows, avgRows);

            const bestLayout = {{
                template: 'plotly_white',
                title: {{ text: 'Best Forget Quality by Lambda', font: {{ size: 18 }} }},
                xaxis: {{ title: 'Lambda', zeroline: false }},
                yaxis: {{ title: 'Forget Quality (pp)', zeroline: false }},
                margin: {{ l: 70, r: 24, t: 60, b: 60 }},
                legend: {{ orientation: 'h', y: -0.18 }},
            }};

            const avgLayout = {{
                template: 'plotly_white',
                title: {{ text: 'Average Forget Quality by Lambda', font: {{ size: 18 }} }},
                xaxis: {{ title: 'Lambda', zeroline: false }},
                yaxis: {{ title: 'Average Forget Quality (pp)', zeroline: false }},
                margin: {{ l: 70, r: 24, t: 60, b: 60 }},
                showlegend: false,
            }};

            if (bestTraces.length === 0) {{
                Plotly.react('bestPlot', [], Object.assign({{}}, bestLayout, {{
                    annotations: [{{
                        text: 'No rows match the current filters.',
                        xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
                        showarrow: false,
                        font: {{ size: 16, color: '#64748b' }},
                    }}],
                }}), {{ responsive: true }});
            }} else {{
                Plotly.react('bestPlot', bestTraces, bestLayout, {{ responsive: true }});
            }}

            if (avgRows.length === 0) {{
                Plotly.react('avgPlot', [], Object.assign({{}}, avgLayout, {{
                    annotations: [{{
                        text: 'No rows match the current filters.',
                        xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
                        showarrow: false,
                        font: {{ size: 16, color: '#64748b' }},
                    }}],
                }}), {{ responsive: true }});
            }} else {{
                Plotly.react('avgPlot', [{{
                    x: avgRows.map(item => item.lambda),
                    y: avgRows.map(item => item.avgQuality),
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ color: '#0f766e', width: 3 }},
                    marker: {{ color: '#0f766e', size: 8 }},
                    customdata: avgRows.map(item => item.count),
                    hovertemplate:
                        'lambda=%{{x}}<br>' +
                        'avg forget quality=%{{y:.2f}} pp<br>' +
                        'attempts=%{{customdata}}<extra></extra>',
                }}], avgLayout, {{ responsive: true }});
            }}
        }}

        forgetPctEl.addEventListener('change', renderPlots);
        kValueEl.addEventListener('change', renderPlots);
        retainThresholdEl.addEventListener('change', renderPlots);

        function setDefault(selectEl, value) {{
            const options = Array.from(selectEl.options).map(opt => Number(opt.value));
            const match = options.includes(value) ? value : options[0];
            selectEl.value = String(match);
        }}

        setDefault(forgetPctEl, AVAILABLE_PCTS[0]);
        setDefault(kValueEl, 10);
        setDefault(retainThresholdEl, 5);

        renderPlots();
    </script>
</body>
</html>
"""


def main() -> None:
    data_df = load_all_rows()
    if data_df.empty:
        available_pcts = DEFAULT_PCTS
        warning_message = f"No tuning_full_results.csv files were found for mode={MODE_DIR}. The dashboard shell was generated without data."
    else:
        available_pcts = sorted({int(v) for v in data_df["forget_pct"].dropna().unique().tolist()})
        warning_message = None

    output_path = BASE_ANALYSIS / MODE_DIR / "lambda_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = build_html(data_df, available_pcts, output_path, warning_message)
    output_path.write_text(html, encoding="utf-8")
    print(f"Lambda dashboard written to {output_path}")
    print(f"Loaded rows: {len(data_df):,}")


if __name__ == "__main__":
    main()