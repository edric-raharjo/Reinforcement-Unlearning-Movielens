#!/usr/bin/env python
# coding: utf-8

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BASE_ANALYSIS = Path("D:/Bob_Skripsi_Do Not Delete/Analysis")
BASE_RESULTS = Path("D:/Bob_Skripsi_Do Not Delete/results")
BASE_RESULTS_DEMO = Path("D:/Bob_Skripsi_Do Not Delete/results_demography")
ALT_CANDIDATE_ROOTS = [
    BASE_RESULTS,
    BASE_RESULTS_DEMO,
    Path("C:/Bob/results"),
    Path("C:/Bob/results_demography"),
]
METHODS = ["Ye_ApxI", "Ye_multi", "New_True_inf", "New_Max"]
METRICS = ["Hit", "NDCG"]
MODES = {
    "Normal": [1, 2, 3, 4, 5, 20],
    "Demography": [1, 2, 3, 4, 5],
}
THRESHOLDS = [1, 5, 10]
KS = [1, 5, 10]

PLOT_COLORS = {
    "Ye_ApxI": "#16a34a",
    "Ye_multi": "#d97706",
    "New_True_inf": "#2563eb",
    "New_Max": "#dc2626",
}

MODE_STYLES = {
    "Normal": {"dash": "solid"},
    "Demography": {"dash": "dash"},
}


def parse_args():
    return int(sys.argv[1]) if len(sys.argv) > 1 else 8


def normalize_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_results(df):
    numeric_cols = [
        "K", "train_lr", "gamma", "hidden_dim", "train_batch",
        "unlearn_lr", "unlearn_iters", "lambda_retain",
        "train_time_s", "unlearn_time_s",
        "base_retain_Hit", "base_retain_NDCG",
        "base_forget_Hit", "base_forget_NDCG",
        "retain_Hit", "retain_NDCG",
        "forget_Hit", "forget_NDCG",
        "base_combined_Hit", "base_combined_NDCG",
        "loss_forget_final", "loss_retain_final", "loss_total_final",
    ]
    df = normalize_numeric(df.copy(), numeric_cols)

    if "method" not in df.columns:
        raise ValueError("Results file is missing the 'method' column.")

    df = df[df["method"].isin(METHODS)].copy()

    df["retain_drop_hit"] = df["base_retain_Hit"] - df["retain_Hit"]
    df["forget_drop_hit"] = df["base_forget_Hit"] - df["forget_Hit"]
    df["retain_drop_ndcg"] = df["base_retain_NDCG"] - df["retain_NDCG"]
    df["forget_drop_ndcg"] = df["base_forget_NDCG"] - df["forget_NDCG"]

    df["retain_drop_hit_pp"] = df["retain_drop_hit"] * 100.0
    df["forget_drop_hit_pp"] = df["forget_drop_hit"] * 100.0
    df["retain_drop_ndcg_pp"] = df["retain_drop_ndcg"] * 100.0
    df["forget_drop_ndcg_pp"] = df["forget_drop_ndcg"] * 100.0

    return df


def resolve_input_file(mode, pct, filename):
    roots = [BASE_ANALYSIS, BASE_RESULTS, BASE_RESULTS_DEMO] + ALT_CANDIDATE_ROOTS
    seen = set()
    candidates = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates.extend([
            root / mode / f"{pct}_percent" / filename,
            root / f"{pct}_percent" / filename,
            root / mode / f"{pct}_percent" / "analyze" / "overall" / filename,
            root / f"{pct}_percent" / "analyze" / "overall" / filename,
            root / mode / f"{pct}_percent" / "analyze" / "diagnose" / filename,
            root / f"{pct}_percent" / "analyze" / "diagnose" / filename,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob(filename):
            parts = {part.lower() for part in candidate.parts}
            if f"{pct}_percent" in parts:
                if mode.lower() == "normal" and ("demography" not in parts):
                    return candidate
                if mode.lower() == "demography" and ("demography" in parts):
                    return candidate

    return None


def load_csvs_for_mode_pct(mode, pct):
    if mode == "Demography" and pct == 20:
        return None, None, [f"{mode} {pct}% is intentionally unavailable and was skipped."]

    results_path = resolve_input_file(mode, pct, "tuning_full_results.csv")
    train_path = resolve_input_file(mode, pct, "train_phase_results.csv")

    alerts = []
    if results_path is None:
        alerts.append(f"Missing results file for {mode} {pct}% (searched Analysis/results/results_demography roots).")
        return None, None, alerts

    results_df = prepare_results(pd.read_csv(results_path))
    train_df = pd.read_csv(train_path) if train_path is not None and train_path.exists() else pd.DataFrame()
    if train_df.empty:
        alerts.append(f"Missing train phase file for {mode} {pct}% (top-model filtering will fall back to all runs).")

    return results_df, train_df, alerts


def key_from_row(row):
    return (
        float(row["train_lr"]),
        float(row["gamma"]),
        int(row["hidden_dim"]),
        int(row["train_batch"]),
    )


def select_top_configs(train_df, k_val, num_top_models):
    if train_df is None or train_df.empty:
        return None, [f"No train phase data available for K={k_val}; using all runs."]

    if "K" not in train_df.columns:
        return None, ["Train results are missing K column; using all runs."]

    sort_metric = "base_combined_Hit" if "base_combined_Hit" in train_df.columns else "base_retain_Hit"

    rank_df = (
        train_df[train_df["K"] == k_val]
        .sort_values(
            [sort_metric, "train_lr", "gamma", "hidden_dim", "train_batch"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["train_lr", "gamma", "hidden_dim", "train_batch"], keep="first")
        .reset_index(drop=True)
    )

    if rank_df.empty:
        return None, [f"No train configs found for K={k_val}; using all runs."]

    top_df = rank_df.head(num_top_models)
    top_configs = {key_from_row(row) for _, row in top_df.iterrows()}
    return top_configs, [
        f"Top {len(top_configs)} configs selected for K={k_val} using {sort_metric}."
    ]


def filter_results_to_top_configs(results_df, top_configs):
    if not top_configs:
        return results_df

    keys = results_df.apply(key_from_row, axis=1)
    mask = keys.isin(top_configs)
    filtered = results_df[mask].copy()
    return filtered if not filtered.empty else results_df


def select_method_row(df, metric, threshold, method):
    retain_col = f"retain_drop_{metric.lower()}_pp"
    forget_col = f"forget_drop_{metric.lower()}_pp"
    mdf = df[df["method"] == method].copy()

    if mdf.empty:
        return None, f"{method}: no data available."

    constrained = mdf[mdf[retain_col] <= threshold].copy()
    if not constrained.empty:
        chosen = constrained.sort_values(
            [forget_col, retain_col, "unlearn_iters"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0].copy()
        note = f"{method}: selected within threshold {threshold} for {metric}."
        return chosen, note

    chosen = mdf.sort_values(
        [forget_col, retain_col, "unlearn_iters"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0].copy()
    note = f"{method}: threshold {threshold} had no match for {metric}; fallback row used."
    return chosen, note


def build_chart_data(num_top_models):
    chart_data = {
        metric: {
            "retain": {str(thr): {str(k): {mode: {} for mode in MODES} for k in KS} for thr in THRESHOLDS},
            "forget": {str(thr): {str(k): {mode: {} for mode in MODES} for k in KS} for thr in THRESHOLDS},
        }
        for metric in METRICS
    }

    alerts = []
    coverage = []

    for mode, pcts in MODES.items():
        for pct in pcts:
            results_df, train_df, load_alerts = load_csvs_for_mode_pct(mode, pct)
            alerts.extend(load_alerts)

            if results_df is None:
                continue

            for k_val in KS:
                top_configs, top_notes = select_top_configs(train_df, k_val, num_top_models)
                if top_configs is None and any("Missing" in msg or "missing" in msg for msg in top_notes):
                    alerts.extend([f"{mode} {pct}% K={k_val}: {msg}" for msg in top_notes])
                filtered_results = filter_results_to_top_configs(results_df, top_configs)
                k_df = filtered_results[filtered_results["K"] == k_val].copy()
                if k_df.empty:
                    alerts.append(f"{mode} {pct}% K={k_val}: no rows left after top-model filtering.")
                    continue

                for metric in METRICS:
                    retain_col = f"retain_drop_{metric.lower()}_pp"
                    forget_col = f"forget_drop_{metric.lower()}_pp"

                    for threshold in THRESHOLDS:
                        for method in METHODS:
                            row, note = select_method_row(k_df, metric, threshold, method)
                            bucket_retain = chart_data[metric]["retain"][str(threshold)][str(k_val)][mode]
                            bucket_forget = chart_data[metric]["forget"][str(threshold)][str(k_val)][mode]

                            bucket_retain.setdefault(method, {"x": [], "y": [], "notes": []})
                            bucket_forget.setdefault(method, {"x": [], "y": [], "notes": []})

                            if row is None:
                                bucket_retain[method]["notes"].append(note)
                                bucket_forget[method]["notes"].append(note)
                                continue

                            bucket_retain[method]["x"].append(int(pct))
                            bucket_retain[method]["y"].append(float(row[retain_col]))
                            bucket_retain[method]["notes"].append(note)

                            bucket_forget[method]["x"].append(int(pct))
                            bucket_forget[method]["y"].append(float(row[forget_col]))
                            bucket_forget[method]["notes"].append(note)

                            coverage.append({
                                "mode": mode,
                                "pct": pct,
                                "k": k_val,
                                "metric": metric,
                                "threshold": threshold,
                                "method": method,
                                "retain_drop": float(row[retain_col]),
                                "forget_drop": float(row[forget_col]),
                                "fallback": "fallback" in note.lower(),
                            })

    for metric in METRICS:
        for drop_type in ["retain", "forget"]:
            for thr in THRESHOLDS:
                for k_val in KS:
                    for mode in MODES:
                        for method in METHODS:
                            series = chart_data[metric][drop_type][str(thr)][str(k_val)][mode].setdefault(method, {"x": [], "y": [], "notes": []})
                            if series["x"]:
                                ordered = sorted(zip(series["x"], series["y"]), key=lambda item: item[0])
                                series["x"] = [item[0] for item in ordered]
                                series["y"] = [item[1] for item in ordered]

    return chart_data, alerts, coverage


def build_summary_rows(coverage):
    return ""


def build_alert_list(alerts):
    if not alerts:
        return "<li>No missing files detected.</li>"

    unique_alerts = []
    seen = set()
    for alert in alerts:
        if alert not in seen:
            seen.add(alert)
            unique_alerts.append(alert)

    return "\n".join(f"<li>{alert}</li>" for alert in unique_alerts)


def build_html(chart_data, alerts, coverage, num_top_models):
    data_json = json.dumps(chart_data)
    alerts_html = build_alert_list(alerts)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Forget Comparison Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.24.1.min.js\"></script>
    <style>
        :root {{
            --bg: #f1f5f9;
            --card: #ffffff;
            --line: #e2e8f0;
            --text: #0f172a;
            --muted: #64748b;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg); color: var(--text); }}
        .container {{ max-width: 1500px; margin: 0 auto; padding: 28px 28px 48px; }}
        h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: -0.4px; }}
        .subtitle {{ color: var(--muted); margin: 0 0 18px; }}
        .panel {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04); }}
        .controls {{ display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; }}
        .control-group {{ min-width: 220px; }}
        .control-group h3 {{ margin: 0 0 8px; font-size: 14px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.6px; }}
        .radio-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .radio-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 7px 11px; border: 1px solid var(--line); border-radius: 999px; background: #fff; cursor: pointer; user-select: none; }}
        .radio-pill input {{ margin: 0; }}
            .mode-checklist {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .top-note {{ font-size: 13px; color: var(--muted); margin-top: 10px; }}
        .tab-bar {{ display: flex; gap: 10px; margin-bottom: 14px; }}
        .tab-btn {{ border: 1px solid var(--line); background: #fff; color: var(--text); padding: 10px 16px; border-radius: 12px; cursor: pointer; font-weight: 700; }}
        .tab-btn.active {{ background: #0f172a; color: white; border-color: #0f172a; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
        .chart-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px; }}
        .chart-title {{ margin: 0 0 8px; font-size: 16px; font-weight: 700; }}
        .chart-note {{ margin: 0 0 12px; font-size: 13px; color: var(--muted); }}
        .chart-box {{ width: 100%; min-height: 420px; }}
        .alerts ul {{ margin: 0; padding-left: 20px; }}
        .alerts li {{ margin-bottom: 6px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border-bottom: 1px solid var(--line); padding: 10px 8px; text-align: left; font-size: 14px; }}
        th {{ background: #f8fafc; font-size: 12px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.5px; }}
    </style>
</head>
<body>
<div class=\"container\">
    <h1>Forget Comparison Dashboard</h1>
    <p class=\"subtitle\">Top {num_top_models} models from train phase results. Solid line = Normal, dashed line = Demography. Click legend items to toggle methods.</p>

    <div class=\"panel controls\">
        <div class=\"control-group\">
            <h3>Threshold</h3>
            <div class=\"radio-row\">
                {''.join(f'<label class="radio-pill"><input type="radio" name="threshold" value="{t}" {"checked" if t == THRESHOLDS[0] else ""}> {t}</label>' for t in THRESHOLDS)}
            </div>
        </div>
        <div class=\"control-group\">
            <h3>K</h3>
            <div class=\"radio-row\">
                {''.join(f'<label class="radio-pill"><input type="radio" name="kval" value="{k}" {"checked" if k == KS[0] else ""}> {k}</label>' for k in KS)}
            </div>
        </div>
        <div class=\"control-group\">
            <h3>Info</h3>
            <div class=\"top-note\">The radio boxes switch among precomputed selections; the HTML is generated once and remains interactive offline.</div>
        </div>
            <div class="control-group">
                <h3>Mode</h3>
                <div class="mode-checklist">
                    <label class="radio-pill"><input type="checkbox" name="mode" value="Normal" checked> Normal</label>
                    <label class="radio-pill"><input type="checkbox" name="mode" value="Demography" checked> Demography</label>
                </div>
            </div>
    </div>

    <div class=\"panel alerts\">
        <h3 style=\"margin:0 0 8px; font-size:14px; text-transform:uppercase; color:var(--muted); letter-spacing:0.6px;\">Data Alerts</h3>
        <ul>{alerts_html}</ul>
    </div>

    <div class="tab-bar">
    <div class=\"tab-bar\">
        <button class=\"tab-btn active\" data-tab=\"Hit\" onclick=\"switchTab('Hit')\">Hit</button>
        <button class=\"tab-btn\" data-tab=\"NDCG\" onclick=\"switchTab('NDCG')\">NDCG</button>
    </div>

    <div id=\"tab-Hit\" class=\"tab-content active\">
        <div class=\"chart-grid\">
            <div class=\"chart-card\">
                <div class=\"chart-title\">Hit: Retain Drop</div>
                <div class=\"chart-note\">Methods are color-coded. Normal is solid; Demography is dashed.</div>
                <div id=\"Hit-retain\" class=\"chart-box\"></div>
            </div>
            <div class=\"chart-card\">
                <div class=\"chart-title\">Hit: Forget Drop</div>
                <div class=\"chart-note\">Higher forget drop is better; the line should rise when the method improves.</div>
                <div id=\"Hit-forget\" class=\"chart-box\"></div>
            </div>
        </div>
    </div>

    <div id=\"tab-NDCG\" class=\"tab-content\">
        <div class=\"chart-grid\">
            <div class=\"chart-card\">
                <div class=\"chart-title\">NDCG: Retain Drop</div>
                <div class=\"chart-note\">Retain drop should stay small or increase only minimally.</div>
                <div id=\"NDCG-retain\" class=\"chart-box\"></div>
            </div>
            <div class=\"chart-card\">
                <div class=\"chart-title\">NDCG: Forget Drop</div>
                <div class=\"chart-note\">This should show whether forgetting improves as the forget percentage increases.</div>
                <div id=\"NDCG-forget\" class=\"chart-box\"></div>
            </div>
        </div>
    </div>
</div>

<script>
const chartData = {data_json};
const methods = {json.dumps(METHODS)};
const colors = {json.dumps(PLOT_COLORS)};
const modeStyles = {json.dumps(MODE_STYLES)};

function currentThreshold() {{
    return document.querySelector('input[name="threshold"]:checked').value;
}}

function currentK() {{
    return document.querySelector('input[name="kval"]:checked').value;
}}

function activeModes() {{
    return Array.from(document.querySelectorAll('input[name="mode"]:checked')).map(el => el.value);
}}

function makeTraces(metric, dropType, threshold, kVal) {{
    const block = (((chartData[metric] || {{}})[dropType] || {{}})[String(threshold)] || {{}})[String(kVal)] || {{}};
    const traces = [];

    for (const mode of Object.keys(block)) {{
        for (const method of methods) {{
            const series = (block[mode] || {{}})[method] || {{x: [], y: [], notes: []}};
            if (!series.x || series.x.length === 0) continue;
            traces.push({{
                x: series.x,
                y: series.y,
                type: 'scatter',
                mode: 'lines+markers',
                name: method,
                legendgroup: method,
                showlegend: mode === 'Normal',
                meta: {{ mode: mode, method: method }},
                line: {{ color: colors[method], width: 3, dash: modeStyles[mode].dash }},
                marker: {{ size: 7 }},
                hovertemplate: 'Mode: ' + mode + '<br>Method: ' + method + '<br>Threshold: ' + threshold + '<br>K: ' + kVal + '<br>Forget %: %{{x}}%<br>Drop: %{{y:.2f}} pp<extra></extra>'
            }});
        }}
    }}
    return traces;
}}

function layoutFor(title) {{
    return {{
        title: {{ text: title, font: {{ size: 15 }} }},
        template: 'plotly_white',
        margin: {{ l: 60, r: 20, t: 50, b: 50 }},
        height: 420,
        legend: {{ orientation: 'h', y: -0.2, x: 0, groupclick: 'togglegroup' }},
        xaxis: {{ title: 'Forget percentage' }},
        yaxis: {{ title: 'Drop (pp)' }},
    }};
}}

function renderCharts() {{
    const threshold = currentThreshold();
    const kVal = currentK();
    const modes = new Set(activeModes());

    const configs = [
        ['Hit-retain', 'Hit', 'retain', 'Hit: Retain Drop'],
        ['Hit-forget', 'Hit', 'forget', 'Hit: Forget Drop'],
        ['NDCG-retain', 'NDCG', 'retain', 'NDCG: Retain Drop'],
        ['NDCG-forget', 'NDCG', 'forget', 'NDCG: Forget Drop'],
    ];

    for (const [divId, metric, dropType, title] of configs) {{
        const traces = makeTraces(metric, dropType, threshold, kVal).filter(trace => trace.meta && modes.has(trace.meta.mode));
        Plotly.react(divId, traces, layoutFor(title), {{responsive: true, displaylogo: false}});
    }}
}}

function switchTab(tabName) {{
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
    document.querySelectorAll('.tab-content').forEach(panel => panel.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    renderCharts();
    setTimeout(() => {{
        const idList = tabName === 'Hit' ? ['Hit-retain', 'Hit-forget'] : ['NDCG-retain', 'NDCG-forget'];
        idList.forEach(id => {{
            const el = document.getElementById(id);
            if (el) Plotly.Plots.resize(el);
        }});
    }}, 50);
}}

document.querySelectorAll('input[name="threshold"]').forEach(el => el.addEventListener('change', renderCharts));
document.querySelectorAll('input[name="kval"]').forEach(el => el.addEventListener('change', renderCharts));
document.querySelectorAll('input[name="mode"]').forEach(el => el.addEventListener('change', renderCharts));

window.addEventListener('resize', () => {{
    ['Hit-retain', 'Hit-forget', 'NDCG-retain', 'NDCG-forget'].forEach(id => {{
        const el = document.getElementById(id);
        if (el) Plotly.Plots.resize(el);
    }});
}});

renderCharts();
</script>
</body>
</html>"""


def main():
    num_top_models = parse_args()
    chart_data, alerts, coverage = build_chart_data(num_top_models)
    html = build_html(chart_data, alerts, coverage, num_top_models)

    out_path = BASE_ANALYSIS / f"compare_forget_dashboard_top{num_top_models}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {out_path}")
    if alerts:
        print(f"Warnings: {len(set(alerts))} alert(s) were embedded into the dashboard.")


if __name__ == "__main__":
    main()
