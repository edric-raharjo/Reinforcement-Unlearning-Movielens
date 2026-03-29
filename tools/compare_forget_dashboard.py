#!/usr/bin/env python
# coding: utf-8

import json
import sys
def build_html(chart_data, alerts, coverage, num_top_models):
    data_json = json.dumps(chart_data)
    alerts_html = build_alert_list(alerts)
    alerts_panel = (
        f'<div class="panel alerts"><h3 style="margin:0 0 8px; font-size:14px; text-transform:uppercase; color:var(--muted); letter-spacing:0.6px;">Data Alerts</h3><ul>{alerts_html}</ul></div>'
        if alerts_html
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Forget Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
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
        @media (min-width: 1024px) {{
            .chart-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>Forget Comparison Dashboard</h1>
    <p class="subtitle">Top {num_top_models} models from train phase results. Solid line = Normal, dashed line = Demography. Click legend items to toggle methods.</p>

    <div class="panel controls">
        <div class="control-group">
            <h3>Threshold</h3>
            <div class="radio-row">
                {''.join(f'<label class="radio-pill"><input type="radio" name="threshold" value="{t}" {"checked" if t == THRESHOLDS[0] else ""}> {t}</label>' for t in THRESHOLDS)}
            </div>
        </div>
        <div class="control-group">
            <h3>K</h3>
            <div class="radio-row">
                {''.join(f'<label class="radio-pill"><input type="radio" name="kval" value="{k}" {"checked" if k == KS[0] else ""}> {k}</label>' for k in KS)}
            </div>
        </div>
        <div class="control-group">
            <h3>Info</h3>
            <div class="top-note">The radio boxes switch among precomputed selections; the HTML is generated once and remains interactive offline.</div>
        </div>
        <div class="control-group">
            <h3>Mode</h3>
            <div class="mode-checklist">
                <label class="radio-pill"><input type="checkbox" name="mode" value="Normal" checked> Normal</label>
                <label class="radio-pill"><input type="checkbox" name="mode" value="Demography" checked> Demography</label>
            </div>
        </div>
    </div>

    {alerts_panel}

    <div class="tab-bar">
        <button class="tab-btn active" data-tab="Hit" onclick="switchTab('Hit')">Hit</button>
        <button class="tab-btn" data-tab="NDCG" onclick="switchTab('NDCG')">NDCG</button>
    </div>

    <div id="tab-Hit" class="tab-content active">
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-title">Hit: Retain Drop</div>
                <div class="chart-note">Methods are color-coded. Normal is solid; Demography is dashed.</div>
                <div id="Hit-retain" class="chart-box"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Hit: Forget Drop</div>
                <div class="chart-note">Higher forget drop is better; the line should rise when the method improves.</div>
                <div id="Hit-forget" class="chart-box"></div>
            </div>
        </div>
    </div>

    <div id="tab-NDCG" class="tab-content">
        <div class="chart-grid">
            <div class="chart-card">
                <div class="chart-title">NDCG: Retain Drop</div>
                <div class="chart-note">Retain drop should stay small or increase only minimally.</div>
                <div id="NDCG-retain" class="chart-box"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">NDCG: Forget Drop</div>
                <div class="chart-note">This should show whether forgetting improves as the forget percentage increases.</div>
                <div id="NDCG-forget" class="chart-box"></div>
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
        return ""

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
        @media (min-width: 1024px) {{
            .chart-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}
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
