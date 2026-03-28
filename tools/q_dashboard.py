#!/usr/bin/env python
# coding: utf-8
"""
q_dashboard_v5.py <forget_pct>

Focuses on the "Binary Switch" hypothesis.
Replaces correlation matrix with a Delta Q Distribution Histogram.
"""

import os, sys
import numpy as np
import pandas as pd
from collections import Counter
import json

if len(sys.argv) < 2:
    raise ValueError("Usage: python q_dashboard.py <forget_pct>")

FORGET_PERCENTAGE = int(sys.argv[1])

is_demo = os.environ.get("RUN_MODE", "Demography") == "Demography"

if is_demo:
    ANALYZE_DIR = f"D:/Bob_Skripsi_Do Not Delete/Analysis/Demography/{FORGET_PERCENTAGE}_percent"
else:
    ANALYZE_DIR = f"D:/Bob_Skripsi_Do Not Delete/Analysis/Normal/{FORGET_PERCENTAGE}_percent"

os.makedirs(ANALYZE_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(ANALYZE_DIR, f"q_summary_{sys.argv[2] if len(sys.argv)>2 else 'def'}.csv")
DETAILED_CSV = os.path.join(ANALYZE_DIR, f"q_detailed_movies_{sys.argv[2] if len(sys.argv)>2 else 'def'}.csv")
THRESHOLD = sys.argv[2] if len(sys.argv) > 2 else "def"
DASHBOARD_HTML = os.path.join(ANALYZE_DIR, f"q_dashboard_{THRESHOLD}.html")

df_s = pd.read_csv(SUMMARY_CSV)
df_d = pd.read_csv(DETAILED_CSV)
methods_list = df_s['Method'].unique()

# Prepare Histogram Data: Get every single Anchored_DQ for Damaged users
# We group by Method to create multi-trace histograms
hist_data = {}
for m in methods_list:
    deltas = df_d[(df_d['Method'] == m) & (df_d['User_Group'] == 'D')]['Anchored_DQ'].dropna().tolist()
    hist_data[m] = deltas

hist_json = json.dumps(hist_data)

# --- TABLE GENERATION (Simplified Terminology) ---
def get_style(val, type="dq"):
    if pd.isna(val): return ""
    if type == "dq":
        if val < -15: return "background:#fee2e2; color:#991b1b; font-weight:600;"
        if abs(val) < 3: return "background:#f0fdf4; color:#166534;"
    return ""

summary_rows = ""
for _, r in df_s.iterrows():
    summary_rows += f"""
    <tr>
        <td><b>{r['Method']}</b></td>
        <td>{int(r['|D| Flipped'])}</td>
        <td style="{get_style(r['Avg_Anchored_DQ_H'])}">{r['Avg_Anchored_DQ_H']:.2f}</td>
        <td style="{get_style(r['Avg_Global_DQ_H'])}">{r['Avg_Global_DQ_H']:.2f}</td>
        <td style="{get_style(r['Avg_Anchored_DQ_D'])}">{r['Avg_Anchored_DQ_D']:.2f}</td>
        <td style="{get_style(r['Avg_Global_DQ_D'])}">{r['Avg_Global_DQ_D']:.2f}</td>
    </tr>"""

# Table logic for Movies/Genres stays same as v4
top_movies = df_d[df_d['User_Group']=='D'].groupby(['Method', 'Movie_Title', 'Genres']).agg({'Anchored_DQ':'mean', 'User_ID':'count'}).reset_index()
top_movies = top_movies.sort_values('User_ID', ascending=False).groupby('Method').head(25)
movie_rows = "".join([f'<tr class="filterable-row" data-method="{r["Method"]}"><td><span class="method-badge">{r["Method"]}</span></td><td>{r["Movie_Title"]}</td><td>{r["Genres"]}</td><td>{r["User_ID"]}</td><td style="{get_style(r["Anchored_DQ"])}">{r["Anchored_DQ"]:.2f}</td></tr>' for _, r in top_movies.iterrows()])

method_switches = "".join([f'<button class="method-switch active-method" onclick="toggleMethod(this, \'{m}\')">{m}</button>' for m in methods_list])

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body {{ font-family: 'Inter', sans-serif; background: #f8fafc; padding: 40px 10%; color: #0f172a; }}
        .card {{ background: white; padding: 25px 35px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05); margin-bottom: 30px; border: 1px solid #e2e8f0; }}
        h1 {{ font-weight: 800; letter-spacing: -1px; margin-bottom: 5px; }}
        h2 {{ font-weight: 700; font-size: 18px; margin-top: 0; border-bottom: 2px solid #f1f5f9; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th {{ text-align: left; background: #f8fafc; padding: 12px; font-size: 11px; text-transform: uppercase; color: #64748b; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 12px; border-bottom: 1px solid #f1f5f9; font-size: 14px; color: #334155; }}
        .method-badge {{ background: #e2e8f0; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
        .filter-section {{ background: #f8fafc; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
        .method-switch {{ padding: 6px 14px; border-radius: 20px; border: 1px solid #cbd5e1; background: white; cursor: pointer; font-size: 12px; font-weight: 700; }}
        .method-switch.active-method {{ background: #10b981; color: white; border-color: #10b981; }}
        #dist-plot {{ height: 450px; }}
    </style>
</head>
<body>
    <h1>Behavioral Diagnosis: The "Binary Switch" Hypothesis</h1>
    
    <div class="card">
        <h2>Penalty Distribution (Anchored \(\Delta Q\) for Damaged Users)</h2>
        <div id="dist-plot"></div>
    </div>

    <div class="card">
        <h2>Method Comparison Summary</h2>
        <table>
            <thead><tr><th>Method</th><th>Damaged</th><th>Anc \(\Delta Q\) (Forget)</th><th>Glo \(\Delta Q\) (Forget)</th><th>Anc \(\Delta Q\) (Damaged)</th><th>Glo \(\Delta Q\) (Damaged)</th></tr></thead>
            <tbody>{summary_rows}</tbody>
        </table>
    </div>

    <div class="card">
        <div class="filter-section"><b>Filter:</b> {method_switches}</div>
        <div id="movie-view">
            <table><thead><tr><th>Method</th><th>Movie</th><th>Genres</th><th>Affected</th><th>Avg Anc \(\Delta Q\)</th></tr></thead>
            <tbody>{movie_rows}</tbody></table>
        </div>
    </div>

    <script>
        const histData = {hist_json};
        let activeMethods = new Set({list(methods_list)});

        function renderPlot() {{
            const traces = Object.keys(histData).filter(m => activeMethods.has(m)).map(m => ({{
                x: histData[m],
                type: 'histogram',
                name: m,
                opacity: 0.6,
                autobinx: false,
                xbins: {{ start: -60, end: 10, size: 2 }}
            }}));

            const layout = {{
                title: 'Frequency of Penalty Magnitudes',
                xaxis: {{ title: 'Anchored ΔQ (Punishment Intensity)', range: [-60, 5] }},
                yaxis: {{ title: 'Count of Occurrences' }},
                barmode: 'overlay',
                plot_bgcolor: 'white'
            }};
            Plotly.newPlot('dist-plot', traces, layout);
        }}

        function toggleMethod(btn, method) {{
            if (activeMethods.has(method)) {{ activeMethods.delete(method); btn.classList.remove('active-method'); }}
            else {{ activeMethods.add(method); btn.classList.add('active-method'); }}
            renderPlot();
            document.querySelectorAll('.filterable-row').forEach(row => {{
                row.style.display = activeMethods.has(row.dataset.method) ? '' : 'none';
            }});
        }}
        renderPlot();
    </script>
</body>
</html>
"""

with open(DASHBOARD_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)