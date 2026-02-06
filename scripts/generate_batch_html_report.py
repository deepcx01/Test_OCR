#!/usr/bin/env python3
"""Batch HTML Report Generator - Creates combined report for batch OCR results."""

import argparse
import json
from pathlib import Path
from datetime import datetime


def generate_batch_html(data: dict, run_number: str = None) -> str:
    """Generate HTML report for batch results."""
    model = data.get("model", "unknown")
    compare_model = data.get("compare_model")
    timestamp = data.get("timestamp", datetime.now().isoformat())
    results = data.get("results", [])
    
    successful = [r for r in results if r.get("success")]
    with_gt = [r for r in successful if r.get("gt_comparison")]
    
    total = len(results)
    avg_sim = 0
    high = medium = low = 0
    
    if with_gt:
        sims = [r["gt_comparison"]["similarity"] for r in with_gt]
        avg_sim = sum(sims) / len(sims)
        high = len([s for s in sims if s >= 90])
        medium = len([s for s in sims if 70 <= s < 90])
        low = len([s for s in sims if s < 70])
    
    rows = ""
    for r in sorted(with_gt, key=lambda x: x["gt_comparison"]["similarity"], reverse=True):
        gt = r["gt_comparison"]
        cls = "high" if gt["similarity"] >= 90 else "medium" if gt["similarity"] >= 70 else "low"
        mc = ""
        if compare_model and r.get("model_comparison"):
            mc = f'<td>{r["model_comparison"]["similarity"]:.1f}%</td>'
        elif compare_model:
            mc = '<td>-</td>'
        rows += f'''<tr><td>{r["image"]}</td><td><span class="badge {cls}">{gt["similarity"]:.1f}%</span></td>
        <td>{gt["total_gt_words"]}</td><td class="ok">{gt["correct_words"]}</td><td class="err">{gt["missing_count"]}</td>{mc}</tr>'''
    
    for r in [x for x in results if not x.get("success")]:
        rows += f'<tr class="fail"><td>{r.get("image","?")}</td><td colspan="4">{r.get("error","Error")[:60]}</td></tr>'
    
    mc_hdr = f'<th>{compare_model.upper()}</th>' if compare_model else ''
    sim_cls = "ok" if avg_sim >= 90 else "warn" if avg_sim >= 70 else "err"
    
    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Batch OCR Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#0d1117;color:#f0f6fc;padding:32px}}
.c{{max-width:1200px;margin:0 auto}}
h1{{font-size:28px;margin-bottom:8px;color:#58a6ff}}
.meta{{color:#8b949e;font-size:14px;margin-bottom:32px}}
.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px}}
.stat{{background:#161b22;border-radius:8px;padding:20px;text-align:center;border:1px solid #30363d}}
.stat .v{{font-size:32px;font-weight:700}}
.stat .l{{font-size:11px;color:#8b949e;text-transform:uppercase;margin-top:4px}}
.ok{{color:#3fb950}}.warn{{color:#d29922}}.err{{color:#f85149}}
table{{width:100%;background:#161b22;border-radius:8px;border-collapse:collapse;border:1px solid #30363d}}
th{{background:#21262d;padding:12px;text-align:left;font-size:11px;text-transform:uppercase;color:#8b949e}}
td{{padding:12px;border-top:1px solid #30363d}}
tr:hover{{background:#21262d}}
.badge{{padding:4px 10px;border-radius:4px;font-size:12px;font-weight:600}}
.badge.high{{background:#3fb95033;color:#3fb950}}
.badge.medium{{background:#d2992233;color:#d29922}}
.badge.low{{background:#f8514933;color:#f85149}}
.fail{{opacity:.5}}
</style></head>
<body><div class="c">
<h1>📊 Batch OCR Report</h1>
<div class="meta">Model: <b>{model.upper()}</b> | Run: #{run_number or "N/A"} | {timestamp[:19]}</div>
<div class="stats">
<div class="stat"><div class="v">{total}</div><div class="l">Files</div></div>
<div class="stat"><div class="v {sim_cls}">{avg_sim:.1f}%</div><div class="l">Avg Score</div></div>
<div class="stat"><div class="v ok">{high}</div><div class="l">High ≥90%</div></div>
<div class="stat"><div class="v warn">{medium}</div><div class="l">Medium</div></div>
<div class="stat"><div class="v err">{low}</div><div class="l">Low &lt;70%</div></div>
</div>
<table><tr><th>File</th><th>Score</th><th>GT Words</th><th>Correct</th><th>Missing</th>{mc_hdr}</tr>{rows}</table>
</div></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", "-r", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--run-number")
    args = parser.parse_args()
    
    data = json.loads(Path(args.results_json).read_text())
    html = generate_batch_html(data, args.run_number)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(html)
    print(f"✅ Report: {args.output}")


if __name__ == "__main__":
    main()
