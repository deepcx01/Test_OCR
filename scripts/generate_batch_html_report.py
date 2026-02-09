import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import html as html_escape

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ocr_runner.similarity_logic import tokenize


def get_word_highlighted_html(text: str, other_text: str):
    """Generate HTML with highlighted words unique to this text."""
    if not text:
        return ""
    
    words1 = tokenize(text)
    words2 = tokenize(other_text)
    
    unique_to_1 = set(words1) - set(words2)
    
    lines = text.splitlines()
    highlighted_lines = []
    
    for line in lines:
        if not line.strip():
            highlighted_lines.append("")
            continue
            
        words = line.split()
        result = []
        for word in words:
            # Simple normalization for matching
            norm = "".join(c.lower() for c in word if c.isalnum())
            
            is_unique = False
            if norm:
                # Check if any tokenized word from this word is in unique_to_1
                t_words = tokenize(word)
                if t_words and all(tw in unique_to_1 for tw in t_words):
                    is_unique = True
            
            if is_unique:
                result.append(f'<span class="w-unique">{html_escape.escape(word)}</span>')
            else:
                result.append(f'<span class="w-common">{html_escape.escape(word)}</span>')
        
        highlighted_lines.append(" ".join(result))
    
    return "<br>".join(highlighted_lines)


def generate_batch_html(data: dict, run_number: str = None) -> str:
    """Generate a self-contained SPA HTML report."""
    model = data.get("model", "unknown").upper()
    compare_model = data.get("compare_model")
    timestamp = data.get("timestamp", datetime.now().isoformat())
    results = data.get("results", [])
    
    # Pre-process results for the UI
    processed_results = []
    for r in results:
        entry = r.copy()
        if r.get("success") and r.get("gt_text") and r.get("ocr_text"):
            entry["ocr_html"] = get_word_highlighted_html(r["ocr_text"], r["gt_text"])
            entry["gt_html"] = get_word_highlighted_html(r["gt_text"], r["ocr_text"])
            
            # Word differences for the chips
            words_ocr = Counter(tokenize(r["ocr_text"]))
            words_gt = Counter(tokenize(r["gt_text"]))
            
            missing = []
            for w, count in words_gt.items():
                diff = count - words_ocr.get(w, 0)
                if diff > 0:
                    missing.append({"w": w, "c": diff})
            
            extra = []
            for w, count in words_ocr.items():
                diff = count - words_gt.get(w, 0)
                if diff > 0:
                    extra.append({"w": w, "c": diff})
            
            entry["missing_words"] = sorted(missing, key=lambda x: x["c"], reverse=True)[:50]
            entry["extra_words"] = sorted(extra, key=lambda x: x["c"], reverse=True)[:50]
            
        processed_results.append(entry)

    # Statistics
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

    # Convert results to JSON for embedding
    results_json = json.dumps(processed_results)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Benchmark Report - {model}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {{
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #f0f6fc;
            --text-dim: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warn: #d29922;
            --err: #f85149;
            --purple: #a371f7;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding: 40px 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{ margin-bottom: 40px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }}
        h1 {{ font-size: 32px; margin-bottom: 8px; background: linear-gradient(135deg, var(--accent), var(--purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .meta {{ color: var(--text-dim); font-size: 14px; }}
        
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .stat {{ background: var(--card); padding: 24px; border-radius: 12px; border: 1px solid var(--border); text-align: center; }}
        .stat .v {{ font-size: 32px; font-weight: 700; }}
        .stat .l {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase; margin-top: 4px; letter-spacing: 0.5px; }}
        
        table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
        th {{ background: #21262d; padding: 16px; text-align: left; font-size: 12px; text-transform: uppercase; color: var(--text-dim); }}
        td {{ padding: 16px; border-top: 1px solid var(--border); }}
        tr:hover {{ background: #21262d; cursor: pointer; }}
        
        .badge {{ padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }}
        .badge.high {{ background: #3fb95022; color: var(--success); }}
        .badge.medium {{ background: #d2992222; color: var(--warn); }}
        .badge.low {{ background: #f8514922; color: var(--err); }}
        
        /* Modal / Detail View */
        #detail-view {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.85); backdrop-filter: blur(4px); z-index: 1000; overflow-y: auto; padding: 40px 20px; }}
        .modal {{ background: var(--bg); max-width: 1100px; margin: 0 auto; border-radius: 16px; border: 1px solid var(--border); padding: 32px; position: relative; }}
        .close {{ position: absolute; top: 20px; right: 20px; font-size: 24px; color: var(--text-dim); cursor: pointer; }}
        
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }}
        .panel {{ background: var(--card); border-radius: 12px; border: 1px solid var(--border); padding: 20px; }}
        .panel h3 {{ font-size: 14px; margin-bottom: 12px; color: var(--text-dim); text-transform: uppercase; }}
        pre {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 2; white-space: pre-wrap; word-break: break-all; max-height: 500px; overflow-y: auto; padding: 12px; background: #0004; border-radius: 8px; }}
        
        .w-common {{ color: var(--success); background: #3fb95015; padding: 2px 4px; border-radius: 4px; }}
        .w-unique {{ color: var(--warn); background: #d2992225; padding: 2px 4px; border-radius: 4px; font-weight: 600; }}
        
        .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
        .chip {{ font-size: 11px; padding: 4px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }}
        .chip.miss {{ background: #f8514922; color: var(--err); border: 1px solid #f8514933; }}
        .chip.ext {{ background: #d2992222; color: var(--warn); border: 1px solid #d2992233; }}
        .chip small {{ opacity: 0.6; margin-left: 4px; }}
    </style>
</head>
<body>
    <div class="container" id="main-view">
        <header>
            <h1>📊 OCR Benchmark Report</h1>
            <div class="meta">Model: <b>{model}</b> | Run: #{run_number or "N/A"} | Generated: {timestamp[:19]}</div>
        </header>
        
        <div class="stats">
            <div class="stat"><div class="v">{total}</div><div class="l">Total Files</div></div>
            <div class="stat"><div class="v" style="color:var(--{('success' if avg_sim >= 90 else 'warn' if avg_sim >= 70 else 'err')})">{avg_sim:.1f}%</div><div class="l">Avg Score</div></div>
            <div class="stat"><div class="v" style="color:var(--success)">{high}</div><div class="l">High ≥90%</div></div>
            <div class="stat"><div class="v" style="color:var(--warn)">{medium}</div><div class="l">Medium</div></div>
            <div class="stat"><div class="v" style="color:var(--err)">{low}</div><div class="l">Low &lt;70%</div></div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Score</th>
                    <th>GT Words</th>
                    <th>Correct</th>
                    <th>Missing</th>
                    {f'<th>{compare_model.upper()}</th>' if compare_model else ''}
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <div id="detail-view">
        <div class="modal">
            <span class="close" onclick="closeDetail()">&times;</span>
            <h2 id="detail-title">File Details</h2>
            <div id="detail-meta" class="meta" style="margin-top:8px"></div>
            
            <div class="grid">
                <div class="panel" style="border-left: 4px solid var(--accent)">
                    <h3>OCR Output ({model})</h3>
                    <pre id="ocr-content"></pre>
                </div>
                <div class="panel" style="border-left: 4px solid var(--success)">
                    <h3>Ground Truth</h3>
                    <pre id="gt-content"></pre>
                </div>
            </div>

            <div class="grid" style="grid-template-columns: 1fr 1fr; margin-top: 24px;">
                <div class="panel">
                    <h3>❌ Missing Words</h3>
                    <div id="missing-chips" class="chips"></div>
                </div>
                <div class="panel">
                    <h3>➕ Extra Words</h3>
                    <div id="extra-chips" class="chips"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const results = {results_json};

        function renderTable() {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = results.map((r, i) => {{
                if (!r.success) {{
                    return `<tr class="fail"><td>${{r.image}}</td><td colspan="5" style="color:var(--err)">Error: ${{r.error || "Failed"}}</td></tr>`;
                }}
                const gt = r.gt_comparison || {{}};
                const cls = (gt.similarity >= 90) ? 'high' : (gt.similarity >= 70) ? 'medium' : 'low';
                const mc = r.model_comparison ? `<td>${{r.model_comparison.similarity.toFixed(1)}}%</td>` : '{'<td>-</td>' if compare_model else ''}';
                
                return `<tr onclick="showDetail(${{i}})">
                    <td><b>${{r.image}}</b></td>
                    <td><span class="badge ${{cls}}">${{gt.similarity ? gt.similarity.toFixed(1) + '%' : '-'}}</span></td>
                    <td>${{gt.total_gt_words || 0}}</td>
                    <td style="color:var(--success)">${{gt.correct_words || 0}}</td>
                    <td style="color:var(--err)">${{gt.missing_count || 0}}</td>
                    ${{mc}}
                </tr>`;
            }}).join('');
        }}

        function showDetail(index) {{
            const r = results[index];
            if (!r || !r.success) return;

            document.getElementById('detail-title').innerText = r.image;
            document.getElementById('detail-meta').innerHTML = `Score: <b>${{r.gt_comparison.similarity.toFixed(1)}}%</b> | Words: ${{r.word_count}} | Basename: ${{r.basename}}`;
            
            document.getElementById('ocr-content').innerHTML = r.ocr_html || 'N/A';
            document.getElementById('gt-content').innerHTML = r.gt_html || 'N/A';
            
            const renderChips = (list, cls) => list.length ? 
                list.map(w => `<span class="chip ${{cls}}">${{w.w}}${{w.c > 1 ? '<small>×'+w.c+'</small>' : ''}}</span>`).join('') :
                '<span style="color:var(--text-dim); font-size:12px">None</span>';
            
            document.getElementById('missing-chips').innerHTML = renderChips(r.missing_words || [], 'miss');
            document.getElementById('extra-chips').innerHTML = renderChips(r.extra_words || [], 'ext');
            
            document.getElementById('detail-view').style.display = 'block';
            document.body.style.overflow = 'hidden';
        }}

        function closeDetail() {{
            document.getElementById('detail-view').style.display = 'none';
            document.body.style.overflow = 'auto';
        }}

        // Close on Esc
        document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeDetail(); }});

        renderTable();
    </script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", "-r", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--run-number")
    args = parser.parse_args()
    
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"❌ Error: {args.results_json} not found")
        sys.exit(1)
        
    data = json.loads(results_path.read_text())
    html = generate_batch_html(data, args.run_number)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"✅ Enhanced Report: {args.output}")


if __name__ == "__main__":
    main()
