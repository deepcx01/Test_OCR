#!/usr/bin/env python3
"""
OCR Comparison Dashboard - Live Server
Compare any two sources: Ground Truth, Doctr, Surya, or PaddleOCR
Generates HTML dynamically without saving files
"""

from pathlib import Path
import os
import sys
from datetime import datetime
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote, parse_qs
import webbrowser
import html as html_escape

# Setup paths - BASE_DIR is the parent of ocr_benchmark where data directories are
SCRIPT_DIR = Path(__file__).parent
OCR_BENCHMARK_DIR = SCRIPT_DIR.parent
BASE_DIR = OCR_BENCHMARK_DIR.parent  # Abstract files directory

GT_DIR = BASE_DIR / "Ground Truth "  # Note: has trailing space in folder name
SURYA_DIR = BASE_DIR / "Surya"
DOCTR_DIR = BASE_DIR / "Doctr"
PADDLE_DIR = BASE_DIR / "PaddleOCR"

PORT = 8084

# Source Configuration - All available text sources
SOURCE_CONFIG = {
    "gt": {
        "dir": GT_DIR,
        "label": "Ground Truth",
        "short": "GT",
        "color": "#00d26a",
        "color_rgb": "0, 210, 106",
        "file_prefix": "ct_",
    },
    "doctr": {
        "dir": DOCTR_DIR / "custom_text",
        "label": "Doctr",
        "short": "DOC",
        "color": "#64ffda",
        "color_rgb": "100, 255, 218",
        "file_prefix": "ct_",
    },
    "surya": {
        "dir": SURYA_DIR / "custom_text",
        "label": "Surya",
        "short": "SUR",
        "color": "#bb86fc",
        "color_rgb": "187, 134, 252",
        "file_prefix": "ct_",
    },
    "paddle": {
        "dir": PADDLE_DIR / "custom_text",
        "label": "PaddleOCR",
        "short": "PDL",
        "color": "#ff7b72",
        "color_rgb": "255, 123, 114",
        "file_prefix": "Input_",
    },
}

# Add ocr_benchmark directory to path for imports
sys.path.insert(0, str(OCR_BENCHMARK_DIR))

# Import from ocr_runner module
from ocr_runner.similarity_logic import compute_similarity, tokenize

print("Loaded similarity_logic from ocr_runner")


def load_text(path: Path) -> str:
    """Load text from file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def get_word_differences(source_text: str, target_text: str):
    """Get missing words and wrong/extra words."""
    source_words = tokenize(source_text)
    target_words = tokenize(target_text)
    
    source_counter = Counter(source_words)
    target_counter = Counter(target_words)
    
    missing = []
    for word, source_count in source_counter.items():
        target_count = target_counter.get(word, 0)
        diff = source_count - target_count
        if diff > 0:
            missing.extend([word] * diff)
    
    wrong = []
    for word, target_count in target_counter.items():
        source_count = source_counter.get(word, 0)
        diff = target_count - source_count
        if diff > 0:
            wrong.extend([word] * diff)
    
    return missing, wrong


def extract_number(filename: str) -> int:
    """Extract number from filename like ct_1, ct_10, Input_1, etc."""
    import re
    match = re.search(r'_(\d+)$', filename)
    return int(match.group(1)) if match else 0


def get_file_path(source_key: str, doc_num: int) -> Path:
    """Get the file path for a specific source and document number."""
    config = SOURCE_CONFIG[source_key]
    filename = f"{config['file_prefix']}{doc_num}.txt"
    return config["dir"] / filename


def get_available_documents():
    """Get list of document numbers available across all sources."""
    # Use Ground Truth as the reference for available documents
    gt_files = sorted(GT_DIR.glob("*.txt"), key=lambda x: extract_number(x.stem))
    return [extract_number(f.stem) for f in gt_files]


def get_all_comparisons(source1: str, source2: str):
    """Get all comparison data between two sources."""
    results = []
    
    if source1 not in SOURCE_CONFIG or source2 not in SOURCE_CONFIG:
        return results
    
    config1 = SOURCE_CONFIG[source1]
    config2 = SOURCE_CONFIG[source2]
    
    if not config1["dir"].exists() or not config2["dir"].exists():
        print(f"Warning: Directories not found!")
        return results
    
    doc_nums = get_available_documents()
    
    for num in doc_nums:
        path1 = get_file_path(source1, num)
        path2 = get_file_path(source2, num)
        
        if not path1.exists() or not path2.exists():
            continue
        
        text1 = load_text(path1)
        text2 = load_text(path2)
        
        sim_result = compute_similarity(text1, text2)
        missing_words, extra_words = get_word_differences(text1, text2)
        
        results.append({
            "num": num,
            "display_name": f"Document {num}",
            "similarity": sim_result.similarity_score / 100.0,
            "source1_words": sim_result.total_gt_words,
            "common_words": sim_result.correct_words,
            "only_in_source1": len(missing_words),
            "only_in_source2": len(extra_words),
        })
    
    return results


def generate_home_html() -> str:
    """Generate home page with source selection interface."""
    
    # Generate source options HTML
    source_options = ""
    for key, config in SOURCE_CONFIG.items():
        source_options += f'<option value="{key}">{config["label"]}</option>\n'
    
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>OCR Comparison Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ 
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
    padding: 40px 20px; 
    color: #e8e8e8;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.container {{ 
    max-width: 800px;
    width: 100%;
    background: rgba(26, 26, 46, 0.95); 
    padding: 48px; 
    border-radius: 24px; 
    box-shadow: 0 16px 64px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05); 
    backdrop-filter: blur(10px);
    text-align: center;
}}
h1 {{ 
    font-size: 42px;
    font-weight: 700; 
    margin-bottom: 16px;
    background: linear-gradient(135deg, #64ffda, #bb86fc, #ff7b72);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.subtitle {{
    color: #8892b0;
    font-size: 18px;
    margin-bottom: 48px;
}}
.comparison-builder {{
    background: rgba(0,0,0,0.3);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 32px;
    border: 1px solid rgba(255,255,255,0.1);
}}
.comparison-builder h2 {{
    color: #ccd6f6;
    font-size: 20px;
    margin-bottom: 24px;
    font-weight: 600;
}}
.selector-row {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 28px;
}}
.selector-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.selector-label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8892b0;
    font-weight: 600;
}}
select {{
    padding: 14px 20px;
    font-size: 16px;
    border-radius: 12px; 
    border: 2px solid rgba(100,255,218,0.3);
    background: rgba(0,0,0,0.4);
    color: #fff;
    font-family: 'Outfit', sans-serif;
    cursor: pointer;
    min-width: 200px;
    transition: all 0.3s ease;
    appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2364ffda' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 12px center;
    background-size: 18px;
    padding-right: 44px;
}}
select:hover {{
    border-color: rgba(100,255,218,0.6);
}}
select:focus {{
    outline: none;
    border-color: #64ffda;
    box-shadow: 0 0 0 3px rgba(100,255,218,0.2);
}}
select option {{
    background: #1a1a2e;
    color: #fff;
    padding: 12px;
}}
.vs-badge {{
    font-size: 24px;
    font-weight: 700; 
    color: #bb86fc;
    padding: 10px 20px;
    background: rgba(187,134,252,0.1);
    border-radius: 50px;
    border: 2px solid rgba(187,134,252,0.3);
}}
.compare-btn {{
    background: linear-gradient(135deg, #64ffda, #00d26a);
    color: #0f0f23;
    border: none;
    padding: 16px 48px;
    font-size: 18px;
    font-weight: 700;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: 'Outfit', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.compare-btn:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(100,255,218,0.3);
}}
.compare-btn:disabled {{
    background: linear-gradient(135deg, #444, #333);
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}}
.error-msg {{
    color: #ff7b72;
    font-size: 14px;
    margin-top: 12px;
    display: none;
}}
.quick-links {{
    margin-top: 40px;
}}
.quick-links h3 {{
    color: #8892b0;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
}}
.quick-links-grid {{
    display: grid; 
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}}
.quick-link {{
    padding: 16px;
    border-radius: 12px; 
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    text-decoration: none;
    color: #ccd6f6;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.3s ease;
}}
.quick-link:hover {{
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-2px);
}}
.quick-link .icon {{
    font-size: 20px;
    display: block;
    margin-bottom: 8px;
}}

@media (max-width: 600px) {{
    .selector-row {{ flex-direction: column; }}
    .quick-links-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
    <h1>OCR Comparison Dashboard</h1>
    <p class="subtitle">Select any two sources to compare word-by-word accuracy</p>
    
    <div class="comparison-builder">
        <h2>Build Your Comparison</h2>
        <div class="selector-row">
            <div class="selector-group">
                <span class="selector-label">Source A (Reference)</span>
                <select id="source1">
                    {source_options}
                </select>
            </div>

            <div class="vs-badge">VS</div>
            
            <div class="selector-group">
                <span class="selector-label">Source B (Compare To)</span>
                <select id="source2">
                    {source_options.replace('value="gt"', 'value="doctr" selected').replace('value="doctr"', 'value="gt"', 1)}
                </select>
            </div>
        </div>

        <button class="compare-btn" onclick="startComparison()">
            Compare Now
        </button>
        <div class="error-msg" id="error-msg">Please select two different sources to compare</div>
    </div>
    
    <div class="quick-links">
        <h3>Quick Comparisons</h3>
        <div class="quick-links-grid">
            <a href="/compare/gt/doctr" class="quick-link">
                <span class="icon">GT / DOC</span>
                Ground Truth vs Doctr
            </a>
            <a href="/compare/gt/surya" class="quick-link">
                <span class="icon">GT / SUR</span>
                Ground Truth vs Surya
            </a>
            <a href="/compare/gt/paddle" class="quick-link">
                <span class="icon">GT / PDL</span>
                Ground Truth vs PaddleOCR
            </a>
            <a href="/compare/doctr/surya" class="quick-link">
                <span class="icon">DOC / SUR</span>
                Doctr vs Surya
            </a>
            <a href="/compare/doctr/paddle" class="quick-link">
                <span class="icon">DOC / PDL</span>
                Doctr vs PaddleOCR
            </a>
            <a href="/compare/surya/paddle" class="quick-link">
                <span class="icon">SUR / PDL</span>
                Surya vs PaddleOCR
            </a>
        </div>
    </div>
</div>

<script>
function startComparison() {{
    const source1 = document.getElementById('source1').value;
    const source2 = document.getElementById('source2').value;
    const errorMsg = document.getElementById('error-msg');
    
    if (source1 === source2) {{
        errorMsg.style.display = 'block';
        return;
    }}
    
    errorMsg.style.display = 'none';
    window.location.href = `/compare/${{source1}}/${{source2}}`;
}}

// Set default: GT vs Doctr
document.getElementById('source1').value = 'gt';
document.getElementById('source2').value = 'doctr';
</script>
</body>
</html>
"""


def generate_summary_html(source1: str, source2: str) -> str:
    """Generate summary HTML page comparing two sources."""
    
    if source1 not in SOURCE_CONFIG or source2 not in SOURCE_CONFIG:
        return "<h1>Invalid source</h1>"
    
    results = get_all_comparisons(source1, source2)
    
    config1 = SOURCE_CONFIG[source1]
    config2 = SOURCE_CONFIG[source2]
    
    # Create a blended color for the theme
    theme_color = config1["color"]
    theme_color_rgb = config1["color_rgb"]
    
    rows = ""
    for r in sorted(results, key=lambda x: x["num"]):
        similarity_class = "similarity-high" if r["similarity"] >= 0.9 else "similarity-medium" if r["similarity"] >= 0.7 else "similarity-low"
        rows += f"""<tr>
            <td><strong>{r['display_name']}</strong></td>
            <td><span class="badge {similarity_class}">{r['similarity']:.1%}</span></td>
            <td>{r['source1_words']}</td>
            <td class="correct">{r['common_words']}</td>
            <td class="missing">{r['only_in_source1']}</td>
            <td class="wrong">{r['only_in_source2']}</td>
            <td><a href="/compare/{source1}/{source2}/{r['num']}" class="view-link">View Details →</a></td>
        </tr>"""

    total_docs = len(results)
    avg_similarity = sum(r["similarity"] for r in results) / total_docs if total_docs > 0 else 0
    high_similarity = len([r for r in results if r["similarity"] >= 0.9])
    medium_similarity = len([r for r in results if 0.7 <= r["similarity"] < 0.9])
    low_similarity = len([r for r in results if r["similarity"] < 0.7])
    total_source1_words = sum(r["source1_words"] for r in results)
    total_common = sum(r["common_words"] for r in results)
    total_only_source1 = sum(r["only_in_source1"] for r in results)
    total_only_source2 = sum(r["only_in_source2"] for r in results)

    # Generate navigation tabs for all comparison pairs
    nav_tabs = ""
    pairs = [
        ("gt", "doctr"), ("gt", "surya"), ("gt", "paddle"),
        ("doctr", "surya"), ("doctr", "paddle"), ("surya", "paddle")
    ]
    for s1, s2 in pairs:
        c1 = SOURCE_CONFIG[s1]
        c2 = SOURCE_CONFIG[s2]
        is_active = (s1 == source1 and s2 == source2) or (s1 == source2 and s2 == source1)
        active_class = "active" if is_active else ""
        nav_tabs += f'<a href="/compare/{s1}/{s2}" class="nav-tab {active_class}">{c1["short"]} / {c2["short"]}</a>\n'

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{config1['label']} vs {config2['label']} — Comparison</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ 
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
    padding: 20px; 
    color: #e8e8e8;
}}
.container {{ 
    width: 100%;
    max-width: 100%;
    background: rgba(26, 26, 46, 0.95); 
    padding: 28px; 
    border-radius: 16px; 
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05); 
    backdrop-filter: blur(10px);
}}
.header-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}}
.home-link {{
    color: #64ffda;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 18px;
    background: rgba(100,255,218,0.1);
    border: 1px solid rgba(100,255,218,0.3);
    border-radius: 10px;
    transition: all 0.3s ease;
}}
.home-link:hover {{
    background: rgba(100,255,218,0.2);
}}
.nav-tabs {{
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}}
.nav-tab {{
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    text-decoration: none;
    transition: all 0.3s ease;
    background: rgba(255,255,255,0.05);
    color: #8892b0;
    border: 1px solid rgba(255,255,255,0.1);
}}
.nav-tab:hover {{
    background: rgba(255,255,255,0.1);
    color: #ccd6f6;
}}
.nav-tab.active {{
    background: linear-gradient(135deg, rgba({theme_color_rgb},0.2), rgba({theme_color_rgb},0.1));
    color: {theme_color};
    border-color: rgba({theme_color_rgb},0.4);
}}
h1 {{ 
    margin-bottom: 28px; 
    color: #fff; 
    font-size: 28px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.source-badges {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}}
.source-badge {{
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 14px;
}}
.source-badge.source1 {{
    background: rgba({config1['color_rgb']},0.2);
    color: {config1['color']};
    border: 1px solid rgba({config1['color_rgb']},0.4);
}}
.source-badge.source2 {{
    background: rgba({config2['color_rgb']},0.2);
    color: {config2['color']};
    border: 1px solid rgba({config2['color_rgb']},0.4);
}}
.vs-text {{
    color: #8892b0;
    font-weight: 600;
}}
.stats {{ 
    display: grid; 
    grid-template-columns: repeat(5, 1fr); 
    gap: 16px; 
    margin-bottom: 28px; 
}}
.stat-card {{ 
    padding: 22px 18px; 
    border-radius: 14px; 
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
    pointer-events: none;
}}
.stat-card.purple {{ background: linear-gradient(135deg, #bb86fc 0%, #8b5cf6 100%); }}
.stat-card.cyan {{ background: linear-gradient(135deg, #64ffda 0%, #00d26a 100%); color: #1a1a2e; }}
.stat-card.green {{ background: linear-gradient(135deg, #00d26a 0%, #00a854 100%); }}
.stat-card.yellow {{ background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); color: #1a1a2e; }}
.stat-card.red {{ background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%); }}
.stat-value {{ font-size: 36px; font-weight: 700; position: relative; }}
.stat-label {{ 
    text-transform: uppercase; 
    font-size: 11px; 
    opacity: 0.9; 
    margin-top: 10px; 
    font-weight: 600;
    letter-spacing: 1px;
    position: relative;
}}
.word-summary {{
    background: rgba(0,0,0,0.3);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
}}
.word-summary h3 {{ 
    margin-bottom: 14px; 
    color: #ccd6f6; 
    font-size: 14px; 
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.word-stats {{ display: flex; gap: 30px; flex-wrap: wrap; font-size: 14px; }}
.word-stats strong {{ color: #8892b0; font-weight: 500; }}
.word-stats .value {{ font-weight: 700; margin-left: 6px; }}
.word-stats .value.green {{ color: #00d26a; }}
.word-stats .value.red {{ color: #ff4757; }}
.word-stats .value.orange {{ color: #ffc107; }}
.word-stats .value.purple {{ color: #bb86fc; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ 
    background: linear-gradient(135deg, rgba({theme_color_rgb},0.15), rgba({theme_color_rgb},0.08));
    color: {theme_color}; 
    padding: 16px 14px; 
    text-align: left;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba({theme_color_rgb},0.2);
}}
th:first-child {{ border-radius: 10px 0 0 0; }}
th:last-child {{ border-radius: 0 10px 0 0; }}
td {{ padding: 16px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
tr {{ transition: background 0.2s ease; }}
tr:hover {{ background: rgba({theme_color_rgb},0.05); }}
td.correct {{ color: #00d26a; font-weight: 700; }}
td.missing {{ color: #ff4757; font-weight: 700; }}
td.wrong {{ color: #ffc107; font-weight: 700; }}
.badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 12px;
}}
.badge.similarity-high {{ background: rgba(0,210,106,0.2); color: #00d26a; border: 1px solid rgba(0,210,106,0.3); }}
.badge.similarity-medium {{ background: rgba(255,193,7,0.2); color: #ffc107; border: 1px solid rgba(255,193,7,0.3); }}
.badge.similarity-low {{ background: rgba(255,71,87,0.2); color: #ff4757; border: 1px solid rgba(255,71,87,0.3); }}
.view-link {{ 
    color: {theme_color}; 
    text-decoration: none; 
    font-weight: 600; 
    background: rgba({theme_color_rgb},0.1); 
    padding: 8px 14px; 
    border-radius: 8px;
    font-size: 12px;
    border: 1px solid rgba({theme_color_rgb},0.2);
    transition: all 0.3s ease;
}}
.view-link:hover {{ 
    background: rgba({theme_color_rgb},0.2); 
    transform: translateX(3px);
}}
.footer {{ 
    margin-top: 28px; 
    text-align: center; 
    color: #8892b0; 
    font-size: 12px;
    padding-top: 22px;
    border-top: 1px solid rgba(255,255,255,0.08);
}}

@media (max-width: 900px) {{
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>
<div class="container">
    <div class="header-row">
        <a href="/" class="home-link">← Back to Comparison Lab</a>
    </div>
    
    <div class="nav-tabs">
        {nav_tabs}
    </div>

    <div class="source-badges">
        <span class="source-badge source1">{config1['label']}</span>
        <span class="vs-text">compared to</span>
        <span class="source-badge source2">{config2['label']}</span>
    </div>
    
    <h1>Comparison Results</h1>
    
    <div class="stats">
        <div class="stat-card purple"><div class="stat-value">{total_docs}</div><div class="stat-label">Documents</div></div>
        <div class="stat-card cyan"><div class="stat-value">{avg_similarity:.1%}</div><div class="stat-label">Avg Match</div></div>
        <div class="stat-card green"><div class="stat-value">{high_similarity}</div><div class="stat-label">High (≥90%)</div></div>
        <div class="stat-card yellow"><div class="stat-value">{medium_similarity}</div><div class="stat-label">Medium (70-89%)</div></div>
        <div class="stat-card red"><div class="stat-value">{low_similarity}</div><div class="stat-label">Low (&lt;70%)</div></div>
    </div>

    <div class="word-summary">
        <h3>Word-Level Summary</h3>
        <div class="word-stats">
            <div><strong>Total {config1['label']} Words:</strong><span class="value purple">{total_source1_words:,}</span></div>
            <div><strong>Common Words:</strong><span class="value green">{total_common:,}</span></div>
            <div><strong>Only in {config1['label']}:</strong><span class="value red">{total_only_source1:,}</span></div>
            <div><strong>Only in {config2['label']}:</strong><span class="value orange">{total_only_source2:,}</span></div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Document</th>
                <th>Match</th>
                <th>{config1['label']} Words</th>
                <th>Common</th>
                <th>Only {config1['label']}</th>
                <th>Only {config2['label']}</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    
    <div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} • Live Server on port {PORT}</div>
</div>
</body>
</html>
"""


def generate_detail_html(source1: str, source2: str, doc_num: int) -> str:
    """Generate detailed comparison HTML for a specific document."""
    
    if source1 not in SOURCE_CONFIG or source2 not in SOURCE_CONFIG:
        return "<h1>Invalid source</h1>"
    
    config1 = SOURCE_CONFIG[source1]
    config2 = SOURCE_CONFIG[source2]
    
    path1 = get_file_path(source1, doc_num)
    path2 = get_file_path(source2, doc_num)
    
    if not path1.exists() or not path2.exists():
        return "<h1>File not found</h1>"
    
    text1 = load_text(path1)
    text2 = load_text(path2)
    
    sim_result = compute_similarity(text1, text2)
    similarity = sim_result.similarity_score / 100.0
    only_source1, only_source2 = get_word_differences(text1, text2)
    
    # Word sets for highlighting
    word_set1 = set(tokenize(text1))
    word_set2 = set(tokenize(text2))
    only_in_1_set = word_set1 - word_set2
    only_in_2_set = word_set2 - word_set1
    
    def highlight_text(full_text, missing_set):
        lines = full_text.splitlines()
        highlighted_lines = []
        for line in lines:
            if not line.strip():
                continue
            words = line.split()
            result = []
            for word in words:
                normalized = tokenize(word)
                if not normalized:
                    result.append(html_escape.escape(word))
                    continue
                norm_word = normalized[0]
                if norm_word in missing_set:
                    result.append(f"<span class='word-unique'>{html_escape.escape(word)}</span>")
                else:
                    result.append(f"<span class='word-common'>{html_escape.escape(word)}</span>")
            highlighted_lines.append(' '.join(result))
        return '<br>'.join(highlighted_lines)
    
    text1_highlighted = highlight_text(text1, only_in_1_set)
    text2_highlighted = highlight_text(text2, only_in_2_set)
    
    # Words HTML
    if only_source1:
        source1_words_counter = Counter(only_source1)
        source1_words_html = " ".join(
            f"<span class='word-tag'>{html_escape.escape(word)}</span>" + 
            (f"<span class='count'>×{count}</span>" if count > 1 else "") 
            for word, count in sorted(source1_words_counter.items())
        )
    else:
        source1_words_html = "<span class='perfect'>All words present in both</span>"
    
    if only_source2:
        source2_words_counter = Counter(only_source2)
        source2_words_html = " ".join(
            f"<span class='word-tag'>{html_escape.escape(word)}</span>" + 
            (f"<span class='count'>×{count}</span>" if count > 1 else "") 
            for word, count in sorted(source2_words_counter.items())
        )
    else:
        source2_words_html = "<span class='perfect'>All words present in both</span>"
    
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Document {doc_num} — {config1['label']} vs {config2['label']}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ 
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
    padding: 20px; 
    color: #e8e8e8;
}}
.container {{ 
    width: 100%;
    max-width: 100%;
    background: rgba(26, 26, 46, 0.95); 
    padding: 24px 28px; 
    border-radius: 16px; 
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05); 
    backdrop-filter: blur(10px);
}}
.header {{ 
    display: flex; 
    align-items: center; 
    justify-content: space-between;
    margin-bottom: 24px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}
.header h1 {{ 
    font-size: 24px; 
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.back-link {{ 
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #64ffda; 
    text-decoration: none;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 18px;
    background: rgba(100,255,218,0.1);
    border: 1px solid rgba(100,255,218,0.3);
    border-radius: 10px;
    transition: all 0.3s ease;
}}
.back-link:hover {{ 
    background: rgba(100,255,218,0.2); 
    transform: translateX(-3px);
}}
.source-badges {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}}
.source-badge {{
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
}}
.source-badge.source1 {{
    background: rgba({config1['color_rgb']},0.2);
    color: {config1['color']};
    border: 1px solid rgba({config1['color_rgb']},0.4);
}}
.source-badge.source2 {{
    background: rgba({config2['color_rgb']},0.2);
    color: {config2['color']};
    border: 1px solid rgba({config2['color_rgb']},0.4);
}}
.vs-text {{
    color: #8892b0;
    font-weight: 600;
    font-size: 14px;
}}
.stats-bar {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px;
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    border: 1px solid rgba(255,255,255,0.08);
}}
.similarity-badge {{
    padding: 12px 22px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 20px;
    letter-spacing: 0.5px;
}}
.similarity-high {{ background: linear-gradient(135deg, #00d26a, #00a854); color: #fff; box-shadow: 0 4px 15px rgba(0,210,106,0.3); }}
.similarity-medium {{ background: linear-gradient(135deg, #ffc107, #ff9800); color: #1a1a2e; box-shadow: 0 4px 15px rgba(255,193,7,0.3); }}
.similarity-low {{ background: linear-gradient(135deg, #ff4757, #ff3838); color: #fff; box-shadow: 0 4px 15px rgba(255,71,87,0.3); }}
.stat-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    padding: 10px 16px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}}
.stat-label {{ color: #8892b0; font-weight: 500; }}
.stat-value {{ font-weight: 700; color: #fff; }}
.stat-value.green {{ color: #00d26a; }}
.stat-value.red {{ color: #ff4757; }}
.stat-value.orange {{ color: #ffc107; }}
.word-analysis {{ 
    display: grid; 
    grid-template-columns: 1fr 1fr; 
    gap: 16px; 
    margin-bottom: 24px; 
}}
.word-box {{ 
    padding: 16px; 
    border-radius: 12px; 
    max-height: 140px; 
    overflow-y: auto; 
}}
.word-box.box1 {{ 
    background: rgba({config1['color_rgb']},0.1); 
    border: 1px solid rgba({config1['color_rgb']},0.3); 
}}
.word-box.box2 {{ 
    background: rgba({config2['color_rgb']},0.1); 
    border: 1px solid rgba({config2['color_rgb']},0.3); 
}}
.word-box h3 {{ 
    margin-bottom: 12px;
    font-size: 12px; 
    font-weight: 700; 
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.word-box.box1 h3 {{ color: {config1['color']}; }}
.word-box.box2 h3 {{ color: {config2['color']}; }}
.words-container {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.word-tag {{ 
    padding: 5px 10px; 
    border-radius: 6px; 
    font-family: 'JetBrains Mono', monospace; 
    font-size: 11px;
    font-weight: 500;
    background: rgba(255,255,255,0.1);
    color: #ccd6f6;
    border: 1px solid rgba(255,255,255,0.2);
}}
.count {{ font-size: 10px; color: #8892b0; margin-right: 4px; }}
.perfect {{ color: #00d26a; font-weight: 600; font-size: 12px; }}
.section-title {{
    font-size: 15px;
    font-weight: 700;
    color: #ccd6f6;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.legend {{
    display: flex;
    gap: 24px;
    padding: 12px 16px;
    background: rgba(0,0,0,0.3);
    border-radius: 10px;
    margin-bottom: 14px;
    font-size: 11px;
    border: 1px solid rgba(255,255,255,0.08);
}}
.legend-item {{ display: flex; align-items: center; gap: 8px; font-weight: 500; color: #8892b0; }}
.comparison-grid {{ 
    display: grid; 
    grid-template-columns: 1fr 1fr; 
    gap: 24px; 
    margin-bottom: 24px;
}}
.text-panel {{ 
    border-radius: 12px; 
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}}
.text-panel.panel1 {{ border-color: rgba({config1['color_rgb']},0.4); }}
.text-panel.panel2 {{ border-color: rgba({config2['color_rgb']},0.4); }}
.panel-header {{
    padding: 14px 18px;
    font-weight: 700;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.text-panel.panel1 .panel-header {{ 
    background: linear-gradient(135deg, rgba({config1['color_rgb']},0.2), rgba({config1['color_rgb']},0.1)); 
    color: {config1['color']}; 
}}
.text-panel.panel2 .panel-header {{ 
    background: linear-gradient(135deg, rgba({config2['color_rgb']},0.2), rgba({config2['color_rgb']},0.1)); 
    color: {config2['color']}; 
}}
.text-content {{
    padding: 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    line-height: 2.2;
    max-height: 550px;
    overflow-y: auto;
    background: rgba(0,0,0,0.4);
}}
.word-common {{ background: rgba(0,210,106,0.2); color: #00d26a; padding: 3px 7px; border-radius: 4px; }}
.word-unique {{ background: rgba(255,193,7,0.25); color: #ffc107; padding: 3px 7px; border-radius: 4px; font-weight: 600; }}

::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.3); border-radius: 4px; }}
::-webkit-scrollbar-thumb {{ background: rgba(100,255,218,0.3); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(100,255,218,0.5); }}

@media (max-width: 900px) {{
    .comparison-grid, .word-analysis {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Document {doc_num}</h1>
        <a href="/compare/{source1}/{source2}" class="back-link">← Back to Summary</a>
    </div>
    
    <div class="source-badges">
        <span class="source-badge source1">{config1['label']}</span>
        <span class="vs-text">vs</span>
        <span class="source-badge source2">{config2['label']}</span>
    </div>
    
    <div class="stats-bar">
        <div class="similarity-badge {'similarity-high' if similarity >= 0.9 else 'similarity-medium' if similarity >= 0.7 else 'similarity-low'}">
            {similarity:.1%} Match
        </div>
        <div class="stat-item"><span class="stat-label">{config1['label']} Words:</span><span class="stat-value">{sim_result.total_gt_words}</span></div>
        <div class="stat-item"><span class="stat-label">Common:</span><span class="stat-value green">{sim_result.correct_words}</span></div>
        <div class="stat-item"><span class="stat-label">Only in {config1['label']}:</span><span class="stat-value red">{len(only_source1)}</span></div>
        <div class="stat-item"><span class="stat-label">Only in {config2['label']}:</span><span class="stat-value orange">{len(only_source2)}</span></div>
    </div>

    <div class="word-analysis">
        <div class="word-box box1">
            <h3>Only in {config1['label']}</h3>
            <div class="words-container">{source1_words_html}</div>
        </div>
        <div class="word-box box2">
            <h3>Only in {config2['label']}</h3>
            <div class="words-container">{source2_words_html}</div>
        </div>
    </div>

    <div class="section-title">Word-by-Word Comparison</div>
    <div class="legend">
        <div class="legend-item"><span class="word-common">word</span> Present in both sources</div>
        <div class="legend-item"><span class="word-unique">word</span> Unique to this source</div>
    </div>
    
    <div class="comparison-grid">
        <div class="text-panel panel1">
            <div class="panel-header">{config1['label']}</div>
            <div class="text-content">{text1_highlighted}</div>
        </div>
        <div class="text-panel panel2">
            <div class="panel-header">{config2['label']}</div>
            <div class="text-content">{text2_highlighted}</div>
        </div>
    </div>
</div>
</body>
</html>
"""


class ComparisonHandler(BaseHTTPRequestHandler):
    """HTTP request handler for comparison pages."""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        
        if path == "/":
            content = generate_home_html()
            self._send_html(content)
        
        elif path.startswith("/compare/"):
            parts = path.split("/")
            # /compare/source1/source2 -> summary
            # /compare/source1/source2/doc_num -> detail
            if len(parts) == 4:
                source1, source2 = parts[2], parts[3]
                if source1 in SOURCE_CONFIG and source2 in SOURCE_CONFIG:
                    content = generate_summary_html(source1, source2)
                    self._send_html(content)
                else:
                    self.send_error(404, "Invalid source")
            elif len(parts) == 5:
                source1, source2 = parts[2], parts[3]
                try:
                    doc_num = int(parts[4])
                    if source1 in SOURCE_CONFIG and source2 in SOURCE_CONFIG:
                        content = generate_detail_html(source1, source2, doc_num)
                        self._send_html(content)
                    else:
                        self.send_error(404, "Invalid source")
                except ValueError:
                    self.send_error(404, "Invalid document number")
            else:
                self.send_error(404, "Not Found")
        
        else:
            self.send_error(404, "Not Found")
    
    def _send_html(self, content):
        """Helper to send HTML response."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def main():
    print("=" * 60)
    print("OCR COMPARISON DASHBOARD - LIVE SERVER")
    print("=" * 60)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"\nAvailable Sources:")
    for key, config in SOURCE_CONFIG.items():
        print(f"  [{config['short']}] {config['label']}: {config['dir']}")
    print(f"\nStarting server on http://localhost:{PORT}")
    print("Compare any two sources interactively")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Open browser
    webbrowser.open(f"http://localhost:{PORT}/")
    
    # Start server
    server = HTTPServer(("", PORT), ComparisonHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
