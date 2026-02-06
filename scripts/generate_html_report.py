#!/usr/bin/env python3
"""
Generate HTML Report for OCR Benchmark Results.

Creates a beautiful, self-contained HTML report with comparison results
that can be downloaded as a GitHub Actions artifact.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_runner.similarity_logic import compute_similarity, tokenize
from ocr_runner.text_processor import load_text_file
from collections import Counter


def get_word_differences(source_text: str, target_text: str):
    """Get missing words and extra words between two texts."""
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
    
    extra = []
    for word, target_count in target_counter.items():
        source_count = source_counter.get(word, 0)
        diff = target_count - source_count
        if diff > 0:
            extra.extend([word] * diff)
    
    return missing, extra


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def generate_html_report(
    model_name: str,
    ocr_text: str,
    gt_text: str = None,
    compare_model_name: str = None,
    compare_text: str = None,
    image_source: str = None,
    run_number: str = None
) -> str:
    """Generate a beautiful HTML report for OCR benchmark results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate metrics
    gt_comparison = None
    model_comparison = None
    
    if gt_text:
        gt_result = compute_similarity(gt_text, ocr_text)
        missing_words, extra_words = get_word_differences(gt_text, ocr_text)
        gt_comparison = {
            "similarity": gt_result.similarity_score,
            "total_gt_words": gt_result.total_gt_words,
            "correct_words": gt_result.correct_words,
            "incorrect_words": gt_result.incorrect_words,
            "missing_words": missing_words[:50],  # Limit for display
            "extra_words": extra_words[:50],
            "missing_count": len(missing_words),
            "extra_count": len(extra_words),
        }
    
    if compare_text and compare_model_name:
        compare_result = compute_similarity(ocr_text, compare_text)
        model_comparison = {
            "model1": model_name,
            "model2": compare_model_name,
            "similarity": compare_result.similarity_score,
        }
    
    # Generate HTML sections
    gt_section = ""
    if gt_comparison:
        similarity_class = "high" if gt_comparison["similarity"] >= 90 else "medium" if gt_comparison["similarity"] >= 70 else "low"
        
        missing_words_html = ""
        if gt_comparison["missing_words"]:
            word_counts = Counter(gt_comparison["missing_words"])
            for word, count in word_counts.most_common(20):
                word_escaped = escape_html(word)
                if count > 1:
                    missing_words_html += f'<span class="word-chip missing">{word_escaped} <small>×{count}</small></span>'
                else:
                    missing_words_html += f'<span class="word-chip missing">{word_escaped}</span>'
            if len(word_counts) > 20:
                missing_words_html += f'<span class="word-chip more">+{len(word_counts) - 20} more</span>'
        else:
            missing_words_html = '<span class="no-issues">None - Perfect match!</span>'
        
        extra_words_html = ""
        if gt_comparison["extra_words"]:
            word_counts = Counter(gt_comparison["extra_words"])
            for word, count in word_counts.most_common(20):
                word_escaped = escape_html(word)
                if count > 1:
                    extra_words_html += f'<span class="word-chip extra">{word_escaped} <small>×{count}</small></span>'
                else:
                    extra_words_html += f'<span class="word-chip extra">{word_escaped}</span>'
            if len(word_counts) > 20:
                extra_words_html += f'<span class="word-chip more">+{len(word_counts) - 20} more</span>'
        else:
            extra_words_html = '<span class="no-issues">None</span>'
        
        gt_section = f'''
        <section class="comparison-section">
            <h2>📊 Ground Truth Comparison</h2>
            
            <div class="stats-grid">
                <div class="stat-card primary">
                    <div class="stat-value {similarity_class}">{gt_comparison["similarity"]:.1f}%</div>
                    <div class="stat-label">Similarity Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{gt_comparison["total_gt_words"]}</div>
                    <div class="stat-label">GT Words</div>
                </div>
                <div class="stat-card success">
                    <div class="stat-value">{gt_comparison["correct_words"]}</div>
                    <div class="stat-label">Correct</div>
                </div>
                <div class="stat-card danger">
                    <div class="stat-value">{gt_comparison["incorrect_words"]}</div>
                    <div class="stat-label">Incorrect</div>
                </div>
            </div>
            
            <div class="word-analysis">
                <div class="word-group">
                    <h4>❌ Missing Words <span class="count">({gt_comparison["missing_count"]})</span></h4>
                    <div class="word-chips">{missing_words_html}</div>
                </div>
                <div class="word-group">
                    <h4>➕ Extra Words <span class="count">({gt_comparison["extra_count"]})</span></h4>
                    <div class="word-chips">{extra_words_html}</div>
                </div>
            </div>
        </section>
        '''
    
    model_section = ""
    if model_comparison:
        model_section = f'''
        <section class="comparison-section">
            <h2>🔄 Model Comparison</h2>
            <div class="model-compare">
                <span class="model-badge">{model_comparison["model1"]}</span>
                <span class="vs">vs</span>
                <span class="model-badge secondary">{model_comparison["model2"]}</span>
                <span class="similarity-badge">{model_comparison["similarity"]:.1f}% similar</span>
            </div>
        </section>
        '''
    
    # OCR Output preview (truncated)
    ocr_preview = escape_html(ocr_text[:2000])
    if len(ocr_text) > 2000:
        ocr_preview += f"\n\n... [{len(ocr_text) - 2000} more characters]"
    
    gt_preview = ""
    if gt_text:
        gt_preview_text = escape_html(gt_text[:2000])
        if len(gt_text) > 2000:
            gt_preview_text += f"\n\n... [{len(gt_text) - 2000} more characters]"
        gt_preview = f'''
        <div class="text-panel">
            <h4>Ground Truth</h4>
            <pre class="text-content gt-text">{gt_preview_text}</pre>
        </div>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Benchmark Report - {model_name}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --accent-cyan: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 48px;
            padding-bottom: 32px;
            border-bottom: 1px solid var(--border);
        }}
        
        .logo {{
            font-size: 48px;
            margin-bottom: 16px;
        }}
        
        h1 {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 16px;
        }}
        
        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .meta-item strong {{
            color: var(--text-primary);
        }}
        
        .model-badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            color: white;
        }}
        
        .model-badge.secondary {{
            background: linear-gradient(135deg, var(--accent-yellow), var(--accent-red));
        }}
        
        section {{
            margin-bottom: 40px;
        }}
        
        h2 {{
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text-primary);
        }}
        
        .comparison-section {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 28px;
            border: 1px solid var(--border);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        
        .stat-card {{
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--border);
        }}
        
        .stat-card.primary {{
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(163, 113, 247, 0.1));
            border-color: rgba(88, 166, 255, 0.3);
        }}
        
        .stat-card.success {{
            border-color: rgba(63, 185, 80, 0.3);
        }}
        
        .stat-card.danger {{
            border-color: rgba(248, 81, 73, 0.3);
        }}
        
        .stat-value {{
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        
        .stat-value.high {{ color: var(--accent-green); }}
        .stat-value.medium {{ color: var(--accent-yellow); }}
        .stat-value.low {{ color: var(--accent-red); }}
        
        .stat-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .word-analysis {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        @media (max-width: 768px) {{
            .word-analysis {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .word-group {{
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .word-group h4 {{
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }}
        
        .word-group h4 .count {{
            color: var(--text-secondary);
            font-weight: 400;
        }}
        
        .word-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .word-chip {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .word-chip.missing {{
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
            border: 1px solid rgba(248, 81, 73, 0.3);
        }}
        
        .word-chip.extra {{
            background: rgba(210, 153, 34, 0.15);
            color: var(--accent-yellow);
            border: 1px solid rgba(210, 153, 34, 0.3);
        }}
        
        .word-chip.more {{
            background: var(--bg-secondary);
            color: var(--text-secondary);
            border: 1px solid var(--border);
        }}
        
        .word-chip small {{
            opacity: 0.7;
        }}
        
        .no-issues {{
            color: var(--accent-green);
            font-size: 14px;
        }}
        
        .model-compare {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            flex-wrap: wrap;
        }}
        
        .vs {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-secondary);
        }}
        
        .similarity-badge {{
            padding: 8px 20px;
            border-radius: 20px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            font-weight: 600;
        }}
        
        .text-panels {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        @media (max-width: 900px) {{
            .text-panels {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .text-panel {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }}
        
        .text-panel h4 {{
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text-secondary);
        }}
        
        .text-content {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            line-height: 1.7;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        
        .ocr-text {{
            border-left: 3px solid var(--accent-cyan);
        }}
        
        .gt-text {{
            border-left: 3px solid var(--accent-green);
        }}
        
        footer {{
            text-align: center;
            padding-top: 32px;
            margin-top: 48px;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 13px;
        }}
        
        footer a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}
        
        footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">📄</div>
            <h1>OCR Benchmark Report</h1>
            <p class="subtitle">Automated text extraction quality assessment</p>
            
            <div class="meta-info">
                <div class="meta-item">
                    <span>🤖</span>
                    <span>Model:</span>
                    <span class="model-badge">{model_name.upper()}</span>
                </div>
                {f'<div class="meta-item"><span>📁</span><span>Source:</span><strong>{escape_html(image_source or "N/A")}</strong></div>' if image_source else ''}
                {f'<div class="meta-item"><span>🔢</span><span>Run:</span><strong>#{run_number}</strong></div>' if run_number else ''}
                <div class="meta-item">
                    <span>🕐</span>
                    <span>Generated:</span>
                    <strong>{timestamp}</strong>
                </div>
            </div>
        </header>
        
        {gt_section}
        
        {model_section}
        
        <section>
            <h2>📝 Text Output Preview</h2>
            <div class="text-panels">
                <div class="text-panel">
                    <h4>OCR Output ({model_name})</h4>
                    <pre class="text-content ocr-text">{ocr_preview}</pre>
                </div>
                {gt_preview}
            </div>
        </section>
        
        <footer>
            <p>Generated by <a href="https://github.com">OCR Benchmark</a> • {timestamp}</p>
        </footer>
    </div>
</body>
</html>'''
    
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML report for OCR benchmark results"
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Primary OCR model name (doctr, surya, paddle)"
    )
    parser.add_argument(
        "--ocr-text", "-t",
        required=True,
        help="Path to OCR output text file"
    )
    parser.add_argument(
        "--gt-text", "-g",
        help="Path to ground truth text file (optional)"
    )
    parser.add_argument(
        "--compare-model",
        help="Second model name for comparison"
    )
    parser.add_argument(
        "--compare-text",
        help="Path to second model's text output"
    )
    parser.add_argument(
        "--image-source",
        help="Original image source path/URL"
    )
    parser.add_argument(
        "--run-number",
        help="GitHub Actions run number"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output HTML file path"
    )
    
    args = parser.parse_args()
    
    # Load texts
    ocr_text = load_text_file(args.ocr_text)
    gt_text = load_text_file(args.gt_text) if args.gt_text else None
    compare_text = load_text_file(args.compare_text) if args.compare_text else None
    
    # Generate HTML
    html = generate_html_report(
        model_name=args.model,
        ocr_text=ocr_text,
        gt_text=gt_text,
        compare_model_name=args.compare_model,
        compare_text=compare_text,
        image_source=args.image_source,
        run_number=args.run_number
    )
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    
    print(f"✅ HTML report generated: {output_path}")


if __name__ == "__main__":
    main()
