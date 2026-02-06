#!/usr/bin/env python3
"""Compare OCR outputs or against ground truth."""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_runner.similarity_logic import compute_similarity, SimilarityResult
from ocr_runner.text_processor import load_text_file
from loguru import logger


def resolve_r2_path(r2_path: str, output_dir: str = "/tmp") -> str:
    """Download file from R2 if path starts with r2://."""
    if not r2_path.startswith("r2://"):
        return r2_path
    
    parts = r2_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid R2 path: {r2_path}")
    
    bucket, remote_path = parts
    local_path = Path(output_dir) / Path(remote_path).name
    
    cmd = ["mc", "cp", f"r2/{bucket}/{remote_path}", str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"R2 download failed: {result.stderr}")
    
    return str(local_path)


def format_result(result: SimilarityResult, label1: str, label2: str) -> str:
    """Format result as text report."""
    lines = [
        "=" * 60,
        "OCR COMPARISON REPORT",
        "=" * 60,
        f"Source 1 ({label1})",
        f"Source 2 ({label2})",
        "-" * 60,
        f"Similarity Score  : {result.similarity_score:.2f}%",
        f"Total Words (S1)  : {result.total_gt_words}",
        f"Correct Words     : {result.correct_words}",
        f"Missing/Incorrect : {result.incorrect_words}",
        "-" * 60,
    ]
    
    if result.missing_words:
        lines.append("Missing Words:")
        for word, count in Counter(result.missing_words).most_common(20):
            lines.append(f"  - '{word}'" + (f" (x{count})" if count > 1 else ""))
        if len(set(result.missing_words)) > 20:
            lines.append(f"  ... and {len(set(result.missing_words)) - 20} more")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare OCR outputs")
    parser.add_argument("--reference", "-r", help="Reference file (ground truth)")
    parser.add_argument("--compare", "-c", help="File to compare against reference")
    parser.add_argument("--source1", "-s1", help="First source file")
    parser.add_argument("--source2", "-s2", help="Second source file")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Save result to file")
    
    args = parser.parse_args()
    
    if args.reference and args.compare:
        file1, file2 = args.reference, args.compare
        label1, label2 = "Reference", "Compare"
    elif args.source1 and args.source2:
        file1, file2 = args.source1, args.source2
        label1, label2 = Path(args.source1).stem, Path(args.source2).stem
    else:
        parser.error("Provide --reference/--compare or --source1/--source2")
        return
    
    file1 = resolve_r2_path(file1)
    file2 = resolve_r2_path(file2)
    
    try:
        text1 = load_text_file(file1)
        text2 = load_text_file(file2)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    
    result = compute_similarity(text1, text2)
    
    if args.json:
        output = json.dumps({
            "source1": label1, "source2": label2,
            "similarity_score": result.similarity_score,
            "total_words": result.total_gt_words,
            "correct_words": result.correct_words,
            "missing_words": result.missing_words,
        }, indent=2)
    else:
        output = format_result(result, label1, label2)
    
    if args.output:
        Path(args.output).write_text(output)
        logger.success(f"Saved: {args.output}")
    else:
        print(output)
    
    sys.exit(0 if result.similarity_score >= 70 else 1)


if __name__ == "__main__":
    main()
