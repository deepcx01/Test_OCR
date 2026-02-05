#!/usr/bin/env python3
"""
Compare OCR Outputs CLI - Compare model outputs or against ground truth.

Usage:
    # Compare model output to ground truth
    python scripts/compare_outputs.py --reference gt.txt --compare doctr_output.txt
    
    # Compare two model outputs
    python scripts/compare_outputs.py --source1 doctr.txt --source2 surya.txt
    
    # Output as JSON
    python scripts/compare_outputs.py --reference gt.txt --compare ocr.txt --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
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
        raise ValueError(f"Invalid R2 path format: {r2_path}")
    
    bucket, remote_path = parts
    filename = Path(remote_path).name
    local_path = Path(output_dir) / filename
    
    import subprocess
    cmd = ["mc", "cp", f"r2/{bucket}/{remote_path}", str(local_path)]
    logger.info(f"Downloading from R2: {r2_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download from R2: {result.stderr}")
    
    return str(local_path)


def format_result(result: SimilarityResult, source1_label: str, source2_label: str) -> str:
    """Format comparison result as human-readable text."""
    lines = [
        "=" * 60,
        "OCR COMPARISON REPORT",
        "=" * 60,
        f"Source 1 ({source1_label})",
        f"Source 2 ({source2_label})",
        "-" * 60,
        f"Similarity Score  : {result.similarity_score:.2f}%",
        f"Total Words (S1)  : {result.total_gt_words}",
        f"Correct Words     : {result.correct_words}",
        f"Missing/Incorrect : {result.incorrect_words}",
        "-" * 60,
    ]
    
    if result.missing_words:
        from collections import Counter
        lines.append("Missing Words:")
        counter = Counter(result.missing_words)
        for word, count in counter.most_common(20):  # Top 20
            if count > 1:
                lines.append(f"  - '{word}' (x{count})")
            else:
                lines.append(f"  - '{word}'")
        if len(counter) > 20:
            lines.append(f"  ... and {len(counter) - 20} more unique words")
    else:
        lines.append("Missing Words: None")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def result_to_dict(result: SimilarityResult, source1_label: str, source2_label: str) -> dict:
    """Convert result to JSON-serializable dictionary."""
    return {
        "source1": source1_label,
        "source2": source2_label,
        "similarity_score": result.similarity_score,
        "total_source1_words": result.total_gt_words,
        "correct_words": result.correct_words,
        "incorrect_words": result.incorrect_words,
        "missing_words": result.missing_words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare OCR outputs or against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare OCR output to ground truth
    python compare_outputs.py --reference gt.txt --compare ocr_output.txt
    
    # Compare two model outputs
    python compare_outputs.py --source1 doctr.txt --source2 surya.txt
    
    # Output as JSON
    python compare_outputs.py --reference gt.txt --compare ocr.txt --json
    
    # Compare files from R2
    python compare_outputs.py --reference r2://bucket/gt.txt --compare r2://bucket/ocr.txt
        """
    )
    
    # Option 1: Reference + Compare (GT vs OCR)
    parser.add_argument(
        "--reference", "-r",
        help="Reference text file (e.g., ground truth)"
    )
    parser.add_argument(
        "--compare", "-c",
        help="Text file to compare against reference"
    )
    
    # Option 2: Source1 + Source2 (Any two sources)
    parser.add_argument(
        "--source1", "-s1",
        help="First source text file"
    )
    parser.add_argument(
        "--source2", "-s2",
        help="Second source text file"
    )
    
    # Output options
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save result to file"
    )
    
    args = parser.parse_args()
    
    # Determine which mode we're in
    if args.reference and args.compare:
        file1, file2 = args.reference, args.compare
        label1, label2 = "Reference", "Compare"
    elif args.source1 and args.source2:
        file1, file2 = args.source1, args.source2
        label1 = Path(args.source1).stem
        label2 = Path(args.source2).stem
    else:
        parser.error("Provide either --reference/--compare or --source1/--source2")
        return
    
    # Resolve R2 paths
    file1 = resolve_r2_path(file1)
    file2 = resolve_r2_path(file2)
    
    # Load text files
    try:
        text1 = load_text_file(file1)
        text2 = load_text_file(file2)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    
    # Compute similarity
    result = compute_similarity(text1, text2)
    
    # Format output
    if args.json:
        output = json.dumps(result_to_dict(result, label1, label2), indent=2)
    else:
        output = format_result(result, label1, label2)
    
    # Save or print
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.success(f"Result saved to: {args.output}")
    else:
        print(output)
    
    # Exit with code based on similarity
    if result.similarity_score >= 90:
        sys.exit(0)  # High match
    elif result.similarity_score >= 70:
        sys.exit(0)  # Acceptable match
    else:
        sys.exit(1)  # Low match


if __name__ == "__main__":
    main()
