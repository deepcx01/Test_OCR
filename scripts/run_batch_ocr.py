#!/usr/bin/env python3
"""Batch OCR Processing - Process multiple images against ground truth."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_runner import run_ocr
from ocr_runner.similarity_logic import compute_similarity
from ocr_runner.text_processor import load_text_file, save_custom_text
from loguru import logger


def list_r2_folder(r2_path: str) -> List[str]:
    """List image files in an R2 folder."""
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    target_path = r2_path.replace("r2://", "r2/")
    if not target_path.endswith("/"):
        target_path += "/"
    
    cmd = ["mc", "ls", target_path]
    logger.info(f"debug: Executing command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Failed to list R2 folder: {result.stderr}")
        return []
        
    logger.info(f"debug: mc ls output (first 200 chars): {result.stdout[:200]!r}")
    
    files = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        # mc ls output format: [DATE] [TIME] [SIZE] [STORAGE_CLASS] filename
        # Example: [2026-02-05 09:34:15 UTC] 391KiB STANDARD Input_1.jpg
        # We need to split by spaces, but handle filenames with spaces
        parts = line.split()
        if len(parts) >= 5:
            # Index 0-2: Date/Time/Zone
            # Index 3: Size
            # Index 4: Storage Class (e.g. STANDARD)
            # Index 5+: Filename (joined back)
            
            # Find the index where the filename likely starts (after STANDARD)
            start_index = 5
            filename = " ".join(parts[start_index:])
            
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(filename)
    
    logger.info(f"debug: Parsed {len(files)} files: {files[:5]}")
    return sorted(files)


def download_r2_file(r2_path: str, local_dir: str) -> str:
    """Download a file from R2."""
    local_path = Path(local_dir) / Path(r2_path).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["mc", "cp", r2_path.replace("r2://", "r2/"), str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr}")
    
    return str(local_path)


def process_image(
    image_path: str,
    gt_path: str,
    model: str,
    compare_model: Optional[str],
    output_dir: str
) -> Dict:
    """Process a single image and return results."""
    image_name = Path(image_path).name
    basename = Path(image_path).stem
    
    logger.info(f"Processing: {image_name}")
    
    result = {"image": image_name, "basename": basename, "model": model, "success": False}
    
    try:
        ocr_result = run_ocr(image_path, model)
        
        if not ocr_result.success:
            result["error"] = ocr_result.error
            return result
        
        result["success"] = True
        result["ocr_text"] = ocr_result.custom_text
        result["word_count"] = len(ocr_result.words)
        
        save_custom_text(ocr_result.custom_text, Path(output_dir) / f"{basename}_{model}.txt")
        
        try:
            gt_text = load_text_file(gt_path)
            gt_sim = compute_similarity(gt_text, ocr_result.custom_text)
            result["gt_comparison"] = {
                "similarity": gt_sim.similarity_score,
                "total_gt_words": gt_sim.total_gt_words,
                "correct_words": gt_sim.correct_words,
                "missing_count": len(gt_sim.missing_words),
            }
            logger.info(f"  Match: {gt_sim.similarity_score:.1f}% | Missing: {len(gt_sim.missing_words)}")
        except FileNotFoundError:
            logger.warning(f"  GT not found: {gt_path}")
            result["gt_comparison"] = None
        
        if compare_model:
            ocr_result_2 = run_ocr(image_path, compare_model)
            if ocr_result_2.success:
                model_sim = compute_similarity(ocr_result.custom_text, ocr_result_2.custom_text)
                result["model_comparison"] = {
                    "model2": compare_model,
                    "similarity": model_sim.similarity_score,
                }
                save_custom_text(ocr_result_2.custom_text, Path(output_dir) / f"{basename}_{compare_model}.txt")
        
    except BaseException as e:
        logger.exception(f"Error processing {image_name}: {e}")
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result


def generate_summary(results: List[Dict], model: str) -> str:
    """Generate text summary."""
    successful = [r for r in results if r.get("success")]
    with_gt = [r for r in successful if r.get("gt_comparison")]
    
    lines = [
        "=" * 60,
        "BATCH OCR BENCHMARK REPORT",
        "=" * 60,
        f"Model: {model}",
        f"Total: {len(results)} | Success: {len(successful)} | With GT: {len(with_gt)}",
        "-" * 60,
    ]
    
    if with_gt:
        similarities = [r["gt_comparison"]["similarity"] for r in with_gt]
        avg = sum(similarities) / len(similarities)
        high = len([s for s in similarities if s >= 90])
        med = len([s for s in similarities if 70 <= s < 90])
        low = len([s for s in similarities if s < 70])
        
        lines.extend([
            f"Avg Similarity: {avg:.2f}%",
            f"High (≥90%): {high} | Medium (70-89%): {med} | Low (<70%): {low}",
            "-" * 60,
            f"{'File':<30} {'Score':>8} {'Words':>8} {'Missing':>8}",
            "-" * 60,
        ])
        
        for r in sorted(with_gt, key=lambda x: x["gt_comparison"]["similarity"], reverse=True):
            gt = r["gt_comparison"]
            lines.append(f"{r['image']:<30} {gt['similarity']:>7.1f}% {gt['total_gt_words']:>8} {gt['missing_count']:>8}")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Batch OCR processing")
    parser.add_argument("--images-folder", "-i", required=True)
    parser.add_argument("--gt-folder", "-g", required=True)
    parser.add_argument("--model", "-m", required=True, choices=["doctr", "surya", "paddle"])
    parser.add_argument("--compare-model", "-c", choices=["doctr", "surya", "paddle"])
    parser.add_argument("--output-dir", "-o", default="outputs")
    parser.add_argument("--output-json")
    parser.add_argument("--output-summary")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    is_r2 = args.images_folder.startswith(("r2://", "r2/"))
    
    if is_r2:
        image_files = list_r2_folder(args.images_folder)
        logger.info(f"Found {len(image_files)} images in R2")
        
        # Debug: List GT folder to verify naming
        if args.gt_folder.startswith(("r2://", "r2/")):
            gt_folder = args.gt_folder.replace("r2://", "r2/")
            logger.info(f"Debug: Listing GT folder: {gt_folder}")
            try:
                available_gt = list_r2_folder(gt_folder)
                logger.info(f"Debug: Found {len(available_gt)} files in GT folder")
                if available_gt:
                    logger.info(f"Debug: First few GT files: {available_gt[:5]}")
            except Exception as e:
                logger.warning(f"Debug: Failed to list GT folder: {e}")
        
        temp_dir = output_dir / "temp_images"
        temp_dir.mkdir(exist_ok=True)
        
        images_to_process = []
        for img in image_files:
            folder_cleaned = args.images_folder.replace("r2://", "r2/")
            r2_path = f"{folder_cleaned.rstrip('/')}/{img}"
            try:
                local = download_r2_file(r2_path, str(temp_dir))
                images_to_process.append((local, img))
            except Exception as e:
                logger.error(f"Failed: {img}: {e}")
    else:
        images_folder = Path(args.images_folder)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(images_folder.glob(ext))
        images_to_process = [(str(p), p.name) for p in sorted(image_files)]
        logger.info(f"Found {len(images_to_process)} images locally")
    # Process images
    results = []
    
    temp_gt_dir = output_dir / "temp_gt"
    temp_gt_dir.mkdir(exist_ok=True)
    
    for local_path, image_name in images_to_process:
        basename = Path(image_name).stem
        
        # Resolve Ground Truth path
        if args.gt_folder.startswith(("r2://", "r2/")):
            gt_folder_cleaned = args.gt_folder.replace("r2://", "r2/")
            gt_r2_path = f"{gt_folder_cleaned.rstrip('/')}/{basename}.json"
            try:
                logger.info(f"Downloading GT from R2: {gt_r2_path}")
                gt_path = download_r2_file(gt_r2_path, str(temp_gt_dir))
            except Exception as e:
                logger.warning(f"Could not download GT {gt_r2_path}: {e}")
                gt_path = gt_r2_path # Fallback to path string (will trigger FileNotFoundError later)
        else:
            gt_path = f"{args.gt_folder.rstrip('/')}/{basename}.json"
            
        result = process_image(local_path, gt_path, args.model, args.compare_model, str(output_dir))
        results.append(result)
    
    summary = generate_summary(results, args.model)
    print(summary)
    
    if args.output_summary:
        Path(args.output_summary).write_text(summary)
        logger.success(f"Summary: {args.output_summary}")
    
    if args.output_json:
        data = {
            "model": args.model,
            "compare_model": args.compare_model,
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        Path(args.output_json).write_text(json.dumps(data, indent=2))
        logger.success(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
