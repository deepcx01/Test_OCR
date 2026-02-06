#!/usr/bin/env python3
"""OCR CLI - Run OCR on images with model selection."""

import argparse
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_runner import run_ocr
from ocr_runner.ocr_router import save_ocr_result
from ocr_runner.text_processor import save_custom_text, save_for_web_ui
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
    logger.info(f"Downloading: {r2_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"R2 download failed: {result.stderr}")
    
    return str(local_path)


def main():
    parser = argparse.ArgumentParser(description="Run OCR on images")
    parser.add_argument("--image", "-i", required=True, help="Image path (local, URL, or r2://)")
    parser.add_argument("--model", "-m", required=True, choices=["doctr", "surya", "paddle"])
    parser.add_argument("--output", "-o", help="Save JSON output")
    parser.add_argument("--output-text", "-t", help="Save custom_text as .txt")
    parser.add_argument("--web-format", "-w", type=int, help="Save for web UI (document number)")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--endpoint-url", help="OCR endpoint URL")
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["RUNPOD_API_KEY"] = args.api_key
    if args.endpoint_url:
        os.environ["OCR_ENDPOINT_URL"] = args.endpoint_url
    
    image_path = resolve_r2_path(args.image)
    result = run_ocr(image_path, args.model)
    
    if not result.success:
        logger.error(f"OCR failed: {result.error}")
        sys.exit(1)
    
    if args.output:
        save_ocr_result(result, args.output)
    
    if args.output_text:
        save_custom_text(result.custom_text, args.output_text)
        logger.success(f"Saved custom_text to: {args.output_text}")
    
    if args.web_format:
        web_path = save_for_web_ui(result.custom_text, args.model, args.web_format)
        logger.success(f"Saved for web UI: {web_path}")
    
    logger.info(f"OCR completed: {len(result.words)} words extracted")
    
    if not args.output and not args.output_text and not args.web_format:
        print(result.custom_text)


if __name__ == "__main__":
    main()
