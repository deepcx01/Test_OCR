#!/usr/bin/env python3
"""
OCR CLI - Run OCR on images with model selection.

Usage:
    python scripts/run_ocr_cli.py --image input.jpg --model doctr --output output.json
    python scripts/run_ocr_cli.py --image r2://bucket/path/image.jpg --model surya
    python scripts/run_ocr_cli.py --image https://example.com/image.jpg --model paddle
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_runner import run_ocr
from ocr_runner.ocr_router import save_ocr_result
from ocr_runner.text_processor import save_custom_text, save_for_web_ui
from loguru import logger


def resolve_r2_path(r2_path: str, output_dir: str = "/tmp") -> str:
    """
    Download file from R2 if path starts with r2://.
    
    Args:
        r2_path: Path starting with r2://bucket/path
        output_dir: Directory to save downloaded file
    
    Returns:
        Local path to downloaded file
    """
    if not r2_path.startswith("r2://"):
        return r2_path
    
    # Parse r2://bucket/path format
    parts = r2_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid R2 path format: {r2_path}")
    
    bucket, remote_path = parts
    filename = Path(remote_path).name
    local_path = Path(output_dir) / filename
    
    # Use minio client to download
    import subprocess
    cmd = ["mc", "cp", f"r2/{bucket}/{remote_path}", str(local_path)]
    logger.info(f"Downloading from R2: {r2_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download from R2: {result.stderr}")
    
    logger.success(f"Downloaded to: {local_path}")
    return str(local_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR on images with model selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Doctr OCR on local image
    python run_ocr_cli.py --image input.jpg --model doctr --output output.json
    
    # Run Surya OCR and save custom text
    python run_ocr_cli.py --image input.jpg --model surya --output-text output.txt
    
    # Run PaddleOCR and save for web UI comparison
    python run_ocr_cli.py --image input.jpg --model paddle --web-format 1
    
    # Run OCR on image from R2 bucket
    python run_ocr_cli.py --image r2://my-bucket/images/doc.jpg --model doctr
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to input image (local path, URL, or r2://bucket/path)"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=["doctr", "surya", "paddle"],
        help="OCR model to use"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save JSON output (default: stdout)"
    )
    parser.add_argument(
        "--output-text", "-t",
        help="Path to save custom_text as .txt file"
    )
    parser.add_argument(
        "--web-format", "-w",
        type=int,
        metavar="DOC_NUM",
        help="Save in web UI format with given document number (e.g., --web-format 1)"
    )
    parser.add_argument(
        "--api-key",
        help="RunPod API key (or set RUNPOD_API_KEY env var)"
    )
    parser.add_argument(
        "--endpoint-url",
        help="OCR endpoint URL (or set OCR_ENDPOINT_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.api_key:
        os.environ["RUNPOD_API_KEY"] = args.api_key
    if args.endpoint_url:
        os.environ["OCR_ENDPOINT_URL"] = args.endpoint_url
    
    # Resolve R2 path if needed
    image_path = resolve_r2_path(args.image)
    
    # Run OCR
    result = run_ocr(image_path, args.model)
    
    if not result.success:
        logger.error(f"OCR failed: {result.error}")
        sys.exit(1)
    
    # Save outputs
    if args.output:
        save_ocr_result(result, args.output)
    
    if args.output_text:
        save_custom_text(result.custom_text, args.output_text)
        logger.success(f"Saved custom_text to: {args.output_text}")
    
    if args.web_format:
        web_path = save_for_web_ui(result.custom_text, args.model, args.web_format)
        logger.success(f"Saved for web UI: {web_path}")
    
    # Print summary
    word_count = len(result.words)
    logger.info(f"OCR completed: {word_count} words extracted")
    
    # If no output file specified, print custom_text
    if not args.output and not args.output_text and not args.web_format:
        print("\n--- Custom Text ---")
        print(result.custom_text)
        print("-------------------")


if __name__ == "__main__":
    main()
