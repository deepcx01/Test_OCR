#!/usr/bin/env python3
"""
Download from R2 - Helper script to download files from R2 bucket.

Usage:
    python scripts/download_from_r2.py --bucket ocr-data --path images/doc.jpg --output inputs/
    python scripts/download_from_r2.py --r2-path r2://ocr-data/images/doc.jpg --output inputs/
"""

import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger


def download_from_r2(bucket: str, remote_path: str, output_dir: str) -> str:
    """
    Download file from R2 bucket using minio client.
    
    Args:
        bucket: R2 bucket name
        remote_path: Path within bucket
        output_dir: Local directory to save file
    
    Returns:
        Local path to downloaded file
    """
    filename = Path(remote_path).name
    local_path = Path(output_dir) / filename
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build mc command
    r2_full_path = f"r2/{bucket}/{remote_path}"
    cmd = ["mc", "cp", r2_full_path, str(local_path)]
    
    logger.info(f"Downloading: {r2_full_path} -> {local_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Download failed: {result.stderr}")
        raise RuntimeError(f"Failed to download from R2: {result.stderr}")
    
    logger.success(f"Downloaded: {local_path}")
    return str(local_path)


def parse_r2_path(r2_path: str) -> tuple:
    """Parse r2://bucket/path format."""
    if not r2_path.startswith("r2://"):
        raise ValueError(f"Invalid R2 path: {r2_path}. Must start with r2://")
    
    parts = r2_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid R2 path format: {r2_path}")
    
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Download files from R2 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download using bucket and path
    python download_from_r2.py --bucket ocr-data --path images/doc_1.jpg --output inputs/
    
    # Download using r2:// URL
    python download_from_r2.py --r2-path r2://ocr-data/images/doc_1.jpg --output inputs/
    
    # Download multiple files
    python download_from_r2.py --bucket ocr-data --path "images/*.jpg" --output inputs/
        """
    )
    
    parser.add_argument(
        "--bucket", "-b",
        help="R2 bucket name"
    )
    parser.add_argument(
        "--path", "-p",
        help="Path within bucket"
    )
    parser.add_argument(
        "--r2-path",
        help="Full R2 path (r2://bucket/path)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Determine bucket and path
    if args.r2_path:
        bucket, remote_path = parse_r2_path(args.r2_path)
    elif args.bucket and args.path:
        bucket, remote_path = args.bucket, args.path
    else:
        parser.error("Provide either --r2-path or both --bucket and --path")
        return
    
    try:
        local_path = download_from_r2(bucket, remote_path, args.output)
        print(local_path)  # Print path for scripting
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
