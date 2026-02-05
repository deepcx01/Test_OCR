"""
OCR Router - Routes OCR requests to appropriate engine.
Supports Doctr, Surya (via API), and PaddleOCR (local).
"""

import base64
import io
import os
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from PIL import Image
from loguru import logger

# API Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY", "")
OCR_URL = os.environ.get("OCR_ENDPOINT_URL", "https://api.runpod.ai/v2/6fzr4ximmk7dwd/runsync")

ModelType = Literal["doctr", "surya", "paddle"]


@dataclass
class OCRResult:
    """Container for OCR results."""
    model: str
    custom_text: str
    text: str
    words: list
    raw_json: Dict[str, Any]
    success: bool
    error: Optional[str] = None


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64-encoded string."""
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_image(image_path: str) -> Image.Image:
    """Load image from path or URL."""
    if image_path.startswith(('http://', 'https://')):
        import urllib.request
        with urllib.request.urlopen(image_path) as response:
            return Image.open(io.BytesIO(response.read()))
    return Image.open(image_path)


def call_ocr_api(base64_image: str, model: str) -> Dict[str, Any]:
    """
    Call the OCR API endpoint for Doctr or Surya.
    
    Args:
        base64_image: Base64-encoded image
        model: "doctr" or "surya"
    
    Returns:
        OCR result dictionary
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "input": {
            "model": model,
            "type": "image",
            "format": "base64",
            "data": base64_image
        }
    }
    
    try:
        response = requests.post(OCR_URL, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        if 'output' in result and 'data' in result['output']:
            return {"success": True, "data": result['output']['data'][0]}
        else:
            return {"success": False, "error": f"Unexpected response format: {result}"}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def run_ocr(image_path: str, model: ModelType) -> OCRResult:
    """
    Run OCR on an image using the specified model.
    
    Args:
        image_path: Path to image file or URL
        model: One of "doctr", "surya", "paddle"
    
    Returns:
        OCRResult with extracted text and metadata
    """
    logger.info(f"Running OCR with model '{model}' on: {image_path}")
    
    try:
        # Load and encode image
        image = load_image(image_path)
        base64_image = pil_image_to_base64(image)
        
        if model in ("doctr", "surya"):
            # Use API endpoint
            result = call_ocr_api(base64_image, model)
            
            if not result["success"]:
                return OCRResult(
                    model=model,
                    custom_text="",
                    text="",
                    words=[],
                    raw_json={},
                    success=False,
                    error=result["error"]
                )
            
            data = result["data"]
            return OCRResult(
                model=model,
                custom_text=data.get("custom_text", ""),
                text=data.get("text", ""),
                words=data.get("words", []),
                raw_json=data,
                success=True
            )
            
        elif model == "paddle":
            # Use local PaddleOCR
            from .paddle_local import run_paddle_ocr
            return run_paddle_ocr(image_path, image)
            
        else:
            return OCRResult(
                model=model,
                custom_text="",
                text="",
                words=[],
                raw_json={},
                success=False,
                error=f"Unknown model: {model}"
            )
            
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return OCRResult(
            model=model,
            custom_text="",
            text="",
            words=[],
            raw_json={},
            success=False,
            error=str(e)
        )


def save_ocr_result(result: OCRResult, output_path: str) -> None:
    """Save OCR result to JSON file."""
    output = {
        "model": result.model,
        "success": result.success,
        "data": [{
            "custom_text": result.custom_text,
            "text": result.text,
            "words": result.words,
            **result.raw_json
        }]
    }
    
    if result.error:
        output["error"] = result.error
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Saved OCR result to: {output_path}")
