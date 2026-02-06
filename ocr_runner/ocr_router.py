"""OCR Router - Routes OCR requests to Doctr, Surya (API), or PaddleOCR (local)."""

import base64
import io
import os
import json
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal
from PIL import Image
from loguru import logger

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
OCR_URL = os.environ.get("OCR_ENDPOINT_URL")

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


def _error_result(model: str, error: str) -> OCRResult:
    """Create an error OCRResult."""
    return OCRResult(model=model, custom_text="", text="", words=[], raw_json={}, success=False, error=error)


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_image(path: str) -> Image.Image:
    """Load image from path or URL."""
    if path.startswith(('http://', 'https://')):
        import urllib.request
        with urllib.request.urlopen(path) as resp:
            return Image.open(io.BytesIO(resp.read()))
    return Image.open(path)


def call_api(base64_image: str, model: str) -> Dict[str, Any]:
    """Call OCR API endpoint."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {"input": {"model": model, "type": "image", "format": "base64", "data": base64_image}}
    
    try:
        resp = requests.post(OCR_URL, headers=headers, json=payload, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        
        if 'output' in result and 'data' in result['output']:
            return {"success": True, "data": result['output']['data'][0]}
        return {"success": False, "error": f"Unexpected format: {result}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def run_ocr(image_path: str, model: ModelType) -> OCRResult:
    """Run OCR on image using specified model."""
    logger.info(f"Running OCR: {model} on {image_path}")
    
    try:
        image = load_image(image_path)
        b64 = pil_to_base64(image)
        
        if model in ("doctr", "surya"):
            result = call_api(b64, model)
            if not result["success"]:
                return _error_result(model, result["error"])
            
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
            from .paddle_local import run_paddle_ocr
            return run_paddle_ocr(image_path, image)
        
        return _error_result(model, f"Unknown model: {model}")
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return _error_result(model, str(e))


def save_ocr_result(result: OCRResult, output_path: str) -> None:
    """Save OCR result to JSON file."""
    output = {
        "model": result.model,
        "success": result.success,
        "data": [{"custom_text": result.custom_text, "text": result.text, "words": result.words, **result.raw_json}]
    }
    if result.error:
        output["error"] = result.error
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    logger.success(f"Saved: {output_path}")
