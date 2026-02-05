"""
OCR Runner Package
Unified OCR pipeline with model routing for Doctr, Surya, and PaddleOCR.
"""

from .ocr_router import run_ocr, OCRResult
from .text_processor import extract_custom_text, create_custom_text, save_for_web_ui

__all__ = ["run_ocr", "OCRResult", "extract_custom_text", "create_custom_text", "save_for_web_ui"]
__version__ = "1.0.0"
