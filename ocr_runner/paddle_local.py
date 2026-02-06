"""PaddleOCR Runner - Local OCR using PaddleOCR."""

from typing import Optional
from PIL import Image
from loguru import logger

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

_ocr_instance: Optional['PaddleOCR'] = None


def get_paddle_instance() -> 'PaddleOCR':
    """Get or create PaddleOCR singleton instance."""
    global _ocr_instance
    if _ocr_instance is None:
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR is not installed")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            enable_mkldnn=False  # Disable MKLDNN to fix runner crash
        )
    return _ocr_instance


def run_paddle_ocr(image_path: str, image: Optional[Image.Image] = None):
    """Run PaddleOCR on an image and return OCRResult."""
    from .ocr_router import OCRResult
    
    try:
        ocr = get_paddle_instance()
        logger.info(f"Running PaddleOCR on: {image_path}")
        result = ocr.predict(image_path)
        
        words = []
        words_bboxes = []
        
        for page in result:
            rec_texts = page.get('rec_texts', []) if hasattr(page, 'get') else getattr(page, 'rec_texts', [])
            rec_polys = page.get('rec_polys', []) if hasattr(page, 'get') else getattr(page, 'rec_polys', [])
            
            for i, text in enumerate(rec_texts):
                words.append(text)
                if rec_polys and i < len(rec_polys):
                    poly = rec_polys[i]
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    words_bboxes.append([
                        int(min(x_coords)), int(min(y_coords)),
                        int(max(x_coords)), int(max(y_coords))
                    ])
        
        return OCRResult(
            model="paddle",
            custom_text="\n".join(words),
            text=" ".join(words),
            words=words,
            raw_json={"words_bboxes": words_bboxes},
            success=True
        )
        
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        return OCRResult(
            model="paddle",
            custom_text="",
            text="",
            words=[],
            raw_json={},
            success=False,
            error=str(e)
        )
