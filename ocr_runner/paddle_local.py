"""
 PaddleOCR Runner
"""

from typing import Optional
from PIL import Image
from loguru import logger

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")


# Lazy initialization of PaddleOCR
_ocr_instance: Optional['PaddleOCR'] = None


def get_paddle_instance() -> 'PaddleOCR':
    """Get or create PaddleOCR instance (singleton)."""
    global _ocr_instance
    if _ocr_instance is None:
        if not PADDLE_AVAILABLE:
            raise ImportError("PaddleOCR is not installed")
        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )
    return _ocr_instance


def create_custom_text_from_paddle(ocr_result: list) -> str:
    """
    Create custom_text format from PaddleOCR results.
    Groups text by approximate line position.
    
    Args:
        ocr_result: Raw PaddleOCR output
    
    Returns:
        Formatted custom_text string
    """
    if not ocr_result or not ocr_result[0]:
        return ""
    
    # Extract text with bounding box info
    lines_data = []
    for item in ocr_result[0]:
        bbox = item[0]
        text = item[1][0]
        confidence = item[1][1]
        
        # Use top-left Y coordinate for line grouping
        y_pos = min(point[1] for point in bbox)
        x_pos = min(point[0] for point in bbox)
        
        lines_data.append({
            "text": text,
            "y": y_pos,
            "x": x_pos,
            "confidence": confidence
        })
    
    # Sort by Y position, then X position
    lines_data.sort(key=lambda item: (item["y"], item["x"]))
    
    # Group into lines (items within 15px Y distance are same line)
    grouped_lines = []
    current_line = []
    current_y = None
    y_threshold = 15
    
    for item in lines_data:
        if current_y is None or abs(item["y"] - current_y) <= y_threshold:
            current_line.append(item)
            if current_y is None:
                current_y = item["y"]
        else:
            if current_line:
                # Sort line items by X position
                current_line.sort(key=lambda x: x["x"])
                grouped_lines.append(current_line)
            current_line = [item]
            current_y = item["y"]
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda x: x["x"])
        grouped_lines.append(current_line)
    
    # Build custom_text
    text_lines = []
    for line in grouped_lines:
        line_text = " ".join(item["text"] for item in line)
        text_lines.append(line_text)
    
    return "\n".join(text_lines)


def run_paddle_ocr(image_path: str, image: Optional[Image.Image] = None):
    """
    Run PaddleOCR on an image.
    
    Args:
        image_path: Path to image file
        image: Optional pre-loaded PIL Image
    
    Returns:
        OCRResult with extracted text
    """
    # Import here to avoid circular imports
    from .ocr_router import OCRResult
    
    try:
        ocr = get_paddle_instance()
        
        # Run OCR using new predict() API (PaddleOCR v3.4+)
        logger.info(f"Running PaddleOCR on: {image_path}")
        result = ocr.predict(image_path)
        
        # Parse new format result - it returns a generator or list of results
        all_texts = []
        words = []
        words_bboxes = []
        
        for page_result in result:
            if hasattr(page_result, 'rec_texts') and page_result.rec_texts:
                # New format with rec_texts and rec_boxes
                for i, text in enumerate(page_result.rec_texts):
                    words.append(text)
                    all_texts.append(text)
                    if hasattr(page_result, 'rec_polys') and i < len(page_result.rec_polys):
                        poly = page_result.rec_polys[i]
                        # Convert polygon to bbox [x1, y1, x2, y2]
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        words_bboxes.append([
                            int(min(x_coords)),
                            int(min(y_coords)),
                            int(max(x_coords)),
                            int(max(y_coords))
                        ])
            elif hasattr(page_result, 'text') and page_result.text:
                # Alternative format
                words.append(page_result.text)
                all_texts.append(page_result.text)
        
        # Create custom_text - join with newlines for line separation
        custom_text = "\n".join(all_texts) if all_texts else ""
        
        # Create full text (single line)
        full_text = " ".join(words)
        
        return OCRResult(
            model="paddle",
            custom_text=custom_text,
            text=full_text,
            words=words,
            raw_json={
                "words_bboxes": words_bboxes,
            },
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
