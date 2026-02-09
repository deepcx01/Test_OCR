"""
Text Processor - Utilities for extracting and creating custom text formats.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any


def extract_custom_text(ocr_json: Union[str, Path, Dict]) -> str:
    """
    Extract custom_text from OCR JSON output.
    
    Args:
        ocr_json: Path to JSON file, or parsed JSON dict
    
    Returns:
        The custom_text string, or empty string if not found
    """
    if isinstance(ocr_json, (str, Path)):
        with open(ocr_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = ocr_json
    
    # Handle different JSON structures
    search_keys = ["custom_text", "custom_texts", "text"]
    
    # Check top-level first
    for key in search_keys:
        if key in data:
            val = data[key]
            return "\n".join(val) if isinstance(val, list) else str(val)
            
    # Check nested in "data" list
    if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
        first_item = data["data"][0]
        for key in search_keys:
            if key in first_item:
                val = first_item[key]
                return "\n".join(val) if isinstance(val, list) else str(val)
    
    return ""


def create_custom_text(raw_text: str) -> str:
    """
    Create custom_text format from raw text.
    Simply cleans and normalizes the text.
    
    Args:
        raw_text: Raw text string
    
    Returns:
        Cleaned custom_text string
    """
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in raw_text.split('\n')]
    # Remove empty lines at start/end but preserve internal structure
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)


def save_custom_text(custom_text: str, output_path: Union[str, Path]) -> None:
    """
    Save custom_text to a .txt file.
    
    Args:
        custom_text: The custom text to save
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(custom_text)


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a file (txt or json).
    
    Args:
        file_path: Path to text or JSON file
    
    Returns:
        Text content
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.json':
        return extract_custom_text(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


def save_for_web_ui(
    custom_text: str,
    model: str,
    doc_number: int,
    base_dir: Union[str, Path] = None
) -> Path:
    """
    Save custom_text in the format expected by the web comparison UI.
    
    Creates files like:
    - Doctr/custom_text/ct_1.txt
    - Surya/custom_text/ct_1.txt
    - PaddleOCR/custom_text/Input_1.txt
    
    Args:
        custom_text: The custom text content to save
        model: Model name ("doctr", "surya", "paddle")
        doc_number: Document number (1, 2, 3, etc.)
        base_dir: Base directory (default: parent of ocr_benchmark, i.e., Abstract files)
    
    Returns:
        Path to saved file
    """
    if base_dir is None:
        # Go up from ocr_runner -> ocr_benchmark -> Abstract files
        base_dir = Path(__file__).parent.parent.parent
    else:
        base_dir = Path(base_dir)
    
    # Model to directory mapping
    model_dirs = {
        "doctr": "Doctr",
        "surya": "Surya", 
        "paddle": "PaddleOCR"
    }
    
    # Model to file prefix mapping
    file_prefixes = {
        "doctr": "ct_",
        "surya": "ct_",
        "paddle": "Input_"
    }
    
    model_lower = model.lower()
    if model_lower not in model_dirs:
        raise ValueError(f"Unknown model: {model}. Expected: doctr, surya, paddle")
    
    # Build output path
    output_dir = base_dir / model_dirs[model_lower] / "custom_text"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{file_prefixes[model_lower]}{doc_number}.txt"
    output_path = output_dir / filename
    
    # Save the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(custom_text)
    
    return output_path

