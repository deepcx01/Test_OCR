"""
OCR Similarity Evaluation Script:
Compares OCR output text against Ground Truth (GT) text to measure content correctness.
"""

import re
import os
from datetime import datetime
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SimilarityResult:
    """Container for similarity evaluation results."""
    similarity_score: float
    total_gt_words: int
    correct_words: int
    incorrect_words: int
    missing_words: List[str]
    incorrect_words_list: List[Tuple[str, str]]  # (expected, found_or_missing)


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    """
    # Remove HTML tags (anything between < and >)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove HTML entities like &nbsp; &amp; etc.
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    - Remove HTML tags (if OCR captured any)
    - Convert to lowercase
    - Preserve important characters: $, #, @ (currency, hashtags, emails)
    - Collapse spaces around joining punctuation (keeps compound values together)
    - Remove joining punctuation (comma, hyphen, slash, period) without splitting
    - Replace other punctuation (colon, etc.) with spaces
    - Normalize all whitespace to single spaces
    - Strip leading/trailing whitespace
    
    This ensures compound values like "12,450", "INV-10234", "15/09/2024" are 
    treated as single tokens, even if OCR adds spaces like "12 , 450".
    Important prefixes like "$500", "#123", "@user" are preserved.
    """

    text = strip_html_tags(text)
    text = text.lower()
    joining_punct = [',', '/']    
    preserve_chars = ['$', '#', '@', '₹', '€', '£', '¥', '%']
    
    for p in joining_punct:
        text = re.sub(r'\s*' + re.escape(p) + r'\s*', p, text)
    
    for p in joining_punct:
        text = text.replace(p, '')
    
    for char in string.punctuation:
        if char not in joining_punct and char not in preserve_chars:
            text = text.replace(char, ' ')
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize normalized text into words.
    
    Returns a list of words (tokens) from the normalized text.
    """
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def compute_similarity(gt_text: str, ocr_text: str) -> SimilarityResult:
    """
    Compute similarity score between Ground Truth and OCR output.

    Formula: similarity_score = (correct_gt_words / total_gt_words) * 100

    Args:
        gt_text: Ground Truth text
        ocr_text: OCR output text    
    Returns:
        SimilarityResult containing score and detailed breakdown
    """
    gt_words = tokenize(gt_text)
    ocr_words = tokenize(ocr_text)
    
    if not gt_words:
        return SimilarityResult(
            similarity_score=100.0 if not ocr_words else 0.0,
            total_gt_words=0,
            correct_words=0,
            incorrect_words=0,
            missing_words=[],
            incorrect_words_list=[]
        )
    
    # Create word frequency counters
    gt_counter = Counter(gt_words)
    ocr_counter = Counter(ocr_words)

    correct_count = 0
    missing_words = []
    incorrect_words_list = []
    
    # For each unique word in GT, check how many are matched in OCR
    for word, gt_count in gt_counter.items():
        ocr_count = ocr_counter.get(word, 0)
        
        # Number of correct matches is the minimum of GT and OCR counts
        matched = min(gt_count, ocr_count)
        correct_count += matched
        
        # Missing occurrences
        missing = gt_count - matched
        if missing > 0:
            for _ in range(missing):
                missing_words.append(word)
                incorrect_words_list.append((word, "MISSING"))
    
  
    total_gt_words = len(gt_words)
    incorrect_count = total_gt_words - correct_count
    

    similarity_score = (correct_count / total_gt_words) * 100 if total_gt_words > 0 else 100.0
    
    return SimilarityResult(
        similarity_score=round(similarity_score, 3),
        total_gt_words=total_gt_words,
        correct_words=correct_count,
        incorrect_words=incorrect_count,
        missing_words=missing_words,
        incorrect_words_list=incorrect_words_list
    )

def load_text_file(filepath: str) -> str:
    """Load text from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def print_result(result: SimilarityResult, gt_file: str = None, ocr_file: str = None):
    """Print formatted similarity result."""
    print("=" * 60)
    print("OCR SIMILARITY EVALUATION REPORT")
    print("=" * 60)
    
    if gt_file:
        print(f"Ground Truth File : {gt_file}")
    if ocr_file:
        print(f"OCR Output File   : {ocr_file}")
    
    print("-" * 60)
    print(f"Similarity Score  : {result.similarity_score:.2f}%")
    print(f"Total GT Words    : {result.total_gt_words}")
    print(f"Correct Words     : {result.correct_words}")
    print(f"Incorrect Words   : {result.incorrect_words}")
    print("-" * 60)
    
    if result.missing_words:
        # Group missing words with counts for cleaner output
        missing_counter = Counter(result.missing_words)
        print("Missing Words:")
        for word, count in missing_counter.items():
            if count > 1:
                print(f"  - '{word}' (x{count})")
            else:
                print(f"  - '{word}'")
    else:
        print("Missing Words: None")
    
    print("=" * 60)

def save_result_text(
    result: SimilarityResult,
    output_dir: str,
    output_filename: str,
    gt_file: str = None,
    ocr_file: str = None
):
    """
    Save OCR similarity result as a TEXT report.
    Handles relative, absolute, and nested output paths safely.
    """

    # If user passed a path (e.g., outputs/output1.txt or /abs/path/output1.txt)
    if os.path.isabs(output_filename) or os.path.dirname(output_filename):
        output_path = output_filename
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("OCR SIMILARITY EVALUATION REPORT\n")
        f.write("=" * 60 + "\n")

        if gt_file:
            f.write(f"Ground Truth File : {gt_file}\n")
        if ocr_file:
            f.write(f"OCR Output File   : {ocr_file}\n")

        f.write("-" * 60 + "\n")
        f.write(f"Similarity Score : {result.similarity_score:.2f}%\n")
        f.write(f"Total GT Words   : {result.total_gt_words}\n")
        f.write(f"Correct Words    : {result.correct_words}\n")
        f.write(f"Incorrect Words  : {result.incorrect_words}\n")
        f.write("-" * 60 + "\n")

        if result.missing_words:
            f.write("Missing Words:\n")
            counter = Counter(result.missing_words)
            for word, count in counter.items():
                if count > 1:
                    f.write(f"  - '{word}' (x{count})\n")
                else:
                    f.write(f"  - '{word}'\n")
        else:
            f.write("Missing Words: None\n")

        f.write("=" * 60 + "\n")

    print(f"\n Text report saved to: {output_path}")


def evaluate_files(
    gt_filepath: str,
    ocr_filepath: str,
    output_filename: str
) -> SimilarityResult:
    gt_text = load_text_file(gt_filepath)
    ocr_text = load_text_file(ocr_filepath)

    result = compute_similarity(gt_text, ocr_text)
    print_result(result, gt_filepath, ocr_filepath)

    save_result_text(
        result,
        output_dir="outputs",
        output_filename=output_filename,
        gt_file=gt_filepath,
        ocr_file=ocr_filepath
    )

    return result


def compute_ocr_similarity(gt_text: str, ocr_text: str) -> float:
    """
    Returns:
        Float between 0.0 and 1.0 representing similarity
    """
    result = compute_similarity(gt_text, ocr_text)
    return result.similarity_score / 100.0  # Convert from percentage to 0-1 scale


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        # Usage: python ocr_similarity.py <gt_file> <ocr_file> <output_filename>
        gt_file = sys.argv[1]
        ocr_file = sys.argv[2]
        output_filename = sys.argv[3]

        evaluate_files(gt_file, ocr_file, output_filename)

    else:
        print("Usage:")
        print("  python ocr_similarity.py <gt_file> <ocr_file> <output_filename>")
        print("Example:")
        print("  python ocr_similarity.py gt.txt ocr.txt output1.txt")
        sys.exit(1)
