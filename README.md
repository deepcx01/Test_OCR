# OCR Runner Pipeline

A unified OCR pipeline that runs on GitHub Actions, routing requests to Doctr/Surya endpoints or running PaddleOCR locally.

## Directory Structure
```
ocr_benchmark/
├── ocr_runner/           # Core OCR package
│   ├── ocr_router.py     # Model routing logic
│   ├── paddle_local.py   # Local PaddleOCR
│   ├── text_processor.py # Custom text utilities
│   └── similarity_logic.py
├── scripts/              # CLI tools
│   ├── run_ocr_cli.py
│   ├── compare_outputs.py
│   ├── compare_web_ui.py # Web comparison dashboard
│   └── download_from_r2.py
├── .github/workflows/    # CI/CD
│   └── ocr_benchmark.yml
└── requirements.txt
```

## Quick Start

### Run OCR locally
```bash
cd ocr_benchmark

# Install dependencies
pip install -r requirements.txt

# Run Doctr OCR
python scripts/run_ocr_cli.py --image ../Input/image.jpg --model doctr --output output.json

# Save for web UI (saves to ../Doctr/custom_text/ct_1.txt)
python scripts/run_ocr_cli.py --image ../Input/image.jpg --model doctr --web-format 1
```

### Compare Outputs
```bash
# Compare OCR output to ground truth
python scripts/compare_outputs.py --reference gt.txt --compare ocr_output.txt

# Compare two model outputs
python scripts/compare_outputs.py --source1 doctr.txt --source2 surya.txt --json
```

### Web Comparison Dashboard
```bash
python scripts/compare_web_ui.py
# Opens browser at http://localhost:8084
```

### Download from R2
```bash
python scripts/download_from_r2.py --r2-path r2://bucket/images/doc.jpg --output inputs/
```

## GitHub Actions

Trigger the `OCR Benchmark` workflow manually from the Actions tab:

1. **image_source**: Path to image (local, URL, or `r2://bucket/path`)
2. **model**: Select `doctr`, `surya`, or `paddle`
3. **gt_source** (optional): Ground truth file for comparison
4. **compare_model** (optional): Second model to compare

### Required Secrets
- `RUNPOD_API_KEY`: API key for Doctr/Surya endpoint
- `OCR_ENDPOINT_URL`: RunPod endpoint URL
- `R2_ACCESS_KEY`, `R2_SECRET_KEY`, `R2_ENDPOINT`: For R2 downloads
