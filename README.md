# Lense - Loss Curve Analyzer

A lightweight Python tool to visualize and analyze machine learning training loss curves from JSON/JSONL logs, with built-in smoothing, metrics calculation, and outlier detection.

## Features
- Parses JSON, JSONL, and single-quoted Python dict logs
- Robust smoothing (SciPy or NumPy fallback moving average)
- Calculates key metrics: slope, SNR, best loss, variance, exposure
- Outlier detection using Median Absolute Deviation (MAD)
- Interactive Tkinter GUI with plot export and clipboard support
- CLI mode for headless use

## Dependencies
- Python 3.8+
- Core: `matplotlib`, `numpy`
- Optional: `scipy` (faster smoothing), `pyperclip` (clipboard copy)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### GUI Mode (default)
```bash
python lense.py
```
1. Paste or load JSONL logs into the input box
2. Click "Analyze Now" or press Ctrl+Enter
3. View metrics and interactive loss plot
4. Export metrics to clipboard or Markdown

### CLI Mode
```bash
python lense.py path/to/training_logs.jsonl
```

## Supported Log Format
Logs must contain a `loss` key (required), with optional fields like `epoch`, `learning_rate`, etc.:
```json
{"loss": 0.52, "epoch": 1.0, "step": 100}
{"loss": 0.48, "epoch": 1.1, "step": 110}
```

## Improvements & Fixes (2026-04-26)
- Replaced naive single-quote regex with `ast.literal_eval` for safer log parsing
- Removed unused `calculate_grade` dead code
- Fixed outlier detection to use MAD instead of standard deviation
- Corrected epoch boundary detection to trigger on whole epoch transitions
- Added `requirements.txt` and this README
