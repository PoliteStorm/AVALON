# PDF to CSV/Excel/JSON Converter UI (Research-Grade)

A powerful web UI for extracting tables from research PDFs and saving them as CSV, Excel, or JSON files. Built with [Streamlit](https://streamlit.io/), supports [Camelot](https://camelot-py.readthedocs.io/), [Tabula-py](https://tabula-py.readthedocs.io/), and OCR (Tesseract) for scanned/image-based PDFs.

## Features
- Upload one or more PDF files (batch processing)
- Extract tables using Camelot, Tabula-py, or OCR (Tesseract)
- Preview all detected tables with metadata (page, shape, preview)
- Select which tables to export
- Choose output format: CSV, Excel, or JSON
- Download individual tables or all as a ZIP
- Specify page ranges and table areas
- Extraction logs and error reporting
- Clean, modern UI with help/instructions

## Requirements
- Python 3.8+
- Java (required for Tabula-py)
- [Poppler](https://poppler.freedesktop.org/) (required for Camelot)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (required for OCR)

## Installation

1. Clone this directory or copy the files to your machine.
2. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

3. Install Poppler, Java, and Tesseract if not already installed:
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get install poppler-utils default-jre tesseract-ocr
     ```
   - **MacOS (with Homebrew):**
     ```bash
     brew install poppler tesseract
     brew install --cask temurin
     ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the provided local URL in your browser. Upload one or more PDFs, select extraction engine and options, preview tables, and download as needed.

## Notes
- Camelot works best with text-based PDFs (not scanned images).
- Tabula-py requires Java and may work better for some PDFs.
- OCR (Tesseract) is for scanned/image-based PDFs and may be slower.
- For best results, use high-quality, well-structured PDFs.

---

**This tool is standalone and does not affect your main research codebase.** 