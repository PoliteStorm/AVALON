import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
from io import BytesIO

st.set_page_config(page_title="PDF to CSV/Excel/JSON Converter", layout="centered")
st.title("PDF Table Extractor: Research-Grade PDF to CSV/Excel/JSON 📝→📊")

with st.expander("ℹ️ How to use this tool (click to expand)"):
    st.markdown("""
    - **Upload one or more PDF files** containing tables (text-based or scanned).
    - **Choose extraction engine:** Camelot (default), Tabula-py, or OCR (for scanned PDFs).
    - **Set options:** Page range, table area, output format (CSV, Excel, JSON).
    - **Preview tables** and select which to export.
    - **Download** individual tables or all as a ZIP.
    - For best results, use high-quality, well-structured PDFs. For images/scans, enable OCR.
    """)

# Extraction engine and options
col1, col2 = st.columns(2)
with col1:
    engine = st.radio("Extraction engine", ["camelot", "tabula-py", "ocr (tesseract)"])
with col2:
    output_format = st.selectbox("Output format", ["CSV", "Excel", "JSON"])

page_range = st.text_input("Page range (e.g. 1,3-5 or all)", value="all")
table_area = st.text_input("Table area (optional, e.g. x1,y1,x2,y2)", value="")

ocr_lang = "eng"
if engine == "ocr (tesseract)":
    ocr_lang = st.text_input("OCR language (Tesseract code, e.g. 'eng')", value="eng")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Helper for OCR extraction
def extract_tables_ocr(pdf_path, lang="eng", page_range="all"):
    import pdf2image
    import pytesseract
    import pdfplumber
    tables = []
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=300, first_page=None, last_page=None)
        for i, img in enumerate(images):
            if page_range != "all":
                # Only process selected pages
                from pdfplumber.utils import parse_page_numbers
                wanted = parse_page_numbers(page_range, len(images))
                if i+1 not in wanted:
                    continue
            text = pytesseract.image_to_string(img, lang=lang)
            # Try to extract tables from OCR text using pandas
            dfs = []
            try:
                import io
                dfs = pd.read_html(io.StringIO(text))
            except Exception:
                pass
            # Fallback: try pdfplumber table extraction
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[i]
                plumber_tables = page.extract_tables()
                for t in plumber_tables:
                    dfs.append(pd.DataFrame(t))
            tables.extend([(i+1, df) for df in dfs])
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
    return tables

# Helper for batch zipping
def zip_tables(tables, output_format):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for meta, df in tables:
            page, idx = meta
            if output_format == "CSV":
                data = df.to_csv(index=False).encode('utf-8')
                ext = "csv"
            elif output_format == "Excel":
                from io import BytesIO
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)
                data = excel_buffer.getvalue()
                ext = "xlsx"
            else:
                data = df.to_json(orient="records").encode('utf-8')
                ext = "json"
            zf.writestr(f"table_page{page}_table{idx+1}.{ext}", data)
    zip_buffer.seek(0)
    return zip_buffer

if uploaded_files:
    all_tables = []
    extraction_logs = []
    for file_idx, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name
        st.subheader(f"File: {uploaded_file.name}")
        tables = []
        error = None
        try:
            if engine == "camelot":
                import camelot
                kwargs = {"pages": page_range}
                if table_area.strip():
                    kwargs["table_areas"] = [table_area]
                tables_c = camelot.read_pdf(tmp_pdf_path, **kwargs)
                if tables_c.n == 0:
                    error = "No tables found with Camelot. Try Tabula-py, OCR, or check your PDF."
                else:
                    for i, table in enumerate(tables_c):
                        all_tables.append(((file_idx+1, i), table.df))
                        tables.append((i, table.df))
            elif engine == "tabula-py":
                import tabula
                kwargs = {"pages": page_range, "multiple_tables": True}
                if table_area.strip():
                    kwargs["area"] = [float(x) for x in table_area.split(",")]
                tables_t = tabula.read_pdf(tmp_pdf_path, **kwargs)
                if not tables_t:
                    error = "No tables found with Tabula-py. Try Camelot, OCR, or check your PDF."
                else:
                    for i, df in enumerate(tables_t):
                        all_tables.append(((file_idx+1, i), df))
                        tables.append((i, df))
            else:
                tables_ocr = extract_tables_ocr(tmp_pdf_path, lang=ocr_lang, page_range=page_range)
                if not tables_ocr:
                    error = "No tables found with OCR. Try Camelot or Tabula-py, or check your PDF."
                else:
                    for i, (page, df) in enumerate(tables_ocr):
                        all_tables.append(((file_idx+1, i), df))
                        tables.append((i, df))
        except Exception as e:
            error = str(e)
        if error:
            st.error(f"Extraction failed: {error}")
            extraction_logs.append(f"{uploaded_file.name}: {error}")
        else:
            st.success(f"Found {len(tables)} table(s) in {uploaded_file.name}.")
            for idx, df in tables:
                st.write(f"Table {idx+1} (shape: {df.shape})")
                st.dataframe(df.head(10))
        os.remove(tmp_pdf_path)
    if all_tables:
        st.markdown("---")
        st.subheader("Export Options")
        table_choices = [f"File {meta[0]} Table {meta[1]+1} (shape: {df.shape})" for meta, df in all_tables]
        selected = st.multiselect("Select tables to export", table_choices, default=table_choices)
        selected_tables = [all_tables[i] for i, name in enumerate(table_choices) if name in selected]
        if len(selected_tables) > 1:
            zip_buffer = zip_tables(selected_tables, output_format)
            st.download_button(
                label=f"Download ALL selected tables as ZIP ({output_format})",
                data=zip_buffer,
                file_name=f"tables_export.zip",
                mime="application/zip"
            )
        for i, (meta, df) in enumerate(selected_tables):
            page, idx = meta
            if output_format == "CSV":
                data = df.to_csv(index=False).encode('utf-8')
                ext = "csv"
            elif output_format == "Excel":
                from io import BytesIO
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)
                data = excel_buffer.getvalue()
                ext = "xlsx"
            else:
                data = df.to_json(orient="records").encode('utf-8')
                ext = "json"
            st.download_button(
                label=f"Download Table {i+1} as {output_format}",
                data=data,
                file_name=f"table_file{page}_table{idx+1}.{ext}",
                mime="text/csv" if ext=="csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if ext=="xlsx" else "application/json"
            )
    if extraction_logs:
        with st.expander("Show extraction logs/errors"):
            for log in extraction_logs:
                st.write(log)
else:
    st.info("Please upload one or more PDF files to begin.") 