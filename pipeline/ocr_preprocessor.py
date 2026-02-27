import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytesseract
import docx
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader

# Configuration
INPUT_DIR = Path("./pipeline/data")
OUTPUT_DIR = Path("./pipeline/output")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"}
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# Thresholds
MIN_CHARS_PER_PAGE = 50       # Ignore page numbers/headers when counting "text pages"
DIGITAL_CONFIDENCE_THRESHOLD = 0.85 # If 85% of pages have text, skip OCR.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_docx_text(docx_path: Path) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para.text in doc.paragraphs])

def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, config=TESSERACT_CONFIG)

def process_pdf(pdf_path: Path) -> str:
    """
    Scans the ENTIRE pdf for text content first.
    If > 85% of pages have selectable text, return that.
    Otherwise, fallback to OCR for the whole document.
    """
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            return ""

        extracted_pages = []
        valid_text_pages = 0

        # Pass 1: Quick Digital Extraction (Milliseconds)
        for page in reader.pages:
            text = page.extract_text() or ""
            extracted_pages.append(text)
            
            # If page has substantive text, count it as a "digital page"
            if len(text.strip()) > MIN_CHARS_PER_PAGE:
                valid_text_pages += 1

        # Pass 2: The Decision
        confidence = valid_text_pages / total_pages

        if confidence > DIGITAL_CONFIDENCE_THRESHOLD:
            logging.info(f"Skipping OCR (Digital Confidence: {confidence:.0%}): {pdf_path.name}")
            return "\n\n".join(extracted_pages)
        else:
            logging.info(f"Low text detected ({confidence:.0%}). Running OCR: {pdf_path.name}")
            # Proceed to fall through to the OCR logic below...

    except Exception as e:
        logging.warning(f"Digital check failed for {pdf_path.name}: {e}")

    # Pass 3: OCR Fallback (Slow)
    # This runs if confidence is low OR if pypdf failed
    pages = convert_from_path(str(pdf_path), dpi=300)
    texts = [ocr_image(page) for page in pages]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)

def process_file(file_path: Path) -> None:
    try:
        output_path = OUTPUT_DIR / (file_path.stem + ".txt")
        if output_path.exists():
            return

        text = ""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text = process_pdf(file_path)
        elif suffix == ".docx":
            text = extract_docx_text(file_path)
        else:
            logging.info(f"Processing Image: {file_path.name}")
            with Image.open(file_path) as img:
                text = ocr_image(img)

        output_path.write_text(text, encoding="utf-8")
        
    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {e}")

def main() -> None:
    if not INPUT_DIR.exists():
        logging.error(f"Input directory {INPUT_DIR} does not exist.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files_to_process = [
        p for p in INPUT_DIR.iterdir() 
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files_to_process:
        logging.warning("No files found to process.")
        return

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(process_file, files_to_process)

if __name__ == "__main__":
    main()