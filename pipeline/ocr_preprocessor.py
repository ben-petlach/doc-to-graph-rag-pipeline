import base64
import io
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytesseract
from pytesseract import Output
import docx
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

from speed_analyzer import tracker

# Configuration
INPUT_DIR = Path("./pipeline/data")
OUTPUT_DIR = Path("./pipeline/output")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"}
TESSERACT_CONFIG = r'--oem 3 --psm 6'

# Thresholds
MIN_CHARS_PER_PAGE = 50       # Ignore page numbers/headers when counting "text pages"
DIGITAL_CONFIDENCE_THRESHOLD = 0.85 # If 85% of pages have text, skip OCR.
OCR_MIN_CONFIDENCE = 80       # Per-word Tesseract confidence threshold (0-100).
                              # If average confidence is below this, fall back to Mistral OCR.

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Mistral OCR fallback (lazily initialized)
# ---------------------------------------------------------------------------
_mistral_client = None

def _get_mistral_client():
    """Lazily initialize the Mistral client."""
    global _mistral_client
    if _mistral_client is None:
        try:
            from mistralai import Mistral
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable is not set")
            _mistral_client = Mistral(api_key=api_key)
        except ImportError:
            raise ImportError(
                "mistralai package is required for OCR fallback. "
                "Install it with: pip install mistralai"
            )
    return _mistral_client


def _pil_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded data URI."""
    buffer = io.BytesIO()
    fmt = "PNG"
    image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def mistral_ocr_image(image: Image.Image) -> str:
    """Send an image to Mistral OCR and return the extracted text."""
    client = _get_mistral_client()
    image_uri = _pil_image_to_base64(image)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": image_uri,
        },
    )

    # Concatenate markdown text from all returned pages
    texts = [page.markdown for page in ocr_response.pages if page.markdown]
    return "\n\n".join(texts)


def mistral_ocr_pdf(pdf_path: Path) -> str:
    """Send a full PDF to Mistral OCR (via base64) and return extracted text."""
    client = _get_mistral_client()

    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    pdf_uri = f"data:application/pdf;base64,{b64}"

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": pdf_uri,
        },
    )

    texts = [page.markdown for page in ocr_response.pages if page.markdown]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)


# ---------------------------------------------------------------------------
# Per-word OCR confidence scoring
# ---------------------------------------------------------------------------

def ocr_image_with_confidence(image: Image.Image) -> tuple[str, float]:
    """
    Run Tesseract OCR and return (extracted_text, mean_confidence).
    Confidence is 0-100 based on per-word scores from image_to_data.
    Words with conf == -1 (failed detection) are excluded from the average.
    """
    data = pytesseract.image_to_data(image, config=TESSERACT_CONFIG, output_type=Output.DICT)
    confidences = [int(c) for c in data["conf"] if int(c) > -1]
    text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)

    if not confidences:
        return text, 0.0

    mean_conf = sum(confidences) / len(confidences)
    return text, mean_conf


# ---------------------------------------------------------------------------
# Core OCR functions
# ---------------------------------------------------------------------------

def extract_docx_text(docx_path: Path) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def ocr_image(image: Image.Image, file_name: str = "image", page: int = 1) -> str:
    """
    OCR a single image. If Tesseract confidence is below OCR_MIN_CONFIDENCE,
    falls back to Mistral OCR.
    """
    tracker.start(file_name, page, "ocr_tesseract")
    text, confidence = ocr_image_with_confidence(image)
    tracker.stop()

    if confidence < OCR_MIN_CONFIDENCE:
        logging.info(
            f"Low Tesseract confidence ({confidence:.1f}%) for {file_name} p{page}. "
            f"Falling back to Mistral OCR."
        )
        tracker.start(file_name, page, "ocr_mistral_fallback")
        try:
            text = mistral_ocr_image(image)
        except Exception as e:
            logging.error(f"Mistral OCR failed for {file_name} p{page}: {e}. Using Tesseract output.")
        tracker.stop()
    else:
        logging.debug(f"Tesseract confidence OK ({confidence:.1f}%) for {file_name} p{page}.")

    return text


def process_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Scans the ENTIRE pdf for text content first.
    If > 85% of pages have selectable text, return that.
    Otherwise, fallback to OCR for the whole document.
    Returns (text, page_count).
    """
    file_name = pdf_path.name

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        
        if total_pages == 0:
            return "", 0

        extracted_pages = []
        valid_text_pages = 0

        # Pass 1: Quick Digital Extraction (Milliseconds)
        for i, page in enumerate(reader.pages, start=1):
            tracker.start(file_name, i, "digital_extract")
            text = page.extract_text() or ""
            tracker.stop()
            extracted_pages.append(text)
            
            # If page has substantive text, count it as a "digital page"
            if len(text.strip()) > MIN_CHARS_PER_PAGE:
                valid_text_pages += 1

        # Pass 2: The Decision
        confidence = valid_text_pages / total_pages

        if confidence > DIGITAL_CONFIDENCE_THRESHOLD:
            logging.info(f"Skipping OCR (Digital Confidence: {confidence:.0%}): {file_name}")
            return "\n\n".join(extracted_pages), total_pages
        else:
            logging.info(f"Low text detected ({confidence:.0%}). Running OCR: {file_name}")
            # Proceed to fall through to the OCR logic below...

    except Exception as e:
        logging.warning(f"Digital check failed for {file_name}: {e}")
        total_pages = 0  # Will be set from image count below

    # Pass 3: OCR Fallback (Slow)
    # This runs if confidence is low OR if pypdf failed
    pages = convert_from_path(str(pdf_path), dpi=300)
    total_pages = total_pages or len(pages)
    texts = [ocr_image(page_img, file_name=file_name, page=i) for i, page_img in enumerate(pages, start=1)]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts), total_pages


def process_file(file_path: Path) -> None:
    try:
        output_path = OUTPUT_DIR / (file_path.stem + ".txt")
        meta_path = OUTPUT_DIR / (file_path.stem + ".meta.json")
        if output_path.exists():
            return

        text = ""
        page_count = 1
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text, page_count = process_pdf(file_path)
        elif suffix == ".docx":
            tracker.start(file_path.name, 1, "docx_extract")
            text = extract_docx_text(file_path)
            tracker.stop()
        else:
            logging.info(f"Processing Image: {file_path.name}")
            with Image.open(file_path) as img:
                text = ocr_image(img, file_name=file_path.name, page=1)

        output_path.write_text(text, encoding="utf-8")

        # Write metadata (page count) for downstream speed analysis
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"file": file_path.name, "page_count": page_count}, f)
        
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

    # Print and save speed report
    print("\n" + tracker.summary())
    tracker.save(OUTPUT_DIR / "speed_report_ocr.csv")


if __name__ == "__main__":
    main()