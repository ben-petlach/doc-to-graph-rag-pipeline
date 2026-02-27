from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Optional

import docx
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Final[set[str]] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"}
TESSERACT_CONFIG: Final[str] = r"--oem 3 --psm 6"

MIN_CHARS_PER_PAGE: Final[int] = 50
DIGITAL_CONFIDENCE_THRESHOLD: Final[float] = 0.85


def extract_docx_text(docx_path: Path) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, config=TESSERACT_CONFIG)


def process_pdf(pdf_path: Path) -> str:
    """
    Scans the PDF for embedded text first. If most pages contain meaningful text,
    return that extraction; otherwise fall back to OCR for the whole document.
    """
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            return ""

        extracted_pages: list[str] = []
        valid_text_pages = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            extracted_pages.append(text)
            if len(text.strip()) > MIN_CHARS_PER_PAGE:
                valid_text_pages += 1

        confidence = valid_text_pages / total_pages
        if confidence > DIGITAL_CONFIDENCE_THRESHOLD:
            logger.info("Skipping OCR (digital PDF detected): %s", pdf_path.name)
            return "\n\n".join(extracted_pages)

        logger.info("Low embedded text detected; running OCR: %s", pdf_path.name)
    except Exception as e:
        logger.warning("Digital check failed for %s: %s", pdf_path.name, e)

    pages = convert_from_path(str(pdf_path), dpi=300)
    texts = [ocr_image(page) for page in pages]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)


def extracted_text_output_path(output_dir: Path, source_file: Path) -> Path:
    return output_dir / f"{source_file.stem}.txt"


def extract_text_from_file(
    filepath: Path,
    *,
    output_dir: Optional[Path] = None,
    write_output: bool = True,
    force: bool = False,
) -> str:
    """
    Extract text from a single file.

    If output_dir/write_output are enabled, also writes `output_dir/{stem}.txt`.
    When `force` is False and the output already exists, returns the existing text.
    """
    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(str(filepath))

    suffix = filepath.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    output_path: Optional[Path] = None
    if output_dir is not None and write_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = extracted_text_output_path(output_dir, filepath)
        if output_path.exists() and not force:
            return output_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        text = process_pdf(filepath)
    elif suffix == ".docx":
        text = extract_docx_text(filepath)
    else:
        with Image.open(filepath) as img:
            text = ocr_image(img)

    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")

    return text

