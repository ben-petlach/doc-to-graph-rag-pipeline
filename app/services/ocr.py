from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Final, Optional

import docx
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader


logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Final[set[str]] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"}
TESSERACT_CONFIG: Final[str] = r"--oem 3 --psm 6"

MIN_CHARS_PER_PAGE: Final[int] = 50
DIGITAL_CONFIDENCE_THRESHOLD: Final[float] = 0.85


# ---------------------------------------------------------------------------
# Mistral OCR fallback (lazily initialized)
# ---------------------------------------------------------------------------
_mistral_client = None


def _get_mistral_client(api_key: Optional[str] = None):
    """Lazily initialize the Mistral client."""
    global _mistral_client
    if _mistral_client is None:
        try:
            from mistralai import Mistral
            if not api_key:
                raise ValueError("MISTRAL_API_KEY is required for OCR fallback")
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


def mistral_ocr_image(image: Image.Image, api_key: str) -> str:
    """Send an image to Mistral OCR and return the extracted text."""
    client = _get_mistral_client(api_key)
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


def extract_docx_text(docx_path: Path) -> str:
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])


def ocr_image(
    image: Image.Image, 
    ocr_min_confidence: float = 80.0,
    mistral_api_key: Optional[str] = None
) -> str:
    """
    OCR a single image. If Tesseract confidence is below ocr_min_confidence,
    falls back to Mistral OCR.
    """
    text, confidence = ocr_image_with_confidence(image)

    if confidence < ocr_min_confidence and mistral_api_key:
        logger.info(
            f"Low Tesseract confidence ({confidence:.1f}%). "
            f"Falling back to Mistral OCR."
        )
        try:
            text = mistral_ocr_image(image, mistral_api_key)
        except Exception as e:
            logger.error(f"Mistral OCR failed: {e}. Using Tesseract output.")
    elif confidence < ocr_min_confidence:
        logger.warning(
            f"Low Tesseract confidence ({confidence:.1f}%) but no Mistral API key available."
        )
    else:
        logger.debug(f"Tesseract confidence OK ({confidence:.1f}%).")

    return text


def process_pdf(
    pdf_path: Path,
    ocr_min_confidence: float = 80.0,
    mistral_api_key: Optional[str] = None
) -> str:
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
    texts = [ocr_image(page, ocr_min_confidence, mistral_api_key) for page in pages]
    return "\n\n--- PAGE BREAK ---\n\n".join(texts)


def extracted_text_output_path(output_dir: Path, source_file: Path) -> Path:
    return output_dir / f"{source_file.stem}.txt"


def extract_text_from_file(
    filepath: Path,
    *,
    output_dir: Optional[Path] = None,
    write_output: bool = True,
    force: bool = False,
    ocr_min_confidence: float = 80.0,
    mistral_api_key: Optional[str] = None,
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
        text = process_pdf(filepath, ocr_min_confidence, mistral_api_key)
    elif suffix == ".docx":
        text = extract_docx_text(filepath)
    else:
        with Image.open(filepath) as img:
            text = ocr_image(img, ocr_min_confidence, mistral_api_key)

    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")

    return text

