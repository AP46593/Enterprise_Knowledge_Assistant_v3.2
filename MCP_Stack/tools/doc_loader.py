"""
Document Loader Module.

Loads documents of various formats and extracts text content.
Supported formats: PDF, DOCX, TXT, CSV, XLSX, XML, PNG, JPG, JPEG, TIFF.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Supported Formats
# =============================================================================

# Maps file extensions to their handler type for dispatch
SUPPORTED_EXTENSIONS = {
    "pdf": "pdf",
    "docx": "docx",
    "txt": "txt",
    "csv": "csv",
    "xlsx": "xlsx",
    "xml": "xml",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "tiff": "image",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DocumentMetadata:
    """
    Metadata attached to a loaded document.

    Attributes:
        file_path: Absolute path to the source file.
        file_name: Filename with extension.
        file_type: Lowercase file extension without dot.
        file_size: File size in bytes.
        modified_date: Last modification timestamp.
        title: Human-readable title (typically the filename stem).
    """

    file_path: str
    file_name: str
    file_type: str
    file_size: int
    modified_date: datetime
    title: str


@dataclass
class DocumentLoadResult:
    """
    Result from document loading.

    Attributes:
        text: Extracted text content (empty string on failure).
        metadata: Document metadata.
        success: Whether text extraction succeeded.
        error: Human-readable error message (None on success).
    """

    text: str
    metadata: DocumentMetadata
    success: bool
    error: Optional[str] = None


# =============================================================================
# Public API
# =============================================================================


def load_document(file_path: str) -> DocumentLoadResult:
    """
    Load a document and extract text content.

    Dispatches to the appropriate format handler based on file extension.
    Returns a DocumentLoadResult with text content on success, or an error
    description on failure. Never raises unhandled exceptions.

    Args:
        file_path: Path to the document file.

    Returns:
        DocumentLoadResult with extracted text or error information.
    """
    path = Path(file_path)

    # Build metadata (even for error cases, provide what we can)
    try:
        stat = path.stat()
        metadata = DocumentMetadata(
            file_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lstrip(".").lower(),
            file_size=stat.st_size,
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            title=path.stem,
        )
    except (OSError, FileNotFoundError) as e:
        # Can't even stat the file
        metadata = DocumentMetadata(
            file_path=str(path),
            file_name=path.name,
            file_type=path.suffix.lstrip(".").lower(),
            file_size=0,
            modified_date=datetime.now(),
            title=path.stem,
        )
        error_msg = f"Unable to access file: {e}"
        logger.error(error_msg)
        return DocumentLoadResult(text="", metadata=metadata, success=False, error=error_msg)

    # Check if the file exists
    if not path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        return DocumentLoadResult(text="", metadata=metadata, success=False, error=error_msg)

    # Determine the file extension and dispatch to the appropriate handler
    extension = path.suffix.lstrip(".").lower()
    handler_type = SUPPORTED_EXTENSIONS.get(extension)

    if handler_type is None:
        error_msg = f"File type .{extension} is not supported. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}"
        logger.warning(error_msg)
        return DocumentLoadResult(text="", metadata=metadata, success=False, error=error_msg)

    # Dispatch to the format-specific handler function
    try:
        handler_map = {
            "pdf": _load_pdf,
            "docx": _load_docx,
            "txt": _load_txt,
            "csv": _load_csv,
            "xlsx": _load_xlsx,
            "xml": _load_xml,
            "image": _load_image,
        }
        handler = handler_map[handler_type]
        text = handler(str(path))

        if not text or not text.strip():
            logger.warning(f"No text content extracted from: {file_path}")

        return DocumentLoadResult(text=text, metadata=metadata, success=True)

    except Exception as e:
        error_msg = f"Unable to read file '{path.name}': {type(e).__name__}: {e}"
        logger.error(error_msg)
        return DocumentLoadResult(text="", metadata=metadata, success=False, error=error_msg)


# =============================================================================
# Format-Specific Handlers
# =============================================================================


def _load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pypdf with pdfplumber fallback.

    Extracts text page-by-page. If pypdf yields no text on a page,
    falls back to pdfplumber for that page.

    Requirements: 1.1
    """
    import pypdf

    text_parts = []

    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
            else:
                # Fallback to pdfplumber for pages with no extracted text
                fallback_text = _load_pdf_page_pdfplumber(file_path, page_num)
                if fallback_text:
                    text_parts.append(fallback_text)

    return "\n\n".join(text_parts)


def _load_pdf_page_pdfplumber(file_path: str, page_num: int) -> str:
    """
    Extract text from a specific PDF page using pdfplumber.

    Used as a fallback when pypdf fails to extract text from a page.

    Args:
        file_path: Path to the PDF file.
        page_num: Zero-based page number to extract.

    Returns:
        Extracted text string, or empty string if extraction fails.
    """
    import pdfplumber

    with pdfplumber.open(file_path) as pdf:
        if page_num < len(pdf.pages):
            page = pdf.pages[page_num]
            text = page.extract_text()
            return text if text else ""
    return ""


def _load_docx(file_path: str) -> str:
    """
    Extract text from a Word DOCX file using python-docx.

    Extracts text from all paragraphs in order.

    Requirements: 1.2
    """
    import docx

    doc = docx.Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def _load_txt(file_path: str) -> str:
    """
    Read plain text content with encoding detection.

    Tries UTF-8 first, then falls back to latin-1, then to
    reading with errors='replace'.

    Requirements: 1.3
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Final fallback: read with replacement characters
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_csv(file_path: str) -> str:
    """
    Parse CSV data and return as structured text.

    Uses pandas to read the CSV and converts the DataFrame to a
    readable string representation.

    Requirements: 1.4
    """
    import pandas as pd

    df = pd.read_csv(file_path)
    # Convert to a readable format: header row + data rows
    lines = []
    lines.append("Columns: " + ", ".join(str(col) for col in df.columns))
    lines.append(f"Total rows: {len(df)}")
    lines.append("")
    lines.append(df.to_string(index=False))
    return "\n".join(lines)


def _load_xlsx(file_path: str) -> str:
    """
    Extract spreadsheet content sheet-by-sheet using openpyxl via pandas.

    Each sheet is extracted separately with its name as a header.

    Requirements: 1.5
    """
    import pandas as pd

    xlsx = pd.ExcelFile(file_path, engine="openpyxl")
    parts = []

    for sheet_name in xlsx.sheet_names:
        df = xlsx.parse(sheet_name)
        sheet_text = []
        sheet_text.append(f"--- Sheet: {sheet_name} ---")
        sheet_text.append("Columns: " + ", ".join(str(col) for col in df.columns))
        sheet_text.append(f"Total rows: {len(df)}")
        sheet_text.append("")
        sheet_text.append(df.to_string(index=False))
        parts.append("\n".join(sheet_text))

    return "\n\n".join(parts)


def _load_xml(file_path: str) -> str:
    """
    Parse XML content and return as readable text via element traversal.

    Uses xml.etree.ElementTree for parsing, falling back to lxml if
    the standard library parser fails.

    Requirements: 1.6
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(file_path)
    except ET.ParseError:
        # Fallback to lxml which is more tolerant
        from lxml import etree

        tree = etree.parse(file_path)
        root = tree.getroot()
        return _extract_xml_text_lxml(root)

    root = tree.getroot()
    return _extract_xml_text(root)


# =============================================================================
# XML Helpers
# =============================================================================


def _extract_xml_text(element) -> str:
    """
    Recursively extract text from a standard library ElementTree element.

    Formats output as "tag: text_content" for each element with text.

    Args:
        element: An xml.etree.ElementTree Element.

    Returns:
        Newline-joined text from the element tree.
    """
    import xml.etree.ElementTree as ET

    parts = []

    # Get the tag name without namespace
    tag = element.tag
    if "}" in tag:
        tag = tag.split("}")[1]

    # Add element text
    if element.text and element.text.strip():
        parts.append(f"{tag}: {element.text.strip()}")

    # Recurse into children
    for child in element:
        child_text = _extract_xml_text(child)
        if child_text:
            parts.append(child_text)

    # Add tail text
    if element.tail and element.tail.strip():
        parts.append(element.tail.strip())

    return "\n".join(parts)


def _extract_xml_text_lxml(element) -> str:
    """
    Recursively extract text from an lxml element.

    Same logic as _extract_xml_text but handles lxml's element interface.

    Args:
        element: An lxml.etree Element.

    Returns:
        Newline-joined text from the element tree.
    """
    parts = []

    # Get the tag name without namespace
    tag = element.tag
    if isinstance(tag, str) and "}" in tag:
        tag = tag.split("}")[1]

    # Add element text
    if element.text and element.text.strip():
        parts.append(f"{tag}: {element.text.strip()}")

    # Recurse into children
    for child in element:
        child_text = _extract_xml_text_lxml(child)
        if child_text:
            parts.append(child_text)

    # Add tail text
    if element.tail and element.tail.strip():
        parts.append(element.tail.strip())

    return "\n".join(parts)


def _load_image(file_path: str) -> str:
    """
    Extract text from image files using OCR (pytesseract + Pillow).

    Falls back to an Ollama vision model (e.g., llava) when pytesseract
    is not available or fails.

    Supports PNG, JPG, JPEG, TIFF formats.

    Requirements: 1.7
    """
    from PIL import Image

    image = Image.open(file_path)

    # Try pytesseract first
    try:
        import pytesseract

        text = pytesseract.image_to_string(image)
        if text and text.strip():
            return text.strip()
        # If tesseract returned empty, fall through to vision model
        logger.info("Tesseract returned empty text for %s, trying vision model fallback", file_path)
    except ImportError:
        logger.info("pytesseract not installed, using vision model fallback for %s", file_path)
    except Exception as e:
        logger.warning("Tesseract OCR failed for %s: %s. Trying vision model fallback.", file_path, e)

    # Fallback: use Ollama vision model
    from MCP_Stack.server_config import VISION_MODEL

    if not VISION_MODEL:
        raise RuntimeError(
            f"Cannot extract text from image '{file_path}': pytesseract unavailable "
            "and VISION_MODEL is not configured. Set VISION_MODEL in server_config.py "
            "or install pytesseract to enable image ingestion."
        )

    return _load_image_with_vision_model(file_path, image)


# =============================================================================
# Vision Model Fallback
# =============================================================================


def _load_image_with_vision_model(file_path: str, image) -> str:
    """
    Extract text from an image using an Ollama vision model (e.g., gemma4).

    Encodes the image as base64 PNG and sends it to the vision model's
    /api/generate endpoint with a text extraction prompt.

    Args:
        file_path: Path to the image file (used for logging).
        image: PIL Image object already loaded from file.

    Returns:
        Extracted text content.

    Raises:
        RuntimeError: If both tesseract and vision model fail.
    """
    import base64
    import io
    import requests

    from MCP_Stack.server_config import OLLAMA_BASE_URL, VISION_MODEL

    # Convert image to base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Call Ollama vision model API
    payload = {
        "model": VISION_MODEL,
        "prompt": (
            "Extract all text from this image. Return only the extracted text content, "
            "preserving the original formatting and layout as much as possible. "
            "Do not add any commentary or explanation."
        ),
        "images": [image_base64],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("response", "").strip()
        if text:
            logger.info("Vision model extracted text from %s (%d chars)", file_path, len(text))
            return text
        else:
            raise ValueError("Vision model returned empty response")
    except Exception as e:
        raise RuntimeError(
            f"Both tesseract and vision model ({VISION_MODEL}) failed for '{file_path}': {e}"
        ) from e
