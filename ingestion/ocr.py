"""OCR fallback for image-only PDFs using Ollama vision models."""

import base64
import logging
from pathlib import Path

import fitz
import httpx

logger = logging.getLogger(__name__)

# Phrases that indicate the vision model failed or refused to extract text.
# Responses matching any of these are treated as empty (not usable for RAG).
_FAILURE_PHRASES = [
    "too small to be legible",
    "too small to read",
    "unable to extract",
    "cannot extract",
    "can't extract",
    "unable to read",
    "cannot read",
    "can't read",
    "no text",
    "no readable text",
    "does not contain any text",
    "there is no text",
    "feel free to let me know",
    "if you need help with anything",
    "i cannot provide",
    "i'm unable to",
    "i am unable to",
]


def _is_ocr_failure(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _FAILURE_PHRASES)


_OCR_PROMPT_BASE = (
    "Extract ALL text from this document image. "
    "Preserve the original structure (paragraphs, lists, tables) as closely as possible. "
    "Output ONLY the extracted text, nothing else."
)


def _build_ocr_prompt(language: str | None = None) -> str:
    if language:
        return f"{_OCR_PROMPT_BASE} The document is in {language}. Keep ALL text in {language}."
    return f"{_OCR_PROMPT_BASE} Keep the text in its original language."


def ocr_pdf_pages(filepath: Path, base_url: str, model: str, language: str | None = None) -> str:
    """Extract text from a PDF by sending each page as an image to an Ollama vision model.

    Args:
        filepath: Path to the PDF file.
        base_url: Ollama API base URL (e.g. http://localhost:11434).
        model: Ollama vision model name (e.g. llava:7b).

    Returns:
        Concatenated extracted text from all pages.
    """
    prompt = _build_ocr_prompt(language)
    if language:
        logger.info("OCR language hint: %s", language)

    doc = fitz.open(filepath)
    page_texts = []

    try:
        # Vision models can be slow, especially on first load.
        # Use a generous timeout: 30s to connect, 5min to generate per page.
        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout) as client:
            for page_num, page in enumerate(doc):
                pixmap = page.get_pixmap(dpi=200)
                image_bytes = pixmap.tobytes("png")
                image_b64 = base64.b64encode(image_bytes).decode("ascii")

                payload = {
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                }

                logger.info("OCR page %d/%d of %s", page_num + 1, len(doc), filepath.name)

                resp = client.post(f"{base_url}/api/generate", json=payload)
                resp.raise_for_status()

                text = resp.json().get("response", "").strip()
                if text:
                    if _is_ocr_failure(text):
                        logger.warning(
                            "OCR model returned a failure response for page %d of %s: %r",
                            page_num + 1,
                            filepath.name,
                            text[:120],
                        )
                    else:
                        page_texts.append(text)
    finally:
        doc.close()

    return "\n\n".join(page_texts)
