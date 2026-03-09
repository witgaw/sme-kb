"""OCR fallback for image-only PDFs using vision models (Ollama or OpenRouter)."""

import base64
import logging
from pathlib import Path

import fitz
import httpx
import openai

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
    "unable to provide a translation",
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


def _ocr_page_ollama(
    client: httpx.Client,
    base_url: str,
    model: str,
    image_b64: str,
    prompt: str,
) -> str:
    resp = client.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "images": [image_b64], "stream": False},
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _ocr_page_openrouter(
    client: openai.OpenAI,
    model: str,
    image_b64: str,
    prompt: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


def ocr_pdf_pages(
    filepath: Path,
    model: str,
    language: str | None = None,
    provider: str = "ollama",
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
) -> str:
    """Extract text from a PDF by sending each page as an image to a vision model.

    Args:
        filepath: Path to the PDF file.
        model: Vision model name (e.g. "llava:7b" for Ollama, "google/gemini-flash-1.5" for
            OpenRouter).
        language: Optional language hint for the OCR prompt.
        provider: "ollama" or "openrouter".
        base_url: Ollama API base URL (ignored for OpenRouter).
        api_key: OpenRouter API key (ignored for Ollama).

    Returns:
        Concatenated extracted text from all pages.
    """
    prompt = _build_ocr_prompt(language)
    if language:
        logger.info("OCR language hint: %s", language)

    if provider == "openrouter":
        if not api_key:
            raise ValueError("api_key required for OpenRouter OCR provider")
        openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        http_client = None
    else:
        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        http_client = httpx.Client(timeout=timeout)
        openai_client = None

    doc = fitz.open(filepath)
    page_texts = []

    try:
        for page_num, page in enumerate(doc):
            pixmap = page.get_pixmap(dpi=200)
            image_b64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")

            logger.info("OCR page %d/%d of %s", page_num + 1, len(doc), filepath.name)

            if provider == "openrouter":
                text = _ocr_page_openrouter(openai_client, model, image_b64, prompt)
            else:
                text = _ocr_page_ollama(http_client, base_url, model, image_b64, prompt)

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
        if http_client is not None:
            http_client.close()

    return "\n\n".join(page_texts)
