"""Multi-format document loader."""

import email
import logging
from email.message import Message
from pathlib import Path
from typing import Any

import fitz  # pymupdf
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

from config import get_config
from ingestion.ocr import ocr_pdf_pages

logger = logging.getLogger(__name__)

# PDFs with fewer average chars per page than this are considered sparse/image-only
SPARSE_TEXT_THRESHOLD = 20


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".eml", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".pdf"}

    def __init__(
        self,
        ocr_enabled: bool | None = None,
        ocr_provider: str | None = None,
        ocr_model: str | None = None,
        ocr_output_dir: Path | None = None,
    ):
        config = get_config()
        self._ocr_enabled = ocr_enabled if ocr_enabled is not None else config.ocr_enabled
        self._ocr_provider = ocr_provider or config.ocr_provider
        self._ocr_model = ocr_model or config.ocr_model
        self._ollama_base_url = config.ollama_base_url
        self._openrouter_api_key = config.openrouter_api_key
        self._ocr_output_dir = ocr_output_dir
        self.ocr_language: str | None = None

    def load(self, filepath: Path | str) -> dict[str, Any]:
        """Load document and return content with metadata.

        Args:
            filepath: Path to the document file.

        Returns:
            Dict with 'content' (str) and 'metadata' (dict).

        Raises:
            ValueError: If file type is not supported.
            FileNotFoundError: If file doesn't exist.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        loaders = {
            ".eml": self._load_eml,
            ".docx": self._load_docx,
            ".xlsx": self._load_xlsx,
            ".pptx": self._load_pptx,
            ".md": self._load_text,
            ".txt": self._load_text,
            ".pdf": self._load_pdf,
        }

        loader = loaders.get(suffix)
        if not loader:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader(filepath)

    def _load_eml(self, filepath: Path) -> dict[str, Any]:
        """Load email file with proper multipart handling."""
        with open(filepath, "rb") as f:
            msg = email.message_from_binary_file(f)

        content = self._extract_email_text(msg)

        return {
            "content": content,
            "metadata": {
                "from": str(msg.get("From") or ""),
                "to": str(msg.get("To") or ""),
                "subject": str(msg.get("Subject") or ""),
                "date": str(msg.get("Date") or ""),
            },
        }

    def _extract_email_text(self, msg: Message) -> str:
        """Extract text content from email, handling multipart messages."""
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Get text content
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            parts.append(payload.decode(charset))
                        except (UnicodeDecodeError, LookupError):
                            parts.append(payload.decode("utf-8", errors="replace"))
                elif content_type == "text/html":
                    # Fallback to HTML if no plain text
                    payload = part.get_payload(decode=True)
                    if payload and not parts:
                        charset = part.get_content_charset() or "utf-8"
                        try:
                            html_content = payload.decode(charset)
                        except (UnicodeDecodeError, LookupError):
                            html_content = payload.decode("utf-8", errors="replace")
                        # Simple HTML stripping
                        parts.append(self._strip_html(html_content))

            return "\n\n".join(parts)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    content = payload.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    content = payload.decode("utf-8", errors="replace")

                if msg.get_content_type() == "text/html":
                    content = self._strip_html(content)

                return content
            return str(msg.get_payload())

    def _strip_html(self, html: str) -> str:
        """Simple HTML tag stripping."""
        import re

        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        html = re.sub(r"<[^>]+>", " ", html)
        # Normalize whitespace
        html = re.sub(r"\s+", " ", html)
        return html.strip()

    def _load_docx(self, filepath: Path) -> dict[str, Any]:
        """Load Word document."""
        doc = Document(filepath)
        content = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

        metadata = {}
        if doc.core_properties:
            props = doc.core_properties
            metadata = {
                "author": props.author,
                "created": str(props.created) if props.created else None,
                "modified": str(props.modified) if props.modified else None,
            }

        return {"content": content, "metadata": metadata}

    def _load_xlsx(self, filepath: Path) -> dict[str, Any]:
        """Load Excel spreadsheet."""
        wb = load_workbook(filepath, data_only=True)
        content_parts = []

        for sheet in wb.worksheets:
            content_parts.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join([str(cell) for cell in row if cell is not None])
                if row_text.strip():
                    content_parts.append(row_text)

        return {
            "content": "\n".join(content_parts),
            "metadata": {"sheets": [sheet.title for sheet in wb.worksheets]},
        }

    def _load_pptx(self, filepath: Path) -> dict[str, Any]:
        """Load PowerPoint presentation."""
        prs = Presentation(filepath)
        content_parts = []

        for i, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text)

            if slide_text:
                content_parts.append(f"Slide {i}:\n" + "\n".join(slide_text))

        return {
            "content": "\n\n".join(content_parts),
            "metadata": {"slide_count": len(prs.slides)},
        }

    def _load_text(self, filepath: Path) -> dict[str, Any]:
        """Load plain text or markdown file."""
        content = filepath.read_text(encoding="utf-8")
        return {"content": content, "metadata": {}}

    def _load_pdf(self, filepath: Path) -> dict[str, Any]:
        """Load PDF using pymupdf for better text extraction.

        When OCR is enabled and the PDF has little/no extractable text
        (avg chars per page < SPARSE_TEXT_THRESHOLD), falls back to
        Ollama vision model for text extraction.
        """
        doc = fitz.open(filepath)
        content_parts = []

        for page in doc:
            text = page.get_text()
            if text.strip():
                content_parts.append(text)

        page_count = len(doc)
        doc.close()

        content = "\n\n".join(content_parts)
        ocr_used = False

        total_chars = sum(len(part) for part in content_parts)
        avg_chars_per_page = total_chars / page_count if page_count > 0 else 0

        ocr_text_path: str | None = None

        if avg_chars_per_page < SPARSE_TEXT_THRESHOLD and self._ocr_enabled:
            cached_path = (
                self._ocr_output_dir / (filepath.stem + ".txt")
                if self._ocr_output_dir is not None
                else None
            )
            if cached_path is not None and cached_path.exists():
                logger.info("OCR cache hit for %s", filepath.name)
                content = cached_path.read_text(encoding="utf-8")
                ocr_used = True
                ocr_text_path = str(cached_path)
            else:
                logger.info(
                    "Sparse text in %s (avg %.0f chars/page), attempting OCR",
                    filepath.name,
                    avg_chars_per_page,
                )
                try:
                    ocr_text = ocr_pdf_pages(
                        filepath,
                        model=self._ocr_model,
                        language=self.ocr_language,
                        provider=self._ocr_provider,
                        base_url=self._ollama_base_url,
                        api_key=self._openrouter_api_key,
                    )
                    if ocr_text.strip():
                        content = ocr_text
                        ocr_used = True
                        if self._ocr_output_dir is not None:
                            self._ocr_output_dir.mkdir(parents=True, exist_ok=True)
                            out_path = self._ocr_output_dir / (filepath.stem + ".txt")
                            out_path.write_text(content, encoding="utf-8")
                            ocr_text_path = str(out_path)
                except Exception:
                    logger.exception("OCR failed for %s", filepath.name)

        metadata: dict[str, Any] = {"page_count": page_count, "ocr_used": ocr_used}
        if ocr_used:
            metadata["ocr_model"] = self._ocr_model
            if ocr_text_path is not None:
                metadata["ocr_text_path"] = ocr_text_path

        return {"content": content, "metadata": metadata}

    @classmethod
    def is_supported(cls, filepath: Path | str) -> bool:
        """Check if file type is supported.

        Args:
            filepath: Path to check.

        Returns:
            True if supported, False otherwise.
        """
        return Path(filepath).suffix.lower() in cls.SUPPORTED_EXTENSIONS
