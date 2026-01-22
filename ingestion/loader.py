"""Multi-format document loader."""

import email
from email.message import Message
from pathlib import Path
from typing import Any

import fitz  # pymupdf
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".eml", ".docx", ".xlsx", ".pptx", ".md", ".txt", ".pdf"}

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
                "from": msg.get("From"),
                "to": msg.get("To"),
                "subject": msg.get("Subject"),
                "date": msg.get("Date"),
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
        """Load PDF using pymupdf for better text extraction."""
        doc = fitz.open(filepath)
        content_parts = []

        for page in doc:
            text = page.get_text()
            if text.strip():
                content_parts.append(text)

        doc.close()

        return {
            "content": "\n\n".join(content_parts),
            "metadata": {"page_count": len(doc)},
        }

    @classmethod
    def is_supported(cls, filepath: Path | str) -> bool:
        """Check if file type is supported.

        Args:
            filepath: Path to check.

        Returns:
            True if supported, False otherwise.
        """
        return Path(filepath).suffix.lower() in cls.SUPPORTED_EXTENSIONS
