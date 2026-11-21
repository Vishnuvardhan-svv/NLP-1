"""Safe text extraction for supported resume document formats."""

from io import BytesIO
import re

MAX_FILE_BYTES = 10 * 1024 * 1024


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from a PDF or DOCX byte payload.

    Raises ValueError for unsupported, empty, or oversized input and RuntimeError
    when a supported document cannot be parsed.
    """
    if not filename or not isinstance(content, bytes) or not content:
        raise ValueError("Please provide a non-empty PDF or DOCX file.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The file is larger than the 10 MB upload limit.")

    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if suffix == "pdf":
            import pymupdf

            with pymupdf.open(stream=content, filetype="pdf") as document:
                return "\n".join(page.get_text("text") for page in document).strip()
        if suffix == "docx":
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                paragraphs.extend(cell.text for row in table.rows for cell in row.cells)
            return "\n".join(text for text in paragraphs if text.strip()).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not read this {suffix.upper()} document: {exc}") from exc

    raise ValueError("Unsupported file type. Upload a PDF or DOCX document.")


def extract_profile_sections(text: str) -> dict[str, list[str]]:
    """Best-effort extraction of education, experience, and project sections."""
    headings = {
        "education": re.compile(r"^\s*(education|academic background|qualifications)\s*:?\s*$", re.I),
        "experience": re.compile(r"^\s*(work experience|professional experience|experience|employment history)\s*:?\s*$", re.I),
        "projects": re.compile(r"^\s*(projects|personal projects|selected projects)\s*:?\s*$", re.I),
    }
    sections: dict[str, list[str]] = {key: [] for key in headings}
    active = None
    for line in text.splitlines():
        matched = next((name for name, pattern in headings.items() if pattern.match(line.strip())), None)
        if matched:
            active = matched
        elif active and line.strip():
            if re.match(r"^[A-Z][A-Z\s&/]{2,30}:?\s*$", line.strip()) and not line.strip().startswith("-"):
                active = None
            else:
                sections[active].append(line.strip())
    return sections
