from pathlib import Path

from app.core.utils.text_cleaner import clean_text


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return clean_text(_read_pdf(path))
    if suffix == ".docx":
        return clean_text(_read_docx(path))
    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .pdf, or .docx")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to read PDF files.") from exc

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Install python-docx to read DOCX files.") from exc

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
