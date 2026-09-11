"""
Extracts plain text from uploaded PDF/DOCX files, in-memory (no temp files
on disk — an uploaded file is untrusted input, and writing it to disk before
even validating its content adds an unnecessary attack surface for no
benefit here, since both libraries accept in-memory bytes directly).

PDF extraction is adapted from train_model/analyze_document.py's
extract_text_from_pdf, changed to work on bytes rather than a file path.
Import as `pymupdf`, not the legacy `fitz` alias — a different, unrelated
PyPI package is also named `fitz` and can shadow PyMuPDF's own module of
the same name depending on install order (hit this exact collision while
building this feature); `pymupdf` is the modern, unambiguous import name
PyMuPDF itself recommends.
"""

import pymupdf
import docx


class UnsupportedFileType(Exception):
    pass


class DocumentParseError(Exception):
    pass


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise DocumentParseError(f"Could not read PDF: {e}") from e


def extract_text_from_docx_bytes(data: bytes) -> str:
    import io
    try:
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        raise DocumentParseError(f"Could not read DOCX: {e}") from e


# Maps the content-types browsers/clients actually send for these file
# types to the right extractor. Extension is checked too (see
# extract_text_from_upload) since content-type headers from a multipart
# upload are client-supplied and not always accurate.
_EXTRACTORS_BY_CONTENT_TYPE = {
    "application/pdf": extract_text_from_pdf_bytes,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extract_text_from_docx_bytes,
}

_EXTRACTORS_BY_EXTENSION = {
    ".pdf": extract_text_from_pdf_bytes,
    ".docx": extract_text_from_docx_bytes,
}


def extract_text_from_upload(uploaded_file) -> str:
    """
    uploaded_file: a Django UploadedFile (from request.FILES).
    Picks the extractor by file extension first (more reliable than a
    client-supplied Content-Type header), falling back to content_type.
    Raises UnsupportedFileType if neither matches a known format.
    """
    name = (uploaded_file.name or "").lower()
    extractor = next((fn for ext, fn in _EXTRACTORS_BY_EXTENSION.items() if name.endswith(ext)), None)

    if extractor is None:
        extractor = _EXTRACTORS_BY_CONTENT_TYPE.get(uploaded_file.content_type)

    if extractor is None:
        raise UnsupportedFileType(
            f"Unsupported file type '{uploaded_file.content_type}' for '{uploaded_file.name}'. "
            "Only .pdf and .docx are supported."
        )

    return extractor(uploaded_file.read())
