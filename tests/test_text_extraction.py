"""Synthetic document fixtures used to verify resume extraction."""

from io import BytesIO

import pytest

from ml.preprocessing.text_extraction import extract_text


def test_extracts_docx_paragraphs_and_table():
    from docx import Document

    doc = Document()
    doc.add_paragraph("Synthetic sample resume: Python developer")
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Synthetic project: data pipeline"
    buffer = BytesIO()
    doc.save(buffer)

    result = extract_text("sample.docx", buffer.getvalue())
    assert "Python developer" in result
    assert "data pipeline" in result


def test_rejects_unsupported_or_empty_input():
    with pytest.raises(ValueError):
        extract_text("resume.txt", b"not supported")
    with pytest.raises(ValueError):
        extract_text("resume.pdf", b"")


def test_rejects_corrupt_pdf():
    with pytest.raises(RuntimeError, match="Could not read"):
        extract_text("broken.pdf", b"not a pdf")
