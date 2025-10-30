from pypdf import PdfReader
import re
from typing import BinaryIO


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    file_obj: open file-like object in 'rb' mode.
    Returns combined text of all pages.
    """
    reader = PdfReader(file_obj)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)

    # light cleanup
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()
