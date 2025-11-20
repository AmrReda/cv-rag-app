import io
import os

from celery import Celery
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from cv_parser import extract_text_from_pdf

BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("cv_worker", broker=BROKER_URL, backend=BROKER_URL)
DATA_DIR = os.environ.get("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)


def _faiss_path(doc_id: str) -> str:
    """
    Constructs the file path for the FAISS index based on the document ID.

    Args:
        doc_id (str): The unique identifier for the document.

    Returns:
        str: The full path to the FAISS index directory.
    """
    return os.path.join(DATA_DIR, f"faiss_{doc_id}")


@app.task(acks_late=True)
def ingest_pdf_task(doc_id: str, pdf_bytes: bytes):
    """
    Celery task to ingest a PDF document.
    Extracts text, chunks it, creates embeddings, and saves the FAISS index.

    Args:
        doc_id (str): The unique identifier for the document.
        pdf_bytes (bytes): The raw bytes of the PDF file.

    Returns:
        dict: A dictionary containing the document ID and the number of chunks created.
    """
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, OpenAIEmbeddings())
    vs.save_local(_faiss_path(doc_id))
    return {"document_id": doc_id, "chunks": len(chunks)}

