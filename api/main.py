from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import io, os, json

# reuse your existing modules
from cv_parser import extract_text_from_pdf
from rag_pipeline import build_prompt_template
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from worker_app import ingest_pdf_task

app = FastAPI(title="CV RAG API", version="0.1.0")

DATA_DIR = os.environ.get("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

def _faiss_paths(doc_id: str):
    """
    Constructs the file path for the FAISS index based on the document ID.

    Args:
        doc_id (str): The unique identifier for the document.

    Returns:
        str: The full path to the FAISS index directory.
    """
    return os.path.join(DATA_DIR, f"faiss_{doc_id}")

def _build_chain_from_text(text: str):
    """
    Builds a RetrievalQA chain from raw text.
    Splits text, creates embeddings, builds FAISS index, and sets up the QA chain.

    Args:
        text (str): The raw text content.

    Returns:
        tuple: A tuple containing the QA chain and the vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()  # set OPENAI_API_KEY env
    vs = FAISS.from_texts(chunks, embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
    prompt = build_prompt_template()
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}, return_source_documents=False
    )
    return qa, vs

class IngestResponse(BaseModel):
    document_id: str

class AskRequest(BaseModel):
    document_id: str
    question: str

class MatchRequest(BaseModel):
    document_id: str
    jd_text: str

@app.post("/ingestions", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), document_id: Optional[str] = None):
    """
    Ingests a PDF file.
    Extracts text, creates embeddings, and stores them in a FAISS index.
    Offloads the heavy lifting to a Celery worker.

    Args:
        file (UploadFile): The PDF file to ingest.
        document_id (Optional[str]): Optional custom document ID.

    Returns:
        IngestResponse: The document ID of the ingested file.
    """
    # simple id: filename without spaces + numeric suffix
    doc_id = (document_id or file.filename.replace(" ", "_")).rsplit(".", 1)[0]

    content = await file.read()
    try:
        # Enqueue task to Celery
        ingest_pdf_task.delay(doc_id, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest: {e}")
    return IngestResponse(document_id=doc_id)

@app.post("/queries")
async def ask(payload: AskRequest):
    """
    Answers a question about a specific document.
    Retrieves relevant context from the FAISS index and uses an LLM to generate the answer.

    Args:
        payload (AskRequest): The request payload containing document ID and question.

    Returns:
        JSONResponse: The answer to the question.
    """
    idx_path = _faiss_paths(payload.document_id)
    if not os.path.exists(idx_path):
        raise HTTPException(status_code=404, detail="Document not found or not embedded.")
    embeddings = OpenAIEmbeddings()
    vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2)
    prompt = build_prompt_template()
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}, return_source_documents=False
    )
    result = qa.invoke({"query": payload.question})
    answer = result.get("result", "").strip()
    return JSONResponse({"answer": answer})

@app.post("/matches")
async def match_candidate(
    file: UploadFile = File(None), 
    jd_file: UploadFile = File(None),
    document_id: Optional[str] = Body(None),
    jd_text: Optional[str] = Body(None)
):
    """
    Scores a candidate against a job description.
    Accepts either file uploads (CV + JD) or existing document_id + JD text.

    Args:
        file (UploadFile): The CV file (optional).
        jd_file (UploadFile): The job description file (optional).
        document_id (Optional[str]): The document ID (optional).
        jd_text (Optional[str]): The job description text (optional).

    Returns:
        JSONResponse: The score and details of the candidate-job description match.
    """
    from rag_pipeline import score_candidate_fit
    
    # 1. Get CV Text
    cv_text = ""
    if file:
        content = await file.read()
        cv_text = extract_text_from_pdf(io.BytesIO(content))
    elif document_id:
        # Load from FAISS? No, we need raw text for scoring. 
        # Ideally we should have stored raw text. 
        # For now, let's assume we can't easily get raw text back from FAISS without storing it.
        # Fallback: We will require file upload for now or implement text persistence.
        # Let's just fail if no file provided for this iteration as per plan "run transient"
        raise HTTPException(status_code=400, detail="CV file upload required for matching in this version.")
    else:
        raise HTTPException(status_code=400, detail="CV file or document_id required.")

    # 2. Get JD Text
    job_description = ""
    if jd_file:
        jd_content = await jd_file.read()
        # Assume JD is PDF for now, or text? Let's try PDF parser if PDF, else decode
        if jd_file.filename.endswith(".pdf"):
            job_description = extract_text_from_pdf(io.BytesIO(jd_content))
        else:
            job_description = jd_content.decode("utf-8")
    elif jd_text:
        job_description = jd_text
    else:
        raise HTTPException(status_code=400, detail="JD file or text required.")

    # 3. Score
    result = score_candidate_fit(cv_text, job_description)
    return JSONResponse(result)

@app.get("/documents")
async def list_documents():
    """
    Lists all ingested documents available in the data directory.
    Returns a list of document IDs.
    """
    if not os.path.exists(DATA_DIR):
        return []
    
    docs = []
    for name in os.listdir(DATA_DIR):
        if name.startswith("faiss_"):
            doc_id = name.replace("faiss_", "")
            docs.append({"document_id": doc_id})
    return docs
