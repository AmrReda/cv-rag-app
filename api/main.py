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
    return os.path.join(DATA_DIR, f"faiss_{doc_id}")

def _build_chain_from_text(text: str):
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

@app.post("/ingestions", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), document_id: Optional[str] = None):
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
