from typing import Dict, Any
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from cv_parser import extract_text_from_pdf
from profile_extract import build_profile_summary_markdown


def ingest_cv_build_chain(file_obj) -> Dict[str, Any]:
    """
    file_obj: binary handle to the uploaded PDF
    Returns a dict with:
      - qa_chain: LangChain RetrievalQA chain
      - profile_md: markdown summary of the CV
    """

    # 1. Extract raw CV text
    raw_text = extract_text_from_pdf(file_obj)

    # 2. Chunk text using LangChain's splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # chars not tokens, fine for now
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(raw_text)

    # 3. Embed + store in FAISS
    embedding_model = OpenAIEmbeddings()  
    # note: this uses text-embedding-3-large/small depending on defaults in your langchain-openai version.
    # you can force: OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_texts(chunks, embedding_model)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 4. LLM for answering questions
    llm = ChatOpenAI(
        model="gpt-4.1-mini",    # adjust to what you have access to
        temperature=0.2,
        max_tokens=400,
    )

    # 5. RetrievalQA chain
    #   - stuff retrieved chunks into the prompt along with the question
    #   - ask the model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # simplest: put top docs into a single prompt
        chain_type_kwargs={
            "prompt": build_prompt_template()
        },
        return_source_documents=False,
    )

    # 6. Build profile summary markdown to show in UI
    profile_md = build_profile_summary_markdown(raw_text)

    return {
        "qa_chain": qa_chain,
        "profile_md": profile_md,
    }


def build_prompt_template():
    """
    Build a custom prompt for the RetrievalQA chain.
    We'll tell the model to only use given context.
    """

    from langchain.prompts import PromptTemplate

    template = """You are an AI CV analyst.
Your job is to answer questions about the candidate based ONLY on the provided CV context.
If you don't find something in the context, say "The CV does not clearly provide that information."

Context from CV:
{context}

Question:
{question}

Instructions:
1. Refer to evidence directly (e.g. 'In Work Experience: ...').
2. Be concise. Bullet points are OK.
3. Do not invent experience that's not explicitly present.

Answer:
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def ask_question(session_state: Dict[str, Any], question: str) -> str:
    """
    Uses the qa_chain created during ingestion to answer user questions.
    """
    qa_chain = session_state["qa_chain"]

    # RetrievalQA expects {"query": "..."}
    result = qa_chain.invoke({"query": question})

    # RetrievalQA returns a dict. By default answer is under 'result'
    answer = result.get("result", "").strip()
    return answer


def score_candidate_fit(cv_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Scores the candidate's fit against a job description.
    Returns a dict with match_score, strengths, gaps, and pitch.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.2, model_kwargs={"response_format": {"type": "json_object"}})
    
    prompt = f"""
    You are an expert technical recruiter. Compare the following CV against the Job Description (JD).
    
    CV Content:
    {cv_text[:4000]}
    
    Job Description:
    {jd_text[:4000]}
    
    Output a JSON object with the following keys:
    - "match_score": integer between 0 and 100
    - "strengths": list of strings (key matching skills/experience)
    - "gaps": list of strings (missing requirements)
    - "pitch": a short 2-3 sentence pitch for this candidate
    
    JSON Output:
    """
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {
            "match_score": 0,
            "strengths": [],
            "gaps": ["Error parsing analysis"],
            "pitch": "Could not analyze fit."
        }


def extract_logistics(cv_text: str) -> Dict[str, str]:
    """
    Extracts logistics information from the CV.
    Returns location, availability, work rights, and remote preference.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    prompt = f"""
    Extract the following logistics information from the CV text below.
    If not found, set value to "Not stated".
    
    CV Text:
    {cv_text[:3000]}
    
    Output JSON with keys:
    - "location": current city/country
    - "availability": notice period or start date
    - "work_rights": visa status or citizenship if mentioned
    - "remote_preference": remote/hybrid/onsite if mentioned
    """
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return {"location": "Unknown", "availability": "Unknown", "work_rights": "Unknown", "remote_preference": "Unknown"}


def extract_experience_timeline(cv_text: str) -> list:
    """
    Extracts a timeline of professional experience.
    Returns a list of objects with title, company, start, end, impact.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    prompt = f"""
    Extract the professional experience timeline from the CV.
    Return a JSON object with a key "timeline" containing a list of roles.
    Each role should have:
    - "title"
    - "company"
    - "start" (e.g. "Jan 2020")
    - "end" (e.g. "Present" or "Dec 2022")
    - "impact" (short summary of key achievement, max 10 words)
    
    CV Text:
    {cv_text[:4000]}
    """
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
        return data.get("timeline", [])
    except:
        return []
