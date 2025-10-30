# CV Insight RAG (LangChain + Gradio)

CV Insight RAG is a lightweight prototype that lets you:

* Upload a candidate CV (PDF)
* Automatically extract key skills and seniority signals
* Ask natural language questions about the CV (e.g. "Does this person have Azure experience?", "Summarise their leadership experience")
* Get grounded answers backed only by the content of the uploaded CV

It uses Retrieval-Augmented Generation (RAG) under the hood: we embed the CV, index it, retrieve the most relevant chunks for every question, then ask an LLM to answer using just that evidence.

This is designed as a starter for:

* Recruiters / Talent teams
* Internal hiring tools
* AI-powered screening assistants
* “Show me how good you are with AI” demo projects

---

## 🔍 High-level Architecture

1. **Upload CV (PDF)**
   The file is parsed and converted to raw text.

2. **Chunk & Embed**
   The text is split into overlapping semantic chunks.
   Each chunk is embedded into a vector using `OpenAIEmbeddings` (via LangChain).

3. **Vector Store (FAISS)**
   All embeddings are stored in an in-memory FAISS index.
   We keep that index in session so you can query it.

4. **RAG Q&A**
   When you ask a question:

   * We embed the question
   * We retrieve the most relevant CV chunks
   * We pass those chunks + your question to an LLM (`ChatOpenAI`)
   * We generate an answer that cites the CV and refuses to hallucinate

5. **UI (Gradio)**

   * Skill/competency summary is shown instantly
   * You get an interactive chat interface for Q&A on that specific CV

---

## 🧱 Tech Stack

* **Python**
* **Gradio** – quick interactive UI
* **LangChain**

  * `ChatOpenAI` for LLM responses
  * `OpenAIEmbeddings` for embeddings
  * `RetrievalQA` for the RAG chain
  * `FAISS` as the vector store
* **pypdf** – PDF text extraction

This prototype currently assumes:

* You are okay using OpenAI for embeddings + generation.
* You are running a single-user, in-memory session (no auth, no persistence).

---

## 📂 Project Structure

```text
cv-rag-app/
├─ app.py                      # Gradio UI: upload CV, chat interface
├─ rag_pipeline.py             # Ingestion + LangChain RetrievalQA chain factory
├─ cv_parser.py                # Extracts text from PDF
├─ profile_extract.py          # Heuristic skill/seniority summary for display
├─ requirements.txt            # Python dependencies
├─ .env.example                # Example for API keys
└─ README.md                   # You're reading this
```

### `app.py`

* Defines the Gradio Blocks interface.
* Manages app state (`gr.State`) including:

  * The active QA chain for the uploaded CV
  * The running chat transcript

### `rag_pipeline.py`

Core logic:

* Extract text from PDF
* Chunk using `RecursiveCharacterTextSplitter`
* Embed with `OpenAIEmbeddings`
* Build a FAISS vectorstore
* Create a LangChain `RetrievalQA` chain using `ChatOpenAI`

Exposes:

* `ingest_cv_build_chain(file_obj)` → builds that chain and returns session state
* `ask_question(session_state, question)` → runs RAG to answer

### `cv_parser.py`

* Opens a PDF and returns cleaned text.

### `profile_extract.py`

* Heuristic (rule-based, token-free) guess of:

  * seniority (Lead / Senior / etc.)
  * tech skills / domains (Azure, Kafka, React, etc.)

Used to render the “Candidate Summary” panel after upload.

> This is intentionally lightweight and doesn’t cost tokens. You can swap it for an LLM-based profiler later.

---

## ⚙️ Setup & Installation

### 1. Clone / copy the project

```bash
git clone <your-repo-url> cv-rag-app
cd cv-rag-app
```

(or just create the folder manually and drop these files in.)

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your OpenAI API key

This app currently uses:

* `ChatOpenAI` for answering
* `OpenAIEmbeddings` for vectorization

Both require `OPENAI_API_KEY`.

You can set it in your shell:

```bash
export OPENAI_API_KEY="sk-..."      # macOS / Linux
setx OPENAI_API_KEY "sk-..."        # Windows PowerShell
```

Or create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
# then edit .env and paste your key
```

`langchain-openai` will automatically pick up `OPENAI_API_KEY` from environment variables.

---

## ▶️ Run the app

From inside the virtual environment:

```bash
python app.py
```

Gradio will start a local web UI.
Open it in your browser.

You’ll see:

* A file uploader for a PDF CV
* A “Candidate Summary” panel with detected skills & seniority guess
* A chat panel where you can ask questions about the CV

Example questions to try:

* “Give me a bullet-point summary of this candidate's cloud experience.”
* “Has this person built event-driven microservices?”
* “Does the CV mention any leadership of teams or mentoring juniors?”
* “Write me a short recruiter pitch for this candidate for a Senior Backend role.”

---

## 💬 How Retrieval Works (RAG Flow)

Under the hood:

1. The PDF is parsed to text (`cv_parser.py`).
2. We split that text into overlapping chunks (800 chars, 200 char overlap) using `RecursiveCharacterTextSplitter`.
   This helps preserve context across bullet points and role descriptions.
3. We create embeddings for each chunk (vector representation) using `OpenAIEmbeddings`.
4. We store those embeddings in FAISS (in memory).
5. When you ask something:

   * We embed the question
   * We find the top ~4 most relevant chunks
   * We build a prompt:

     * Your question
     * Those chunks as “context”
     * A system-style instruction that says “Do not make things up. If the CV doesn’t say it, say that.”
   * We send that to an LLM to generate the answer.

LangChain’s `RetrievalQA` chain handles the retrieval + prompt assembly + LLM call.

This means the model is grounded in the CV.
If the CV doesn’t include a skill, good answers should explicitly say `"The CV does not clearly provide that information."`

---

## 🔐 Notes / Limitations

* **No persistence yet.**
  When you upload a CV, it lives in memory for that session only.
  If you refresh, it’s gone.
  If you upload a new CV, it replaces the previous one.

* **Single-user assumption.**
  This prototype is not multi-tenant safe.
  If you run this on a server, all users share the same in-memory state.
  For production, you'd isolate per session/user, or persist indexes per CV.

* **Embeddings cost money.**
  Every upload will call embeddings on all chunks.
  Large CVs cost slightly more than short CVs.
  You can switch to local embeddings later (e.g. `sentence-transformers`) if you don’t want to call OpenAI.

* **We don't redact PII.**
  Candidate name, email, phone etc. are in the text and in the context passed to the model.
  For compliance you may want to add a redaction pass.
  You could also add a “Generate anonymised CV” button (very easy with the current structure).

---

## 🛠 Roadmap / Next Features

These are natural upgrades:

1. **Job Description Match**

   * Upload a JD alongside the CV
   * Create a second vector index for the JD
   * Ask: “How well does this person match this role? Score / gaps / pitch.”

2. **Export Recruiter Pitch**

   * Button: “Generate email pitch to hiring manager”
   * Output: bullet points + role fit + notice period (if stated in the CV)

3. **Multi-CV Compare**

   * Upload 2+ CVs
   * Ask: “Who has stronger Azure experience?”

4. **Persistence**

   * Store FAISS indexes on disk using `FAISS.save_local()` / `FAISS.load_local()`
   * Give each CV an ID and reload them later

5. **Local / Private Mode**

   * Swap `OpenAIEmbeddings` for `HuggingFaceEmbeddings`
   * Swap `ChatOpenAI` for a local model via `langchain_community.llms.Ollama`
   * Result: fully offline / on-prem mode for privacy-sensitive clients

---

## ❓ FAQ

**Q: Do you retrain a model on the CV?**
A: No. We’re not fine-tuning. We’re doing retrieval: embed CV → search → feed only relevant passages to the LLM at question time.

**Q: Can it hallucinate?**
A: It's less likely because we explicitly tell it “Only answer using this context.”
If the info is missing, it should say so. Still, you should not treat the output as a legal/HR decision on its own.

**Q: Can I run this offline without OpenAI?**
A: Not in this base version.
But yes, architecturally it’s easy:

* Replace `OpenAIEmbeddings` with `HuggingFaceEmbeddings`
* Replace `ChatOpenAI` with a local LLM via Ollama or similar
* Everything else stays the same

---

## ✅ TL;DR for stakeholders

* Upload a PDF CV.
* The app builds a private knowledge base from that CV.
* You can ask targeted questions and get grounded answers.
* There’s no DB, no infra — just `python app.py`.

This is the minimal working skeleton for an AI recruiter assistant / talent screener / CV intelligence tool, built with LangChain + Gradio.
