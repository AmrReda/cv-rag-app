# CV Insight RAG (FastAPI + Gradio + Celery)

CV Insight RAG is a production-ready service that lets you:

* Upload a candidate CV (PDF)
* Automatically extract key skills, seniority, logistics, and experience timeline
* Score candidates against a Job Description (JD)
* Ask natural language questions about the CV
* Get grounded answers backed only by the content of the uploaded CV

It uses Retrieval-Augmented Generation (RAG) under the hood: we embed the CV, index it, retrieve the most relevant chunks for every question, then ask an LLM to answer using just that evidence.

This project has been refactored from a simple prototype into a scalable service with:
* **FastAPI** for the backend API
* **Celery + Redis** for async background processing (ingestion)
* **Docker** for containerization
* **Terraform** for AWS infrastructure (ECS Fargate, RDS, ElastiCache)
* **Gradio** for the demo UI

---

## 🔍 High-level Architecture

1. **Upload CV (PDF)**
   The file is uploaded via API or UI.
   The ingestion task is offloaded to a Celery worker.

2. **Chunk & Embed (Async Worker)**
   The worker parses the PDF, splits text into chunks, and creates embeddings using `OpenAIEmbeddings`.

3. **Vector Store (FAISS)**
   Embeddings are stored in a local FAISS index (persisted to disk in `data/`).
   Each CV gets a unique Document ID.

4. **RAG Q&A**
   When you ask a question:
   * We load the specific FAISS index for that document.
   * We retrieve relevant chunks.
   * We pass chunks + question to `ChatOpenAI`.

5. **JD Matching & Analysis**
   * Upload a JD to score the candidate (0-100 match).
   * Extract logistics (location, notice period) and experience timeline automatically.

---

## 🧱 Tech Stack

* **Backend:** FastAPI, Uvicorn
* **Async Workers:** Celery, Redis
* **AI/ML:** LangChain, OpenAI (GPT-4o-mini), FAISS
* **Infrastructure:** Docker, Terraform (AWS ECS Fargate)
* **UI:** Gradio

---

## 📂 Project Structure

```text
cv-rag-app/
├─ api/
│  └─ main.py                  # FastAPI entrypoint
├─ infra/
│  └─ aws/                     # Terraform IaC
├─ app.py                      # Gradio UI
├─ worker_app.py               # Celery worker for ingestion
├─ rag_pipeline.py             # Core RAG & Analysis logic
├─ cv_parser.py                # PDF extraction
├─ profile_extract.py          # Heuristic extraction
├─ Dockerfile                  # Multi-stage build
├─ requirements.txt            # Core dependencies
└─ requirements-api.txt        # API/Worker dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone the project

```bash
git clone <your-repo-url> cv-rag-app
cd cv-rag-app
```

### 2. Environment Setup

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
DATA_DIR=./data
REDIS_URL=redis://localhost:6379/0
```

### 3. Run with Docker Compose (Recommended)

(Create a `docker-compose.yml` if you want full local orchestration, otherwise run services individually)

### 4. Run Locally (Manual)

**Start Redis:**
```bash
docker run -d -p 6379:6379 redis
```

**Start Worker:**
```bash
celery -A worker_app worker --loglevel=info
```

**Start API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Start UI (Optional):**
```bash
python app.py
```

---

## 🚀 API Endpoints

The FastAPI service exposes the following endpoints:

*   `POST /ingestions`: Upload a PDF CV. Returns `document_id`. (Async)
*   `POST /queries`: Ask a question about a document. `{ document_id, question }`
*   `POST /matches`: Score a candidate against a JD. Upload CV+JD or use `document_id`.
*   `GET /documents`: List all ingested documents.

---

## ☁️ Infrastructure (AWS)

The `infra/aws` folder contains Terraform code to deploy:
*   **VPC & Networking**: Public/Private subnets, ALB.
*   **ECS Fargate**: API Service and Worker Service.
*   **Data Stores**: S3 (Blobs), RDS (Postgres - placeholder), ElastiCache (Redis).
*   **Secrets**: OpenAI API Key stored in AWS Secrets Manager.

To deploy:
```bash
cd infra/aws
terraform init
terraform apply -var="db_password=..."
```

---

## ✅ Features

1.  **Async Ingestion**: Uploads are fast; processing happens in the background.
2.  **Persistence**: CVs are indexed and saved to disk; reload them anytime.
3.  **JD Scoring**: Compare a CV against a Job Description to get a match score, strengths, and gaps.
4.  **Smart Extraction**: Auto-extracts logistics (visa, location) and career timeline.
5.  **Multi-CV**: Switch between candidates easily in the UI.

---

## ❓ FAQ

**Q: Can I run this offline?**
A: Yes, swap `OpenAIEmbeddings` and `ChatOpenAI` for local alternatives (Ollama/HuggingFace) in `rag_pipeline.py`.

**Q: Is data persisted?**
A: Yes, FAISS indexes are saved to `DATA_DIR`. In AWS, you would mount an EFS volume to `/data` for persistence across Fargate tasks.
