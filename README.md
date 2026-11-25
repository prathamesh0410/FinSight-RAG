# FinSight RAG — Banking Document Intelligence System

## 📖 What It Is

**FinSight RAG** is a banking-specialized Retrieval-Augmented Generation (RAG) system that enables users to upload regulatory documents (RBI circulars, loan agreements, compliance policies, risk reports) and ask natural language questions with **verified inline citations**.

> **In simple words:** You upload banking PDFs, then ask questions in plain English, and it gives you exact answers with proof of exactly which document, page, and clause the answer came from.

---

## 🚀 Quick Start — 3 Ways to Run

### Option 1: **Docker Compose** (Recommended — everything in one command)

```bash
# 1. Navigate to project
cd FinSight-RAG

# 2. Copy environment file
cp .env.example .env

# 2.5. EDIT .env — add your OPENAI_API_KEY (required for LLM)
# In VS Code: Ctrl+P → .env → paste your key

# 3. Start all services
docker-compose up --build

# 4. Access endpoints:
# 📄 API Docs:      http://localhost:8000/docs        ← Swagger/OpenAPI
# 💬 Streamlit UI:  http://localhost:8501            ← Old Python UI
# 🎨 HTML UI:       open frontend/chat.html in browser  ← NEW Modern UI
```

---

### Option 2: **Without Docker** (Python direct)

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -e .[dev,streamlit,frontend]

# 3. Set environment variables
copy .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Run API server (new terminal)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4. Run Streamlit UI (new terminal)
streamlit run streamlit_app/app.py --server.port 8501

# 5. Open HTML UI (any browser)
# Open frontend/chat.html — it will auto-connect to http://localhost:8000/api/v1
```

---

### Option 3: **HTML Frontend Only** (No Python backend needed)

```bash
# 1. Open the frontend in any browser (Chrome/Firefox/Edge)
open frontend/chat.html

# 2. The HTML UI auto-connects to http://localhost:8000/api/v1
# If you don't have the API running, it will show a setup notice

# 3. Features:
#   • Upload PDF/DOCX/TXT documents via drag-and-drop
#   • Ask questions in natural language
#   • Get answers with inline citations (document name, page, clause)
#   • Sample queries for quick starts
#   • Responsive — works on mobile & desktop
```

---

## 📦 Project Structure

```
FinSight-RAG/
├── app/                          # Python FastAPI backend
│   ├── main.py                   # Entry point
│   ├── core/                     # Config, DB, Security
│   ├── api/routes/               # Endpoints (upload, query, metadata)
│   ├── models/                   # SQLAlchemy models
│   ├── schemas/                  # Pydantic request/response
│   └── services/                 # AI brain (RAG, embeddings, search)
├── streamlit_app/                # Old Python-based UI
│   └── app.py
├── frontend/                     # NEW Modern HTML + Tailwind UI
│   └── chat.html                 # Interactive web interface
├── docker-compose.yml            # Multi-service container setup
├── Dockerfile                    # API container
├── Dockerfile.streamlit          # Streamlit container
├── pyproject.toml                # Dependencies + tools config
├── .env.example                  # Environment variables template
├── README.md                     # This file
└── .gitignore
```

---

## 🏗 How It Works (The RAG Pipeline)

```mermaid
graph TD
    subgraph "Upload Pipeline (Pipeline A)"
        U1[User Uploads PDF/DOCX] -->|FastAPI POST /documents/upload| A[IngestionService]
        A -->|Extract text| B[Document Processor]
        B -->|Smart Chunking (Legal-aware)| C[EmbeddingService]
        C -->|Vector embeddings| D[PostgreSQL + pgvector]
    end

    subgraph "Query Pipeline (Pipeline B)"
        Q1[User Types Question] -->|POST /chat/query| E[RAGPipeline]
        E -->|Embed query| F[Vector Store (pgvector + HNSW)]
        F -->|Hybrid Search (BM25 + Vector)| G[Cross-Encoder Re-Ranker]
        G -->|Top-k Chunks| H[LLM (OpenAI/Anthropic)]
        H -->|Answer + Citations| I[User Interface]
    end
```

---

## 🎯 Role Coverage (Proves 5 Roles)

| Role | What This Project Proves |
|------|--------------------------|
| **GenAI Engineer** | Full RAG pipeline: chunking → embeddings → retrieval → generation |
| **ML Engineer** | Embedding models, retrieval metrics, re-ranker optimization |
| **AI/ML Banking** | RBI circular parsing, loan metadata, risk tagging, compliance keywords |
| **Python Backend Engineer** | FastAPI async, PostgreSQL+pgvector, Docker, CI/CD pipeline |
| **Data Analyst** | Query logging, analytics dashboards, Power BI integration point |

---

## 📊 Running the APIs

### Test Document Upload
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@sample.pdf" \
  -F "document_type=rbi_circular"
```

### Test Query
```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does RBI say about capital adequacy?", "top_k": 5, "use_reranker": true}'
```

### Sample Response
```json
{
  "query": "What does RBI say about capital adequacy?",
  "answer": "According to RBI Circular No. RBI/2023-24/17, banks must maintain minimum capital adequacy ratio of 9% of risk-weighted assets...",
  "citations": [
    {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 3,
      "content_preview": "...capital adequacy ratio...",
      "page_number": 12,
      "clause_reference": "§4.1(a)",
      "similarity_score": 0.94
    }
  ],
  "response_time_ms": 1240,
  "retrieval_count": 8,
  "context_used": 5
}
```

### Health Check
```bash
curl http://localhost:8000/health
# → {"status":"healthy","service":"finsight-rag","version":"0.1.0"}
```

### API Documentation (Swagger UI)
```bash
# Visit in browser:
http://localhost:8000/docs
```

---

## 📦 Dependencies (pyproject.toml)

```toml
# Core dependencies
fastapi>=0.110.0, uvicorn[standard]>=0.29.0, sqlalchemy>=2.0.0, alembic>=1.13.0,
asyncpg>=0.29.0, pgvector>=0.2.0, pydantic>=2.7.0, python-jose[cryptography]>=3.3.0,
passlib[bcrypt]>=1.7.4, python-multipart>=0.0.9, langchain>=0.2.0, redis>=5.0.0,
celery>=5.4.0, structlog>=24.0.0, rank-bm25>=0.2.2

# Dev & Testing
pytest>=8.0.0, pytest-asyncio>=0.23.0, pytest-cov>=5.0.0, playwright>=1.40.0,
ruff>=0.5.0, mypy>=1.10.0, testcontainers>=4.7.0

# ML / AI
sentence-transformers>=3.0.0, openai, orjson>=3.10.0

# Frontend
# HTML frontend uses CDN (Tailwind CSS, Font Awesome) — no build steps required
```

---

## 🔐 Environment Variables (.env.example)

```env
APP_NAME="FinSight RAG"
APP_ENV="development"
DEBUG=true
SECRET_KEY="change-me-in-production-use-openssl-rand-hex-32"
API_V1_PREFIX="/api/v1"
DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/finsight"
REDIS_URL="redis://localhost:6379/0"
OPENAI_API_KEY="sk-..."           ← REQUIRED: Get from platform.openai.com
ANTHROPIC_API_KEY="sk-ant-..."    ← OPTIONAL: For Anthropic Claude
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNKING_STRATEGY="legal"
TOP_K=5
SIMILARITY_THRESHOLD=0.7
USE_RERANKER=true
RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DIMENSION=1536
MAX_FILE_SIZE_MB=50
SUPPORTED_EXTENSIONS=".pdf,.docx,.txt,.md"
STREAMLIT_SERVER_PORT=8501
```

---

## 🤝 Author

**Prathamesh Bandekar**
- LinkedIn: [prathamesh-r20](https://linkedin.com/in/prathamesh-r20)
- Email: prathamesh04102003@gmail.com
- GitHub: [PrathameshBandekar-123](https://github.com/PrathameshBandekar-123)

---

*Built as a portfolio project demonstrating production-grade GenAI engineering practices with banking domain specialization.*