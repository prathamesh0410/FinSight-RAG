# FinSight RAG — Banking Document Intelligence System

> **Production-ready RAG system for regulatory document querying with inline citations**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue?logo=postgresql)
![pgvector](https://img.shields.io/badge/pgvector-0.2.0+-purple?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-ff4b4d?logo=streamlit)

---

## 📖 What It Is

FinSight RAG is a **banking-specialized Retrieval-Augmented Generation (RAG) system** that enables users to upload regulatory documents (RBI circulars, loan agreements, compliance policies, risk reports) and ask natural language questions with **verified inline citations**.

Built for:
- ✅ **GenAI Engineers** — Full RAG pipeline from chunking to generation
- ✅ **ML Engineers** — Embedding models, retrieval evaluation, vector DB optimization
- ✅ **AI in Banking** — Domain-aware chunking, regulatory metadata, compliance tagging
- ✅ **Python Backend Engineers** — FastAPI, PostgreSQL+pgvector, async architecture
- ✅ **QA Automation Engineers** — Comprehensive test suite, CI/CD pipeline

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- 8GB+ RAM

### Local Development
```bash
# 1. Clone
git clone https://github.com/PrathameshBandekar-123/FinSight-RAG.git
cd FinSight-RAG

# 2. Environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Start with Docker Compose
docker-compose up --build

# 4. Access endpoints
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Streamlit UI: http://localhost:8501
```

### Docker Quick Start
```bash
docker-compose up --build -d
# Stop: docker-compose down
```

### Without Docker
```bash
# Install dependencies
pip install -e .[dev,streamlit]

# Set environment
export OPENAI_API_KEY="sk-..."

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run Streamlit
streamlit run streamlit_app/app.py --server.port 8501
```

---

## 🏗 Architecture

```mermaid
graph TD
    subgraph "Upload Layer"
        U1[PDF/DOCX Upload] -->|FastAPI| A[API Gateway]
    end

    subgraph "Processing Layer"
        A -->|Ingest| B[Document Processor]
        B -->|Chunk| C[Legal-Aware Chunker]
        C -->|Embed| D[Embedding Service]
        D -->|Store| E[PostgreSQL + pgvector]
    end

    subgraph "Retrieval Layer"
        E -->|Semantic Search| F[Vector Store (pgvector + HNSW)]
        F -->|Re-Rank| G[Cross-Encoder]
        G -->|Select| H[Top-k Chunks]
    end

    subgraph "Generation Layer"
        H -->|Context| I[LLM (OpenAI/Anthropic)]
        I -->|Answer| J[Generated Response]
        J -->|Citations| K[Inline Citation Formatter]
    end

    subgraph "User Layer"
        K -->|Streamlit UI| L[User]
        L -->|Query| A
    end
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Python 3.11, SQLAlchemy 2.0, Alembic |
| **Database** | PostgreSQL 15+, pgvector extension |
| **Vector Search** | pgvector, HNSW indexes, BM25 hybrid search |
| **AI/ML** | LangChain, OpenAI/Anthropic API, Sentence-Transformers |
| **Frontend** | Streamlit, HTML/CSS/JS |
| **DevOps** | Docker, Docker Compose, GitHub Actions CI/CD |
| **Testing** | pytest, pytest-asyncio, coverage, codecov |

---

## 📦 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST /api/v1/documents/upload` | Upload & process document |
| `POST /api/v1/chat/query` | Query documents via RAG |
| `GET /api/v1/chat/citations/{doc_id}` | Get document citations |
| `GET /api/v1/health` | Health check |
| `GET /` | API root info |

### Sample Request: Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@rbi_circular.pdf" \
  -F "document_type=rbi_circular"
```

### Sample Request: Query
```bash
curl -X POST "http://localhost:8000/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does RBI say about capital adequacy?", "top_k": 5, "use_reranker": true}'
```

### Sample Response: Query
```json
{
  "query": "What does RBI say about capital adequacy?",
  "answer": "According to RBI Circular No. RBI/2023-24/17, banks must maintain a minimum capital adequacy ratio of 9% of risk-weighted assets...",
  "citations": [
    {
      "document_id": "123e4567-e89b-12d3-a456-426614174000",
      "chunk_index": 3,
      "content_preview": "...capital adequacy ratio...",
      "page_number": 12,
      "section_title": "Capital Adequacy",
      "clause_reference": "§4.1(a)",
      "similarity_score": 0.94
    }
  ],
  "response_time_ms": 1240,
  "retrieval_count": 8,
  "context_used": 5
}
```

### Sample Response: Citation Format
> **[Source: RBI/2023-24/17 §4.1(a)]** — displayed inline after every claim

---

## 🎯 Role Coverage

| Role | How This Project Proves Fit |
|------|----------------------------|
| **GenAI Engineer** | Full RAG pipeline: chunking → embeddings → retrieval → generation with citations |
| **ML Engineer** | Embedding models, retrieval metrics (RAGAS), re-ranker optimization, drift detection ready |
| **AI/ML Banking** | RBI circular parsing, loan agreement metadata, risk tagging, compliance keywords |
| **Python Backend Engineer** | FastAPI async, PostgreSQL+pgvector, Redis caching, Docker, CI/CD |
| **QA Engineer** | Test suite (pytest), CI pipeline (GitHub Actions), code coverage >80% |
| **Data Analyst** | Query logging, analytics dashboards, Power BI integration point |

---

## 📊 Project Metrics (Target)

| Metric | Target |
|--------|--------|
| **API Latency** | P50 < 200ms, P99 < 800ms |
| **RAG Retrieval** | Recall@10 ≥ 0.75 on banking docs |
| **Citation Accuracy** | 100% — every claim must map to a retrieved chunk |
| **Test Coverage** | ≥ 80% (unit + integration) |
| **Docker Build Time** | < 5 minutes |
| **Streamlit Load Time** | < 2 seconds |

---

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Prathamesh Bandekar**
- LinkedIn: [prathamesh-r20](https://linkedin.com/in/prathamesh-r20)
- Email: prathamesh04102003@gmail.com
- GitHub: [PrathameshBandekar-123](https://github.com/PrathameshBandekar-123)
- Portfolio: [GitHub Profile](https://github.com/PrathameshBandekar-123)

---

*Built as a portfolio project demonstrating production-grade GenAI engineering practices with banking domain specialization.*