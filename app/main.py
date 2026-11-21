"""FinSight RAG - Main FastAPI application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import status
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import get_settings
from app.core.database import close_db, init_db
from app.api.routes import documents, chat, evaluation, metadata
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.rag import RAGPipeline
from app.services.embeddings import EmbeddingService
from app.services.evaluation import EvaluationService

import structlog

structlog.configure(
    level=get_settings().LOG_LEVEL,
    processors=[
        structlog.processors.JSONRenderer(),
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Initialize database
    await init_db()

    # Initialize services
    get_settings()
    document_processor = DocumentProcessor("rbi_circular")
    vector_store = VectorStoreService(None)
    embedding_service = EmbeddingService()
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=None,  # Will initialize with OpenAI
        embedder=embedding_service,
        settings=get_settings(),
    )
    evaluation_service = EvaluationService()

    # Run startup tasks
    logger.info("FinSight RAG starting up...")

    yield

    # Shutdown tasks
    await close_db()
    logger.info("FinSight RAG shutdown complete")


app = FastAPI(
    title="FinSight RAG",
    description="Banking-specialized RAG system for regulatory document intelligence",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
instrumentator = Instrumentator(
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": "FinSight RAG",
        "version": "0.1.0",
        "description": "Banking-specialized RAG system for regulatory document intelligence",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Root"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "finsight-rag",
        "version": "0.1.0",
    }


@app.exception_handler(status.HTTP_404_NOT_FOUND, status_code=404)
async def not_found(request: Request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found", "error": "not_found"},
    )


@app.exception_handler(status.HTTP_422_UNPROCESSABLE_ENTITY, status_code=422)
async def validation_error(request: Request, exc):
    """Custom validation error handler."""
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation Error", "error": "validation_error"},
    )


# Include routers
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")
app.include_router(metadata.router, prefix="/api/v1")
