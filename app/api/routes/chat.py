"""Chat/query API routes for RAG system."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import time

from app.core.database import get_db
from app.core.config import get_settings
from app.schemas.document import QueryRequest, QueryResponse
from app.services.vector_store_sqlite import SQLiteVectorStoreService
from app.services.rag import RAGPipeline
from app.services.embeddings import EmbeddingService
from app.services.document_processor import DocumentProcessor
from app.services.llm_client import LLMClient

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=QueryResponse, status_code=200)
async def process_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Process a user query through the RAG pipeline."""
    start_time = time.time()
    settings = get_settings()

    # Initialize services
    vector_store = SQLiteVectorStoreService(db)
    embedding_service = EmbeddingService()
    llm_client = LLMClient()
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
        embedder=embedding_service,
        settings=settings,
    )

    # Process query through RAG pipeline
    try:
        result = await rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            use_reranker=request.use_reranker,
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            citations=[
                {
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "content_preview": c.content_preview,
                    "page_number": c.page_number,
                    "section_title": c.section_title,
                    "clause_reference": c.clause_reference,
                    "similarity_score": c.similarity_score,
                }
                for c in result.citations
            ],
            response_time_ms=response_time_ms,
            retrieval_count=result.retrieval_count,
            context_used=result.context_used,
            evaluation_metrics=result.evaluation_metrics,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/history", tags=["Chat"])
async def get_query_history(db: AsyncSession = Depends(get_db)):
    """Get recent query history for evaluation."""
    return {"message": "Query history endpoint - to be implemented"}


@router.get("/citations/{document_id}", tags=["Chat"])
async def get_document_citations(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get citations for a specific document."""
    vector_store = SQLiteVectorStoreService(db)
    chunks = await vector_store.get_document_chunks(document_id)

    return {
        "document_id": document_id,
        "chunks": chunks,
        "total_chunks": len(chunks),
    }
