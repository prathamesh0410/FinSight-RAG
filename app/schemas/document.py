"""Pydantic schemas for document models."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class DocumentResponse(BaseModel):
    """Response model for document metadata."""
    id: UUID
    filename: str
    document_type: str
    status: str
    chunk_count: int
    created_at: datetime
    processed_at: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    rbi_circular_number: Optional[str] = None
    risk_category: Optional[str] = None
    tags: List[str] = []


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    document_id: UUID
    chunk_count: int
    status: str = "completed"


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    total: int
    documents: List[DocumentResponse]


class DocumentDetailResponse(BaseModel):
    """Detailed document response with chunks."""
    id: UUID
    filename: str
    document_type: str
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    effective_date: Optional[datetime] = None
    rbi_circular_number: Optional[str] = None
    jurisdiction: Optional[str] = None
    loan_type: Optional[str] = None
    risk_category: Optional[str] = None
    tags: List[str] = []


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    use_reranker: bool = True


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    response_time_ms: int
    retrieval_count: int
    context_used: int
    evaluation_metrics: Optional[Dict[str, float]] = None


class CitationResponse(BaseModel):
    """Citation response model."""
    document_id: UUID
    chunk_index: int
    content_preview: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    clause_reference: Optional[str] = None
    similarity_score: float


class EvaluationMetrics(BaseModel):
    """Evaluation metrics response model."""
    faithfulness_score: float
    answer_relevance_score: float
    context_precision_score: float
    context_recall_score: float
    overall_score: float


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    service: str = "finsight-rag"
    version: str = "0.1.0"