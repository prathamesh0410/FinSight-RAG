"""Document and chunk models with SQLite vector support."""
import uuid
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    Integer,
    UUID as SA_UUID,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentType:
    """Document type constants."""
    RBI_CIRCULAR = "rbi_circular"
    LOAN_AGREEMENT = "loan_agreement"
    COMPLIANCE_POLICY = "compliance_policy"
    RISK_REPORT = "risk_report"
    REGULATORY_GUIDELINE = "regulatory_guideline"
    OTHER = "other"


class Document(Base):
    """Document model for uploaded files."""
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        SA_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False, default=DocumentType.OTHER)

    # Metadata extracted from document
    rbi_circular_number: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    jurisdiction: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    loan_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    risk_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Processing status
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_documents_type", "document_type"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_rbi_circular", "rbi_circular_number"),
    )


class DocumentChunk(Base):
    """Document chunk with vector embedding."""
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        SA_UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[List[float]] = mapped_column(JSON, nullable=False)

    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    clause_reference: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    bm25_tokens: Mapped[List[str]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_document_chunk_idx", "document_id", "chunk_index"),
    )


class QueryLog(Base):
    """Log of user queries for evaluation and analytics."""
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        SA_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[List[uuid.UUID]] = mapped_column(JSON, default=list)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    faithfulness_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    answer_relevance_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    context_precision_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    context_recall_score: Mapped[Optional[float]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_query_logs_created_at", "created_at"),
    )
