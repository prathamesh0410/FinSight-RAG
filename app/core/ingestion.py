"""Document ingestion and processing pipeline."""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from pathlib import Path

from fastapi import UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import verify_password
from app.models.document import Document, DocumentChunk, DocumentType, QueryLog
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.rag import RAGService
from app.services.search import SearchService
from app.services.evaluation import EvaluationService

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, db: AsyncSession, vector_store: VectorStoreService, rag_service: RAGService, search_service: SearchService, evaluation_service: EvaluationService):
        self.db = db
        self.vector_store = vector_store
        self.rag_service = rag_service
        self.search_service = search_service
        self.evaluation_service = evaluation_service

    async def upload_document(self, file: UploadFile, document_type: str) -> Document:
        """Upload and process a document."""
        # Validate file type and size
        if file.content_type not in get_settings().supported_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

        if await self._check_file_size(file):
            raise HTTPException(status_code=400, detail="File too large")

        # Save temporary file
        temp_path = await self._save_temp_file(file)

        try:
            # Process document
            document = await self._process_document(temp_path, document_type)

            # Update document status
            await self._update_document_status(document.id, "completed")

            # Return document metadata
            return document

        finally:
            # Cleanup
            await self._cleanup_temp_file(temp_path)

    async def _check_file_size(self, file: UploadFile) -> bool:
        """Check if file exceeds size limit."""
        max_size = get_settings().MAX_FILE_SIZE_MB * 1024 * 1024
        contents = await file.read(max_size + 1)
        await file.seek(0)
        return len(contents) > max_size

    async def _save_temp_file(self, file: UploadFile) -> Path:
        """Save uploaded file to temporary location."""
        temp_dir = Path("/tmp") / "finsight_ingestion"
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return file_path

    async def _cleanup_temp_file(self, file_path: Path) -> None:
        """Remove temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete temp file {file_path}: {str(e)}")

    async def _process_document(self, file_path: Path, document_type: str) -> Document:
        """Process document file and return Document model."""
        # Determine document type if not specified
        if document_type == DocumentType.OTHER:
            document_type = self._detect_document_type(file_path)

        # Create document record
        document = Document(
            filename=file.filename,
            original_filename=file.filename,
            file_size=file.size,
            content_type=file.content_type,
            document_type=document_type,
            status="processing",
            chunk_count=0,
        )

        # Save to database first
        async with self.db.begin() as session:
            document = Document(
                filename=file.filename,
                original_filename=file.filename,
                file_size=document.file_size,
                content_type=document.content_type,
                document_type=document_type,
                status="processing",
                chunk_count=0,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)

            # Process document
            processor = DocumentProcessor(document_type)
            chunks = await processor.process_document(file_path)

            # Save chunks and update document
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk.content,
                    token_count=len(chunk.text.split()),
                    embedding=chunk.embedding,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    clause_reference=chunk.clause_reference,
                    chunk_metadata=chunk.chunk_metadata,
                    bm25_tokens=chunk.bm25_tokens
                )
                session.add(db_chunk)

            document.chunk_count = len(chunks)
            document.status = "completed"
            document.processed_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(document)

            # Add to vector store
            await self.vector_store.add_documents(
                chunks=[c.content for c in chunks],
                metadata=[{
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "section_title": c.section_title,
                    "clause_reference": c.clause_reference,
                    "document_type": c.document_type
                } for c in chunks]
            )

            # Log query for evaluation
            query_log = QueryLog(
                query_text="document_upload",
                retrieved_chunk_ids=[],
                answer="",
                citations=[],
                response_time_ms=0,
                faithfulness_score=0.0,
                answer_relevance_score=0.0,
                context_precision_score=0.0,
                context_recall_score=0.0
            )
            session.add(query_log)
            await session.commit()

            return document

    def _detect_document_type(self, file_path: Path) -> str:
        """Detect document type based on file extension and content."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return DocumentType.RBI_CIRCULAR  # Most common for banking docs
        elif ext == ".docx":
            return DocumentType.LOAN_AGREEMENT
        elif ext == ".txt":
            return DocumentType.REGULATORY_GUIDELINE
        else:
            return DocumentType.OTHER

    async def _update_document_status(self, document_id: uuid.UUID, status: str) -> None:
        """Update document status in database."""
        async with self.db.begin() as session:
            await session.execute(
                update(Document).where(Document.id == document_id)
                .values(status=status, updated_at=datetime.now(UTC))
            )

    async def _update_document_status(self, document_id: uuid.UUID, status: str) -> None:
        """Update document status in database."""
        async with self.db.begin() as session:
            await session.execute(
                update(Document).where(Document.id == document_id)
                .values(status=status, updated_at=datetime.now(UTC))
            )

    async def _cleanup_temp_file(self, file_path: Path) -> None:
        """Remove temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete temp file {file_path}: {str(e)}")

    async def _process_document(self, file_path: Path, document_type: str) -> list[DocumentChunk]:
        """Process document file and return list of DocumentChunk objects."""
        # Initialize processor
        processor = DocumentProcessor(document_type)

        # Extract text from file
        text_content = await self._extract_text(file_path)

        # Parse and chunk text
        chunks = await processor.process_document(text_content)

        return chunks

    async def _extract_text(self, file_path: Path) -> str:
        """Extract text content from file based on type."""
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return await self._extract_pdf_text(file_path)
        elif ext == ".docx":
            return await self._extract_docx_text(file_path)
        elif ext == ".txt":
            return await self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        from pdfplumber import pdfplumber
        text_content = ""

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n\n"

        return text_content.strip()

    async def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        import docx
        doc = docx.Document(file_path)
        text_content = ""

        for para in doc.paragraphs:
            text_content += para.text + "\n\n"

        return text_content.strip()

    async def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    async def _update_document_status(self, document_id: uuid.UUID, status: str) -> None:
        """Update document status in database."""
        async with self.db.begin() as session:
            await session.execute(
                update(Document).where(Document.id == document_id)
                .values(status=status, updated_at=datetime.now(UTC))
            )