"""Document management API routes."""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.core.config import get_settings
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService
from app.services.rag import RAGPipeline
from app.services.embeddings import EmbeddingService
from app.core.security import verify_password

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload and process a regulatory document."""
    settings = get_settings()

    # Validate file type
    if file.content_type not in settings.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {settings.supported_extensions}"
        )

    # Validate file size
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    contents = await file.read(max_size + 1)
    await file.seek(0)
    if len(contents) > max_size:
        raise HTTPException(status_code=400, detail="File too large")

    # Process document
    processor = DocumentProcessor(document_type="rbi_circular")
    text_content = await processor._extract_pdf_text(file)
    chunks = processor.process_document(text_content)

    # Generate embeddings
    embedding_service = EmbeddingService()
    embeddings = await embedding_service.embed_docots([c.content for c in chunks])

    # Store in vector store
    vector_store = VectorStoreService(db)
    await vector_store.add_document_chunks(
        chunks=[{"content": c.content, "token_count": len(c.content.split())} for c in chunks],
        embeddings=embeddings,
        document_id=uuid.uuid4()
    )

    # Log to query history
    from app.models.document import QueryLog
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
    db.add(query_log)
    await db.commit()

    return DocumentUploadResponse(
        message="Document uploaded and processed successfully",
        document_id=str(uuid.uuid4()),
        chunk_count=len(chunks)
    )