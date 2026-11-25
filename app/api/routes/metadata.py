"""Document metadata and statistics API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from app.core.database import get_db
from app.models.document import Document, DocumentType, QueryLog
from sqlalchemy import select, func, desc

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/metadata/stats", response_model=Dict[str, Any])
async def get_document_metadata_stats(db: AsyncSession = Depends(get_db)):
    """Get metadata statistics for indexed documents."""
    try:
        result = await db.execute(
            select(
                Document.document_type,
                func.count(Document.id).label("count"),
            ).group_by(Document.document_type)
        )
        rows = result.all()

        doc_stats = {}
        for row in rows:
            doc_stats[row[0]] = int(row[1])

        return {
            "total_documents": sum(doc_stats.values()),
            "documents_by_type": doc_stats,
            "vector_indexed": True,
            "metadata_extraction_enabled": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata stats: {str(e)}")
