"""Vector store service using pgvector with HNSW indexes."""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.database import get_db
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Vector store operations using PostgreSQL pgvector."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str
    ) -> None:
        """Add document chunks with embeddings to vector store."""
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_model = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk["content"],
                    token_count=chunk.get("token_count", 0),
                    embedding=embedding,
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title"),
                    clause_reference=chunk.get("clause_reference"),
                    chunk_metadata=chunk.get("metadata", {}),
                    bm25_tokens=chunk.get("bm25_tokens", [])
                )
                self.db.add(chunk_model)

            await self.db.commit()
            logger.info(f"Added {len(chunks)} document chunks to vector store")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to add chunks to vector store: {str(e)}")
            raise

    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using cosine similarity."""
        try:
            # Build query with pgvector cosine distance
            query = """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.token_count,
                    dc.page_number,
                    dc.section_title,
                    dc.clause_reference,
                    dc.chunk_metadata,
                    1 - (dc.embedding <=> :query_embedding) AS similarity
                FROM document_chunks dc
                WHERE 1 - (dc.embedding <=> :query_embedding) >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """

            params = {
                "query_embedding": str(query_embedding),
                "threshold": similarity_threshold,
                "limit": top_k
            }

            if document_type:
                query += """ AND dc.document_id IN (
                    SELECT d.id FROM documents d WHERE d.document_type = :doc_type
                )"""
                params["doc_type"] = document_type

            result = await self.db.execute(text(query), params)
            rows = result.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "chunk_index": row[2],
                    "content": row[3],
                    "token_count": row[4],
                    "page_number": row[5],
                    "section_title": row[6],
                    "clause_reference": row[7],
                    "metadata": row[8] if row[8] else {},
                    "similarity": float(row[9])
                })

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise

    async def hybrid_search(
        self,
        query_embedding: List[float],
        bm25_tokens: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 and vector similarity."""
        try:
            # Get semantic results
            semantic_results = await self.semantic_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                similarity_threshold=similarity_threshold
            )

            # Get BM25 results (token overlap scoring)
            bm25_results = await self._bm25_search(
                tokens=bm25_tokens,
                top_k=top_k * 2
            )

            # Combine and re-rank
            combined = self._combine_scores(
                semantic_results,
                bm25_results,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )

            # Sort by combined score
            combined.sort(key=lambda x: x["combined_score"], reverse=True)

            return combined[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise

    async def _bm25_search(
        self,
        tokens: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """BM25-style search using token overlap."""
        if not tokens:
            return []

        try:
            # Build token matching query
            conditions = []
            for token in tokens[:10]:  # Limit tokens for performance
                conditions.append(f"dc.bm25_tokens @> '{token}'")

            if not conditions:
                return []

            token_condition = " OR ".join(conditions)

            query = f"""
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.token_count,
                    dc.page_number,
                    dc.section_title,
                    dc.clause_reference,
                    dc.chunk_metadata,
                    COUNT(*) FILTER (WHERE {conditions[0]}) as match_count
                FROM document_chunks dc
                WHERE {' OR '.join(conditions)}
                GROUP BY dc.id, dc.document_id, dc.chunk_index, dc.content,
                         dc.token_count, dc.page_number, dc.section_title,
                         dc.clause_reference, dc.chunk_metadata
                ORDER BY match_count DESC
                LIMIT :limit
            """

            result = await self.db.execute(text(query), {"limit": top_k})
            rows = result.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "chunk_index": row[2],
                    "content": row[3],
                    "token_count": row[4],
                    "page_number": row[5],
                    "section_title": row[6],
                    "clause_reference": row[7],
                    "metadata": row[8] if row[8] else {},
                    "bm25_score": float(row[9]) if row[9] else 0.0,
                    "similarity": 0.0
                })

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []

    def _combine_scores(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        bm25_weight: float,
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine semantic and BM25 scores into unified ranking."""
        # Normalize scores
        combined_map = {}

        for result in semantic_results:
            chunk_id = result["id"]
            combined_map[chunk_id] = {
                **result,
                "semantic_score": result.get("similarity", 0.0),
                "bm25_score": 0.0,
                "combined_score": result.get("similarity", 0.0) * vector_weight
            }

        for result in bm25_results:
            chunk_id = result["id"]
            if chunk_id in combined_map:
                combined_map[chunk_id]["bm25_score"] = result.get("bm25_score", 0.0)
                combined_map[chunk_id]["combined_score"] = (
                    combined_map[chunk_id]["semantic_score"] * vector_weight +
                    result.get("bm25_score", 0.0) * bm25_weight
                )
            else:
                combined_map[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "bm25_score": result.get("bm25_score", 0.0),
                    "combined_score": result.get("bm25_score", 0.0) * bm25_weight
                }

        return list(combined_map.values())

    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        try:
            query = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index)

            result = await self.db.execute(query)
            rows = result.scalars().all()

            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "chunk_index": row.chunk_index,
                    "page_number": row.page_number,
                    "section_title": row.section_title,
                    "clause_reference": row.clause_reference,
                    "metadata": row.chunk_metadata if row.chunk_metadata else {}
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get document chunks: {str(e)}")
            raise