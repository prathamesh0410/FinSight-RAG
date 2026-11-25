"""Vector store service using SQLite with sqlite-vec extension."""
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
import sqlite3
import sqlite_vec
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.database import get_db
from app.models.document import DocumentChunk
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class SQLiteVectorStoreService:
    """Vector store operations using SQLite with sqlite-vec extension."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.settings = get_settings()
        self._conn = None

    async def _get_sqlite_connection(self):
        """Get a raw SQLite connection with sqlite-vec loaded."""
        # Get the database path from the engine URL
        db_url = str(self.db.bind.url)
        if "sqlite" in db_url:
            db_path = db_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]
        else:
            db_path = "regulations.db"

        if self._conn is None:
            self._conn = sqlite3.connect(db_path)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        return self._conn

    async def add_document_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str
    ) -> None:
        """Add document chunks with embeddings to vector store."""
        try:
            doc_uuid = uuid.UUID(document_id) if isinstance(document_id, str) else document_id
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_model = DocumentChunk(
                    document_id=doc_uuid,
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
        """Perform semantic search using cosine similarity with sqlite-vec."""
        try:
            import numpy as np

            conn = await self._get_sqlite_connection()

            # Create virtual table if not exists
            cursor = conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding float[384]
                )
            """)

            # Sync embeddings to vec table
            await self._sync_embeddings_to_vec(conn)

            # Convert query embedding to binary format for sqlite-vec
            query_embedding_binary = np.array(query_embedding, dtype=np.float32).tobytes()

            if document_type:
                # Need to join with documents table
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
                        1 - vec_distance_cosine(vc.embedding, ?) AS similarity
                    FROM document_chunks dc
                    JOIN vec_chunks vc ON dc.rowid = vc.rowid
                    JOIN documents d ON dc.document_id = d.id
                    WHERE 1 - vec_distance_cosine(vc.embedding, ?) >= ?
                    AND d.document_type = ?
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                params = (query_embedding_binary, query_embedding_binary, similarity_threshold, document_type, top_k)
            else:
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
                        1 - vec_distance_cosine(vc.embedding, ?) AS similarity
                    FROM document_chunks dc
                    JOIN vec_chunks vc ON dc.rowid = vc.rowid
                    WHERE 1 - vec_distance_cosine(vc.embedding, ?) >= ?
                    ORDER BY similarity DESC
                    LIMIT ?
                """
                params = (query_embedding_binary, query_embedding_binary, similarity_threshold, top_k)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

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
                    "metadata": json.loads(row[8]) if row[8] else {},
                    "similarity": float(row[9])
                })

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            raise

    async def _sync_embeddings_to_vec(self, conn):
        """Sync embeddings from document_chunks to vec_chunks virtual table."""
        try:
            import json
            import numpy as np

            # Get all chunks with embeddings - use rowid for vec table
            cursor = conn.execute("""
                SELECT rowid, id, embedding FROM document_chunks
                WHERE embedding IS NOT NULL
            """)
            rows = cursor.fetchall()

            for row in rows:
                rowid, chunk_id, embedding_json = row
                # Parse JSON embedding and convert to binary
                embedding_list = json.loads(embedding_json)
                embedding_binary = np.array(embedding_list, dtype=np.float32).tobytes()

                # Check if already in vec table
                check = conn.execute("SELECT rowid FROM vec_chunks WHERE rowid = ?", (rowid,))
                if not check.fetchone():
                    conn.execute(
                        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                        (rowid, embedding_binary)
                    )

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to sync embeddings to vec table: {str(e)}")

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
            conn = await self._get_sqlite_connection()

            # Build token matching query using JSON array contains
            conditions = []
            for token in tokens[:10]:
                conditions.append(f"json_each.value = '{token}'")

            if not conditions:
                return []

            # Use json_each to search in bm25_tokens array
            placeholders = ",".join(["?" for _ in tokens[:10]])
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
                    COUNT(*) as match_count
                FROM document_chunks dc, json_each(dc.bm25_tokens)
                WHERE json_each.value IN ({placeholders})
                GROUP BY dc.id, dc.document_id, dc.chunk_index, dc.content,
                         dc.token_count, dc.page_number, dc.section_title,
                         dc.clause_reference, dc.chunk_metadata
                ORDER BY match_count DESC
                LIMIT ?
            """

            params = tokens[:10] + [top_k]
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

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
                    "metadata": json.loads(row[8]) if row[8] else {},
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
            doc_uuid = uuid.UUID(document_id) if isinstance(document_id, str) else document_id
            query = select(DocumentChunk).where(
                DocumentChunk.document_id == doc_uuid
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