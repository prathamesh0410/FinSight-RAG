"""RAG (Retrieval-Augmented Generation) pipeline."""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Citation:
    """Citation for a generated answer."""
    document_id: str
    chunk_index: int
    content_preview: str
    page_number: Optional[int]
    section_title: Optional[str]
    clause_reference: Optional[str]
    similarity_score: float

    def __str__(self) -> str:
        parts = []
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.clause_reference:
            parts.append(f"Clause: {self.clause_reference}")
        if self.page_number:
            parts.append(f"Page: {self.page_number}")
        if self.similarity_score > 0:
            parts.append(f"Relevance: {self.similarity_score:.2f}")
        return " | ".join(parts) if parts else "Source document"


@dataclass
class RAGResult:
    """Complete RAG result with answer and citations."""
    answer: str
    citations: List[Citation]
    query: str
    response_time_ms: int
    retrieval_count: int
    context_used: int
    evaluation_metrics: Optional[Dict[str, float]] = None


class RAGPipeline:
    """Production RAG pipeline with retrieval, generation, and evaluation."""

    def __init__(
        self,
        vector_store,
        llm_client,
        embedder,
        settings,
        reranker=None
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    async def process_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_reranker: Optional[bool] = None
    ) -> RAGResult:
        """Process a user query through the full RAG pipeline."""
        start_time = __import__("time").time()

        top_k = top_k or self.settings.TOP_K
        similarity_threshold = similarity_threshold or self.settings.SIMILARITY_THRESHOLD
        use_reranker = use_reranker if use_reranker is not None else self.settings.USE_RERANKER

        # Step 1: Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Step 2: Retrieve relevant chunks
        retrieval_k = top_k * 2 if use_reranker else top_k
        retrieved_chunks = await self.vector_store.semantic_search(
            query_embedding=query_embedding,
            top_k=retrieval_k,
            similarity_threshold=similarity_threshold
        )

        if not retrieved_chunks:
            response_time_ms = int((__import__("time").time() - start_time) * 1000)
            return RAGResult(
                answer="I don't have enough information in the indexed documents to answer this question.",
                citations=[],
                query=query,
                response_time_ms=response_time_ms,
                retrieval_count=0,
                context_used=0
            )

        # Step 3: Rerank if enabled
        if use_reranker and self.reranker:
            retrieved_chunks = await self._rerank_chunks(
                query, retrieved_chunks
            )

        # Step 4: Build context from retrieved chunks
        context_parts = []
        citations = []
        for i, chunk in enumerate(retrieved_chunks[:top_k]):
            context_parts.append(chunk["content"])
            citation = Citation(
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
                content_preview=chunk["content"][:100] + "...",
                page_number=chunk.get("page_number"),
                section_title=chunk.get("section_title"),
                clause_reference=chunk.get("clause_reference"),
                similarity_score=chunk.get("similarity", 0.0)
            )
            citations.append(citation)

        # Step 5: Generate answer using LLM
        context_text = "\n\n".join([c["content"] for c in retrieved_chunks[:top_k]])

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context_text)

        answer = await self.llm_client.generate(system_prompt, user_prompt)

        # Step 6: Log query for evaluation
        response_time_ms = int((__import__("time").time() - start_time) * 1000)

        evaluation_metrics = await self._compute_evaluation_metrics(
            query, answer, retrieved_chunks, context_text
        )

        return RAGResult(
            answer=answer,
            citations=citations,
            query=query,
            response_time_ms=response_time_ms,
            retrieval_count=len(retrieved_chunks),
            context_used=len(retrieved_chunks),
            evaluation_metrics=evaluation_metrics
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are a banking and financial regulatory assistant.
Your role is to answer questions based strictly on the provided regulatory documents.

CRITICAL INSTRUCTIONS:
1. ONLY use information from the provided context below — do NOT use external knowledge
2. If the answer is not in the context, say "I don't have that information in the provided documents"
3. Provide inline citations after each claim using the format: [Source: RBI Circular No.XYZ §3.2(a)]
4. Format citations consistently: always include document type, number, and clause/section
5. If multiple sources support a point, list all relevant citations
6. Never hallucinate regulatory requirements or create imaginary clauses
7. Keep answers concise and professional
8. If a question involves legal advice, recommend consulting a qualified professional"""

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build the user prompt with context."""
        return f"""Context (regulatory documents):
----------------------------------------
{context}
----------------------------------------

User Question: {query}

Instructions:
- Answer based strictly on the context above
- Provide inline citations in format: [Source: RBI Circular No.XYZ §3.2(a)]
- If the answer is not in the context, say: "I don't have that information in the provided documents"
- Be concise but comprehensive"""
```