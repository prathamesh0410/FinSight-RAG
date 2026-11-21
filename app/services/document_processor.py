"""Document processing pipeline: parsing, chunking, metadata extraction."""
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from pdfplumber import pdfplumber
import docx

logger = logging.getLogger(__name__)


@dataclass
class ParsedChunk:
    """A parsed text chunk with metadata."""
    chunk_index: int
    content: str
    token_count: int
    embedding: List[float]
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    clause_reference: Optional[str] = None
    chunk_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.chunk_metadata is None:
            self.chunk_metadata = {}


class LegalAwareChunker:
    """
    Chunking strategy specialized for legal/regulatory documents.
    Preserves clause boundaries, definitions, and section hierarchy.
    """

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into chunks preserving clause boundaries."""
        chunks = []

        # First, try to identify document structure
        sections = LegalAwareChunker._identify_sections(text)

        if sections:
            # Chunk by sections if found
            for section_name, section_text in sections.items():
                section_chunks = LegalAwareChunker._chunk_by_size(
                    section_text, chunk_size, overlap
                )
                for chunk in section_chunks:
                    chunk["section_title"] = section_name
                    chunks.append(chunk)
        else:
            # Fallback to recursive chunking
            chunks = LegalAwareChunker._recursive_chunk(text, chunk_size, overlap)

        return chunks

    @staticmethod
    def _identify_sections(text: str) -> Dict[str, str]:
        """Identify major sections in a document."""
        sections = {}

        # Common section patterns in banking/regulatory docs
        section_patterns = [
            r"(?i)(Section\s+[IVX\d]+[.:]?\s*.+)",
            r"(?i)(^Article\s+\d+[.:]?\s*.+)",
            r"(?i)(^Clause\s+\d+[.:]?\s*.+)",
            r"(?i)(^[\d]+\.\s+.+)",
        ]

        lines = text.split("\n")
        current_section = ""
        current_lines = []

        for line in lines:
            is_section_header = any(
                re.match(pattern, line.strip())
                for pattern in section_patterns
            )

            if is_section_header and current_lines:
                # Save previous section
                sections[current_section] = "\n".join(current_lines)
                current_lines = [line]
                current_section = line.strip()
            elif is_section_header and not current_lines:
                current_section = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Don't forget last section
        if current_lines and current_section:
            sections[current_section] = "\n".join(current_lines)

        return sections if sections else {}

    @staticmethod
    def _chunk_by_size(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk text by approximate size with overlap."""
        words = text.split()
        if not words:
            return []

        chunks = []
        current_chunk = []
        current_size = 0
        current_index = 0

        for word in words:
            current_chunk.append(word)
            current_size += 1

            if current_size >= chunk_size:
                # Include overlap from previous chunk
                start = max(0, current_index - overlap) if current_index > overlap else 0
                chunk_words = current_chunk[start:]
                chunk_text = " ".join(chunk_words)

                chunks.append({
                    "content": chunk_text,
                    "token_count": len(chunk_words),
                    "chunk_index": current_index,
                })
                current_index += len(chunk_words)
                current_size = 0
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk

        # Handle remaining words
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "token_count": len(current_chunk),
                "chunk_index": current_index,
            })

        return chunks

    @staticmethod
    def _recursive_chunk(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Simple recursive text chunking."""
        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "content": chunk_text,
                    "token_count": len(chunk_words),
                    "chunk_index": i,
                })

        return chunks


class SimpleChunker:
    """Basic chunking without structure awareness."""

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into basic chunks."""
        words = text.split()
        if not words:
            return []

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "content": chunk_text,
                    "token_count": len(chunk_words),
                    "chunk_index": i,
                })

        return chunks


@dataclass
class ExtractedMetadata:
    """Metadata extracted from a document."""
    document_type: str
    rbi_circular_number: Optional[str] = None
    effective_date: Optional[str] = None
    jurisdiction: Optional[str] = None
    loan_type: Optional[str] = None
    risk_category: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MetadataExtractor:
    """Extract regulatory and domain metadata from documents."""

    @staticmethod
    def extract_from_pdf(text: str, filename: str) -> ExtractedMetadata:
        """Extract metadata from document text."""
        metadata = ExtractedMetadata(document_type="other")

        # Extract RBI circular numbers
        rbi_patterns = [
            r"RBI\s*[\s/.-]*Circular[\s/.-]*No\.\s*(\d+[\d/.-]*)",
            r"Circular[\s/.-]*No\.\s*(\d+[\d/.-]*)",
            r"Ref\.\s*(\d+[\d/.-]*)",
        ]
        for pattern in rbi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata.rbi_circular_number = match.group(1)
                break

        # Extract effective dates
        date_patterns = [
            r"(?i)effective[\s/.-]*(?:from|date)[:\s]+([A-Za-z\d\s]+)",
            r"(?i)date[\s/.-]*of[\s/.-]*effect[:\s]+([A-Za-z\d\s]+)",
            r"\d{2}[/-]\d{2}[/-]\d{4}",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata.effective_date = match.group(1)
                break

        # Extract jurisdiction
        jurisdiction_keywords = ["India", "Indian", "RBI", "Domestic"]
        for kw in jurisdiction_keywords:
            if kw.lower() in text.lower():
                metadata.jurisdiction = kw
                break

        # Extract loan type indicators
        loan_keywords = ["home loan", "personal loan", "car loan", "business loan", "education loan"]
        text_lower = text.lower()
        for kw in loan_keywords:
            if kw in text_lower:
                metadata.loan_type = kw
                break

        # Extract risk category
        risk_keywords = ["credit risk", "market risk", "operational risk", "liquidity risk"]
        text_lower = text.lower()
        for kw in risk_keywords:
            if kw in text_lower:
                metadata.risk_category = kw
                break

        # Extract tags from headings and key phrases
        metadata.tags = MetadataExtractor._extract_tags(text)

        return metadata

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        """Extract relevant tags from document text."""
        tags = []
        text_lower = text.lower()

        # Risk-related tags
        risk_tags = ["compliance", "regulatory", "audit", "risk", "capital adequacy"]
        for tag in risk_tags:
            if tag in text_lower:
                tags.append(tag)

        # Document-type tags
        doc_tags = ["circular", "guideline", "policy", "report", "framework"]
        for tag in doc_tags:
            if tag in text_lower:
                tags.append(tag)

        return list(set(tags))

    @classmethod
    def extract(cls, text: str, filename: str) -> ExtractedMetadata:
        """Extract all metadata from document."""
        return cls.extract_from_pdf(text, filename)