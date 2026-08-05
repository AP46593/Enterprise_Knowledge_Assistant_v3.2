"""
Semantic Chunker Module.

Splits document text into semantically coherent chunks with rich metadata
and contextual headers. Respects paragraph and sentence boundaries, applies
configurable overlap between consecutive chunks, and generates unique chunk IDs.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from MCP_Stack.server_config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DocumentMetadata:
    """
    Metadata attached to a source document for chunking context.

    Attributes:
        file_path: Path to the source file.
        file_name: Filename with extension.
        file_type: Lowercase file extension without dot.
        file_size: File size in bytes.
        modified_date: Last modification timestamp.
        title: Human-readable document title.
    """

    file_path: str
    file_name: str
    file_type: str
    file_size: int
    modified_date: datetime
    title: str


@dataclass
class ChunkMetadata:
    """
    Metadata for a document chunk.

    Attributes:
        source_path: Path to the source document.
        document_title: Title of the parent document.
        chunk_index: Zero-based position of this chunk in the document.
        total_chunks: Total number of chunks in the document.
        content_hash: SHA-256 hash of the chunk text for deduplication.
    """

    source_path: str
    document_title: str
    chunk_index: int
    total_chunks: int
    content_hash: str


@dataclass
class Chunk:
    """
    A single chunk of document text with metadata and context header.

    Attributes:
        chunk_id: Unique identifier combining document hash and chunk index.
        text: The chunk's text content (may include overlap from previous chunk).
        context_header: Auto-generated summary header for retrieval context.
        metadata: Full metadata for this chunk.
    """

    chunk_id: str
    text: str
    context_header: str
    metadata: ChunkMetadata


# =============================================================================
# Public API
# =============================================================================


def chunk_document(
    text: str,
    metadata: DocumentMetadata,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split text into chunks respecting sentence/paragraph boundaries.

    Algorithm:
      1. Split text at paragraph boundaries (\\n\\n)
      2. For paragraphs exceeding chunk_size, split at sentence boundaries
      3. For sentences exceeding chunk_size, split at word boundaries
      4. Merge small segments until reaching chunk_size
      5. Apply chunk_overlap from end of previous chunk to start of next
      6. Attach metadata and generate contextual summary header per chunk

    Args:
        text: The document text to chunk.
        metadata: Document metadata for attribution.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap characters between consecutive chunks (default from config).

    Returns:
        List of Chunk objects. Empty list if text is empty.
    """
    if not text or not text.strip():
        return []

    # Ensure sane parameters
    chunk_size = max(chunk_size, 50)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    # The effective capacity for content must leave room for overlap prefix
    # on all chunks after the first. We use chunk_size as the final max
    # (including overlap), so content budget = chunk_size - overlap for chunks 2+.
    content_budget = chunk_size - chunk_overlap

    # Step 1: Split into segments respecting boundaries
    segments = _split_into_segments(text, content_budget)

    # Step 2: Merge small segments into chunks up to content_budget
    merged = _merge_segments(segments, content_budget, chunk_size)

    # Step 3: Apply overlap between consecutive chunks
    chunks_text = _apply_overlap(merged, chunk_overlap)

    # Step 4: Compute document hash for chunk IDs
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    # Step 5: Build Chunk objects with metadata
    total_chunks = len(chunks_text)
    chunks: list[Chunk] = []

    for idx, chunk_text in enumerate(chunks_text):
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        chunk_meta = ChunkMetadata(
            source_path=metadata.file_path,
            document_title=metadata.title,
            chunk_index=idx,
            total_chunks=total_chunks,
            content_hash=content_hash,
        )

        context_header = _generate_context_header(chunk_text, metadata.title, idx, total_chunks)

        chunk_id = f"{doc_hash}_{idx}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                context_header=context_header,
                metadata=chunk_meta,
            )
        )

    logger.info(
        f"Chunked document '{metadata.title}' into {total_chunks} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def _split_into_segments(text: str, chunk_size: int) -> list[str]:
    """
    Split text into segments respecting paragraph, sentence, and word boundaries.

    Priority order:
      1. Paragraph boundaries (\\n\\n)
      2. Sentence boundaries (. ! ?)
      3. Word boundaries (spaces)
    """
    # Split at paragraph boundaries
    paragraphs = re.split(r"\n\n+", text)
    segments: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= chunk_size:
            segments.append(para)
        else:
            # Paragraph too large: split at sentence boundaries
            sentences = _split_sentences(para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= chunk_size:
                    segments.append(sentence)
                else:
                    # Sentence too large: split at word boundaries
                    word_segments = _split_at_words(sentence, chunk_size)
                    segments.extend(word_segments)

    return segments


def _split_sentences(text: str) -> list[str]:
    """
    Split text at sentence boundaries while keeping the delimiter with
    the preceding sentence.

    Splits on '. ', '! ', '? ' patterns (period/exclamation/question
    followed by a space), preserving the punctuation with the sentence.
    """
    # Use a regex that splits after sentence-ending punctuation followed by space
    # Keep the delimiter with the preceding segment
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _split_at_words(text: str, chunk_size: int) -> list[str]:
    """
    Split text at word boundaries to fit within chunk_size.

    For oversized sentences that cannot be split at sentence boundaries.
    """
    words = text.split()
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 for the space separator (except for the first word)
        added_len = len(word) + (1 if current else 0)

        if current_len + added_len > chunk_size and current:
            segments.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len

    if current:
        segments.append(" ".join(current))

    return segments


def _merge_segments(segments: list[str], content_budget: int, first_chunk_budget: int) -> list[str]:
    """
    Merge small consecutive segments into larger chunks.

    The first chunk can use `first_chunk_budget` characters (no overlap prepended).
    Subsequent chunks use `content_budget` (leaves room for overlap prefix).

    Joins segments with paragraph separator (\\n\\n) when they fit together.
    """
    if not segments:
        return []

    merged: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    separator = "\n\n"
    sep_len = len(separator)

    # First chunk gets the full budget; subsequent get the reduced budget
    current_budget = first_chunk_budget

    for segment in segments:
        # Calculate length if we add this segment to current
        added_len = len(segment) + (sep_len if current_parts else 0)

        if current_len + added_len <= current_budget:
            current_parts.append(segment)
            current_len += added_len
        else:
            # Flush current buffer
            if current_parts:
                merged.append(separator.join(current_parts))
            current_parts = [segment]
            current_len = len(segment)
            # After the first chunk, use the content_budget
            current_budget = content_budget

    # Flush remaining
    if current_parts:
        merged.append(separator.join(current_parts))

    return merged


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """
    Apply overlap between consecutive chunks.

    For each chunk after the first, prepend the last `overlap` characters
    from the previous chunk to the beginning of the current chunk.
    """
    if not chunks or overlap <= 0:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        # Take the last `overlap` characters from the previous chunk
        overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
        # Prepend overlap to current chunk
        result.append(overlap_text + chunks[i])

    return result


def _generate_context_header(
    chunk_text: str,
    document_title: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    Generate a contextual summary header for a chunk using extractive method.

    Extracts the first meaningful sentence or phrase from the chunk as a
    representative summary. Falls back to a positional description if no
    suitable sentence is found.
    """
    # Extract first sentence as a representative summary
    first_sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", chunk_text, re.DOTALL)

    if first_sentence_match:
        summary = first_sentence_match.group(1).strip()
        # Truncate overly long summaries
        if len(summary) > 150:
            summary = summary[:147] + "..."
    else:
        # Use first N characters as fallback
        summary = chunk_text[:100].strip()
        if len(chunk_text) > 100:
            summary += "..."

    position = f"[{document_title} - Part {chunk_index + 1}/{total_chunks}]"
    return f"{position} {summary}"
