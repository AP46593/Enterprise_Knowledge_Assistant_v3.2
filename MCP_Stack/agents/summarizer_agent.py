"""
Summarizer Agent Module.

Generates summaries of documents using iterative refinement over document
chunks with content-hash-based caching. Retrieves all chunks for a document
from ChromaDB, summarizes chunk-by-chunk while refining the running summary,
and caches results keyed by content hash.

Requirements: 10.1, 10.2, 10.3, 10.4
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from MCP_Stack.server_config import (
    CACHE_DIR,
    DEFAULT_MODEL,
    KNOWLEDGE_BASE_DIR,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)
from MCP_Stack.tools.tracing import get_tracer_callbacks

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SummaryResult:
    """
    Result from document summarization.

    Attributes:
        document_id: The unique identifier of the summarized document.
        summary: The generated summary text (or error message on failure).
        from_cache: Whether the summary was served from cache.
        chunk_count: Number of chunks that were summarized.
    """

    document_id: str
    summary: str
    from_cache: bool
    chunk_count: int


# =============================================================================
# Storage Paths (shared with ingest.py)
# =============================================================================

# ChromaDB persistent storage path
CHROMADB_PATH = Path(KNOWLEDGE_BASE_DIR) / "chromadb"

# Name of the ChromaDB collection for document chunks
COLLECTION_NAME = "rag_chunks"

# Path to the document registry JSON file
REGISTRY_PATH = Path(KNOWLEDGE_BASE_DIR) / "registry.json"


# =============================================================================
# Registry Helpers
# =============================================================================


def _load_registry() -> dict:
    """
    Load the document registry from disk.

    Returns:
        Registry dict with 'version' and 'documents' keys.
        Returns a fresh empty registry if the file doesn't exist or is corrupt.
    """
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load registry: %s", e)
    return {"version": "1.0", "documents": {}}


def _find_document_in_registry(document_id: str) -> Optional[dict]:
    """
    Look up a document entry by document_id in the registry.

    Args:
        document_id: The unique document identifier to search for.

    Returns:
        The registry entry dict if found, or None if not found.
    """
    registry = _load_registry()
    for _path, entry in registry.get("documents", {}).items():
        if entry.get("document_id") == document_id:
            return entry
    return None


# =============================================================================
# Cache Operations
# =============================================================================


def _get_cache_path(content_hash: str) -> Path:
    """
    Get the cache file path for a given content hash.

    Strips the 'sha256:' prefix and uses the hex digest as the filename.

    Args:
        content_hash: The full content hash string (e.g., "sha256:abc123...").

    Returns:
        Path object pointing to the cache JSON file.
    """
    # Strip the 'sha256:' prefix for filename
    hash_value = content_hash.replace("sha256:", "")
    return Path(CACHE_DIR) / f"{hash_value}.json"


def _check_cache(content_hash: str) -> Optional[str]:
    """
    Check if a cached summary exists for the given content hash.

    Args:
        content_hash: The SHA-256 content hash of the document.

    Returns:
        The cached summary string, or None if no cache entry exists.
    """
    cache_path = _get_cache_path(content_hash)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_entry = json.load(f)
            summary = cache_entry.get("summary")
            if summary:
                logger.info("Cache hit for content hash: %s", content_hash[:20])
                return summary
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read cache entry: %s", e)
    return None


def _save_cache(
    content_hash: str, document_id: str, summary: str, chunk_count: int
) -> None:
    """
    Save a summary to the cache keyed by content hash.

    Args:
        content_hash: The SHA-256 content hash of the document.
        document_id: The document identifier.
        summary: The generated summary text.
        chunk_count: Number of chunks that were summarized.
    """
    cache_path = _get_cache_path(content_hash)
    os.makedirs(str(cache_path.parent), exist_ok=True)

    cache_entry = {
        "content_hash": content_hash,
        "document_id": document_id,
        "summary": summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
    }

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2)
        logger.debug("Cached summary for hash: %s", content_hash[:20])
    except OSError as e:
        logger.warning("Failed to write cache entry: %s", e)


# =============================================================================
# ChromaDB Chunk Retrieval
# =============================================================================


def _get_chunks_for_document(document_id: str) -> list[dict]:
    """
    Retrieve all chunks for a document from ChromaDB, ordered by chunk_index.

    Args:
        document_id: The document identifier.

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys, sorted by chunk_index.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        return []

    chunks = []
    for i, chunk_id in enumerate(results["ids"]):
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            }
        )

    # Sort by chunk_index for proper ordering
    chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
    return chunks


# =============================================================================
# LLM Summarization
# =============================================================================


def _get_llm():
    """
    Create a ChatOllama instance for summarization with optional tracing.

    Returns:
        Configured ChatOllama instance ready for invocation.
    """
    from langchain_ollama import ChatOllama

    callbacks = get_tracer_callbacks()
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        callbacks=callbacks,
    )


def _summarize_first_chunk(llm, chunk_text: str) -> str:
    """
    Generate an initial summary from the first chunk.

    Args:
        llm: The ChatOllama LLM instance.
        chunk_text: Text of the first chunk.

    Returns:
        Initial summary string.
    """
    prompt = (
        "You are a document summarizer. Write a concise summary of the following "
        "text that captures the key points and main arguments.\n\n"
        f"Text:\n{chunk_text}\n\n"
        "Summary:"
    )

    response = llm.invoke(prompt)
    return response.content.strip()


def _refine_summary(llm, existing_summary: str, chunk_text: str) -> str:
    """
    Refine an existing summary with additional context from a new chunk.

    Args:
        llm: The ChatOllama LLM instance.
        existing_summary: The current running summary.
        chunk_text: Text of the new chunk to incorporate.

    Returns:
        Refined summary string.
    """
    prompt = (
        "You are a document summarizer. You have a running summary of a document "
        "and new text from the next section. Refine the summary to incorporate "
        "the new information while keeping it concise and coherent.\n\n"
        f"Current summary:\n{existing_summary}\n\n"
        f"New text:\n{chunk_text}\n\n"
        "Refined summary:"
    )

    response = llm.invoke(prompt)
    return response.content.strip()


# =============================================================================
# Public API
# =============================================================================


def summarize_document(document_id: str) -> SummaryResult:
    """
    Generate a summary of a document using iterative refinement.

    Workflow:
    1. Look up document in registry, get content hash
    2. Check cache — if hit, return cached summary
    3. Retrieve all chunks for the document from ChromaDB
    4. Apply iterative refinement: summarize first chunk, then refine
       running summary with each subsequent chunk
    5. Cache result and return

    This function is wired as an MCP tool.

    Args:
        document_id: The document identifier to summarize.

    Returns:
        SummaryResult with the summary, cache status, and chunk count.
    """
    # Step 1: Look up document in registry
    doc_entry = _find_document_in_registry(document_id)
    if doc_entry is None:
        logger.error("Document not found in registry: %s", document_id)
        return SummaryResult(
            document_id=document_id,
            summary=f"Error: Document '{document_id}' not found in the registry.",
            from_cache=False,
            chunk_count=0,
        )

    content_hash = doc_entry.get("content_hash", "")

    # Step 2: Check cache
    if content_hash:
        cached_summary = _check_cache(content_hash)
        if cached_summary is not None:
            return SummaryResult(
                document_id=document_id,
                summary=cached_summary,
                from_cache=True,
                chunk_count=doc_entry.get("chunk_count", 0),
            )

    # Step 3: Retrieve all chunks from ChromaDB
    chunks = _get_chunks_for_document(document_id)
    if not chunks:
        logger.warning("No chunks found for document: %s", document_id)
        return SummaryResult(
            document_id=document_id,
            summary=f"Error: No chunks found for document '{document_id}'. "
            "The document may not have been ingested yet.",
            from_cache=False,
            chunk_count=0,
        )

    # Step 4: Iterative refinement
    logger.info(
        "Summarizing document '%s' with %d chunks...",
        document_id,
        len(chunks),
    )

    try:
        llm = _get_llm()

        # Summarize first chunk
        summary = _summarize_first_chunk(llm, chunks[0]["text"])

        # Refine with subsequent chunks
        for chunk in chunks[1:]:
            summary = _refine_summary(llm, summary, chunk["text"])

    except Exception as e:
        logger.error("Summarization failed for document '%s': %s", document_id, e)
        return SummaryResult(
            document_id=document_id,
            summary=f"Error: Summarization failed — {type(e).__name__}: {e}",
            from_cache=False,
            chunk_count=len(chunks),
        )

    # Step 5: Cache and return
    if content_hash:
        _save_cache(content_hash, document_id, summary, len(chunks))

    logger.info(
        "Summary generated for document '%s': %d chunks processed",
        document_id,
        len(chunks),
    )

    return SummaryResult(
        document_id=document_id,
        summary=summary,
        from_cache=False,
        chunk_count=len(chunks),
    )
