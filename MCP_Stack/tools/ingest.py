"""
Ingest Pipeline and Document Registry Module.

Orchestrates the full ingestion workflow: load → PII redact → chunk → embed → store.
Manages a document registry (registry.json) for change detection via content hashing.
Provides tools for startup ingestion, manual ingestion, document listing, and deletion.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11,
              14.1, 14.2, 14.3, 14.4, 14.5, 15.1, 15.2, 15.3, 15.4
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from MCP_Stack.server_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    KNOWLEDGE_SOURCE_DIR,
    OLLAMA_BASE_URL,
    PII_USE_LLM,
)
from MCP_Stack.tools.doc_loader import SUPPORTED_EXTENSIONS, load_document
from MCP_Stack.agents.pii_redactor import redact
from MCP_Stack.tools.chunker import DocumentMetadata, chunk_document

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IngestResult:
    """
    Result from ingesting a single document.

    Attributes:
        file_path: Path to the ingested file.
        success: Whether ingestion completed successfully.
        chunk_count: Number of chunks stored in the knowledge base.
        error: Error message if ingestion failed (None on success).
    """

    file_path: str
    success: bool
    chunk_count: int
    error: Optional[str] = None


@dataclass
class IngestReport:
    """
    Report from batch ingestion (startup or ingest_all).

    Attributes:
        total_files: Total number of supported files found.
        ingested: Count of files successfully ingested.
        skipped: Count of files skipped (unchanged content hash).
        failed: Count of files that failed ingestion.
        results: Per-file IngestResult list.
    """

    total_files: int
    ingested: int
    skipped: int
    failed: int
    results: list[IngestResult] = field(default_factory=list)


@dataclass
class DocumentEntry:
    """
    Entry representing an ingested document in the registry.

    Attributes:
        document_id: Short hash-based document identifier.
        file_path: Path to the source file.
        file_name: Filename with extension.
        file_type: Lowercase file extension.
        content_hash: SHA-256 hash of file contents for change detection.
        chunk_count: Number of chunks stored.
        ingested_at: ISO timestamp of last ingestion.
        file_size: File size in bytes.
        modified_date: File modification timestamp.
        status: Current status (e.g., "ingested").
    """

    document_id: str
    file_path: str
    file_name: str
    file_type: str
    content_hash: str
    chunk_count: int
    ingested_at: str
    file_size: int
    modified_date: str
    status: str


@dataclass
class DeleteResult:
    """
    Result from deleting a document from the knowledge base.

    Attributes:
        document_id: The document ID that was deleted.
        success: Whether deletion completed successfully.
        chunks_removed: Number of chunks removed from storage.
        error: Error message if deletion failed (None on success).
    """

    document_id: str
    success: bool
    chunks_removed: int
    error: Optional[str] = None


# =============================================================================
# Registry Paths
# =============================================================================

# JSON file tracking all ingested documents and their content hashes
REGISTRY_PATH = Path(KNOWLEDGE_BASE_DIR) / "registry.json"

# Pickled BM25 corpus and chunk ID mapping
BM25_INDEX_PATH = Path(KNOWLEDGE_BASE_DIR) / "bm25_index.pkl"

# ChromaDB persistent storage directory
CHROMADB_PATH = Path(KNOWLEDGE_BASE_DIR) / "chromadb"

# ChromaDB collection name for document chunk embeddings
COLLECTION_NAME = "rag_chunks"


# =============================================================================
# Module-Level Storage (Lazy Initialization)
# =============================================================================

_chromadb_client = None
_chromadb_collection = None
_bm25_corpus: list[list[str]] = []
_bm25_chunk_ids: list[str] = []
_storage_initialized = False


def _init_chromadb():
    """
    Initialize ChromaDB persistent client and collection.

    Creates the storage directory if it doesn't exist and returns
    a client/collection pair using cosine similarity.

    Returns:
        Tuple of (chromadb.PersistentClient, chromadb.Collection).
    """
    import chromadb

    os.makedirs(str(CHROMADB_PATH), exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _init_bm25():
    """
    Load BM25 index from disk or initialize empty structures.

    Returns:
        Tuple of (corpus: list of token lists, chunk_ids: list of chunk ID strings).
    """
    if BM25_INDEX_PATH.exists():
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                return data.get("corpus", []), data.get("chunk_ids", [])
        except Exception as e:
            logger.warning("Failed to load BM25 index, rebuilding: %s", e)
    return [], []


def _save_bm25(corpus: list[list[str]], chunk_ids: list[str]) -> None:
    """
    Persist BM25 corpus and chunk ID mapping to disk as a pickle file.

    Args:
        corpus: List of tokenized documents (each is a list of lowercase words).
        chunk_ids: Parallel list of chunk IDs corresponding to each corpus entry.
    """
    os.makedirs(str(BM25_INDEX_PATH.parent), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"corpus": corpus, "chunk_ids": chunk_ids}, f)


def _ensure_storage_initialized() -> None:
    """
    Lazy-initialize storage backends on first use.

    Initializes ChromaDB client/collection and loads the BM25 index from disk.
    Subsequent calls are no-ops (guarded by _storage_initialized flag).
    """
    global _chromadb_client, _chromadb_collection, _bm25_corpus, _bm25_chunk_ids, _storage_initialized

    if _storage_initialized:
        return

    _chromadb_client, _chromadb_collection = _init_chromadb()
    _bm25_corpus, _bm25_chunk_ids = _init_bm25()
    _storage_initialized = True


# =============================================================================
# Embedding Helper
# =============================================================================


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using OllamaEmbeddings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    from langchain_ollama import OllamaEmbeddings

    embeddings_model = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=EMBEDDING_MODEL,
    )
    return embeddings_model.embed_documents(texts)


# =============================================================================
# Registry Operations
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
            logger.warning("Failed to load registry, starting fresh: %s", e)
    return {"version": "1.0", "documents": {}}


def _save_registry(registry: dict) -> None:
    """
    Save the document registry to disk as formatted JSON.

    Args:
        registry: The registry dict to persist.
    """
    os.makedirs(str(REGISTRY_PATH.parent), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)


def _compute_content_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of a file's contents for change detection.

    Reads the file in 8KB blocks for memory efficiency with large files.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hash string prefixed with "sha256:" (e.g., "sha256:abc123...").
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return f"sha256:{sha256.hexdigest()}"


def _compute_document_id(file_path: str) -> str:
    """
    Generate a short document ID from the file path.

    Uses the first 8 characters of the SHA-256 hash of the file path string.

    Args:
        file_path: Path string to derive the ID from.

    Returns:
        8-character hex string document ID.
    """
    return hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:8]


def _is_supported_file(file_path: Path) -> bool:
    """
    Check if a file has a supported extension for ingestion.

    Args:
        file_path: Path object for the file to check.

    Returns:
        True if the file extension is in the SUPPORTED_EXTENSIONS map.
    """
    extension = file_path.suffix.lstrip(".").lower()
    return extension in SUPPORTED_EXTENSIONS


def _scan_knowledge_source() -> list[Path]:
    """
    Scan the knowledge_source directory for all supported files recursively.

    Returns:
        Sorted list of Path objects for files with supported extensions.
    """
    source_dir = Path(KNOWLEDGE_SOURCE_DIR)
    if not source_dir.exists():
        logger.warning("Knowledge source directory does not exist: %s", KNOWLEDGE_SOURCE_DIR)
        return []

    supported_files = []
    for file_path in source_dir.rglob("*"):
        if file_path.is_file() and _is_supported_file(file_path):
            supported_files.append(file_path)

    return sorted(supported_files)


# =============================================================================
# Chunk Storage Operations
# =============================================================================


def _store_chunks(chunks, document_id: str) -> int:
    """
    Store chunks in both ChromaDB (semantic) and BM25 (keyword) indexes.

    Generates embeddings for all chunks, adds them to ChromaDB with metadata,
    and appends tokenized text to the BM25 corpus.

    Args:
        chunks: List of Chunk objects from the chunker module.
        document_id: The document ID to associate with all chunks.

    Returns:
        Number of chunks successfully stored.
    """
    global _bm25_corpus, _bm25_chunk_ids

    _ensure_storage_initialized()

    if not chunks:
        return 0

    # Prepare data for ChromaDB
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_path": chunk.metadata.source_path,
            "document_title": chunk.metadata.document_title,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            "content_hash": chunk.metadata.content_hash,
            "context_header": chunk.context_header,
            "document_id": document_id,
        }
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = _get_embeddings(documents)

    # Store in ChromaDB
    _chromadb_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    # Store in BM25 index
    for chunk in chunks:
        tokens = chunk.text.lower().split()
        _bm25_corpus.append(tokens)
        _bm25_chunk_ids.append(chunk.chunk_id)

    # Persist BM25 index
    _save_bm25(_bm25_corpus, _bm25_chunk_ids)

    return len(chunks)


def _remove_chunks_by_document_id(document_id: str) -> int:
    """
    Remove all chunks for a document from both ChromaDB and BM25 index.

    Used during re-ingestion and document deletion to ensure stale data
    is cleaned up before new chunks are stored.

    Args:
        document_id: The document ID whose chunks should be removed.

    Returns:
        Number of chunks removed from ChromaDB.
    """
    global _bm25_corpus, _bm25_chunk_ids

    _ensure_storage_initialized()

    # Find and remove from ChromaDB
    try:
        results = _chromadb_collection.get(
            where={"document_id": document_id},
        )
        chunk_ids_to_remove = results["ids"] if results["ids"] else []
    except Exception as e:
        logger.warning("Error querying ChromaDB for deletion: %s", e)
        chunk_ids_to_remove = []

    removed_count = len(chunk_ids_to_remove)

    if chunk_ids_to_remove:
        _chromadb_collection.delete(ids=chunk_ids_to_remove)

        # Remove from BM25 index
        ids_set = set(chunk_ids_to_remove)
        new_corpus = []
        new_chunk_ids = []
        for i, cid in enumerate(_bm25_chunk_ids):
            if cid not in ids_set:
                new_corpus.append(_bm25_corpus[i])
                new_chunk_ids.append(cid)

        _bm25_corpus = new_corpus
        _bm25_chunk_ids = new_chunk_ids
        _save_bm25(_bm25_corpus, _bm25_chunk_ids)

    return removed_count


# =============================================================================
# Core Ingest Logic
# =============================================================================


def _ingest_single_file(file_path: str, registry: dict) -> IngestResult:
    """
    Execute the full ingestion pipeline for a single file.

    Pipeline: load → PII redact → chunk → embed → store

    If the file was previously ingested, old chunks are removed first.
    """
    path = Path(file_path)
    rel_path = str(path)

    try:
        # Compute content hash
        content_hash = _compute_content_hash(file_path)
        document_id = _compute_document_id(rel_path)

        # Remove old chunks if this document was previously ingested
        if rel_path in registry.get("documents", {}):
            old_doc_id = registry["documents"][rel_path].get("document_id", document_id)
            _remove_chunks_by_document_id(old_doc_id)
            logger.info("Removed old chunks for modified document: %s", path.name)

        # Step 1: Load document
        load_result = load_document(file_path)
        if not load_result.success:
            return IngestResult(
                file_path=rel_path,
                success=False,
                chunk_count=0,
                error=load_result.error or "Failed to load document",
            )

        # Step 2: PII redaction
        redaction_result = redact(load_result.text, use_llm=PII_USE_LLM)
        clean_text = redaction_result.redacted_text

        # Step 3: Chunking
        doc_metadata = DocumentMetadata(
            file_path=rel_path,
            file_name=path.name,
            file_type=path.suffix.lstrip(".").lower(),
            file_size=path.stat().st_size,
            modified_date=datetime.fromtimestamp(path.stat().st_mtime),
            title=path.stem,
        )
        chunks = chunk_document(
            text=clean_text,
            metadata=doc_metadata,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            # Document produced no chunks (empty content after redaction)
            logger.warning("No chunks produced for document: %s", path.name)
            # Still register as ingested but with 0 chunks
            _update_registry_entry(registry, rel_path, document_id, content_hash, 0, path)
            return IngestResult(file_path=rel_path, success=True, chunk_count=0)

        # Step 4: Embed and store
        chunk_count = _store_chunks(chunks, document_id)

        # Step 5: Update registry
        _update_registry_entry(registry, rel_path, document_id, content_hash, chunk_count, path)

        logger.info(
            "Ingested '%s': %d chunks stored (hash: %s)",
            path.name,
            chunk_count,
            content_hash[:20] + "...",
        )

        return IngestResult(file_path=rel_path, success=True, chunk_count=chunk_count)

    except Exception as e:
        error_msg = f"Ingestion failed for '{path.name}': {type(e).__name__}: {e}"
        logger.error(error_msg)
        return IngestResult(file_path=rel_path, success=False, chunk_count=0, error=error_msg)


def _update_registry_entry(
    registry: dict,
    file_path: str,
    document_id: str,
    content_hash: str,
    chunk_count: int,
    path: Path,
) -> None:
    """
    Update or create a registry entry for an ingested document.

    Args:
        registry: The registry dict to update (mutated in place).
        file_path: The file path key for the registry entry.
        document_id: The computed document ID.
        content_hash: SHA-256 hash of the file contents.
        chunk_count: Number of chunks stored for this document.
        path: Path object for stat() access.
    """
    stat = path.stat()
    registry.setdefault("documents", {})[file_path] = {
        "document_id": document_id,
        "file_path": file_path,
        "file_name": path.name,
        "file_type": path.suffix.lstrip(".").lower(),
        "content_hash": content_hash,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "file_size": stat.st_size,
        "modified_date": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "status": "ingested",
    }


# =============================================================================
# Public API
# =============================================================================


def startup_ingest() -> IngestReport:
    """
    Scan knowledge_source directory and ingest new/modified documents.

    Skips files whose content hash matches the registry (unchanged).
    Continues processing remaining documents if one fails.
    Reports progress per file.

    Returns:
        IngestReport with counts and per-file results.
    """
    logger.info("Starting startup ingestion scan...")

    registry = _load_registry()
    supported_files = _scan_knowledge_source()

    report = IngestReport(
        total_files=len(supported_files),
        ingested=0,
        skipped=0,
        failed=0,
    )

    for file_path in supported_files:
        rel_path = str(file_path)

        try:
            # Check if file is unchanged
            current_hash = _compute_content_hash(str(file_path))
            existing_entry = registry.get("documents", {}).get(rel_path)

            if existing_entry and existing_entry.get("content_hash") == current_hash:
                # File unchanged, skip
                logger.debug("Skipping unchanged file: %s", file_path.name)
                report.skipped += 1
                report.results.append(
                    IngestResult(file_path=rel_path, success=True, chunk_count=existing_entry.get("chunk_count", 0))
                )
                continue

            # File is new or modified, ingest it
            status = "new" if not existing_entry else "modified"
            logger.info("Ingesting %s file: %s", status, file_path.name)

            result = _ingest_single_file(str(file_path), registry)
            report.results.append(result)

            if result.success:
                report.ingested += 1
                logger.info(
                    "✓ %s: %d chunks (%s)",
                    file_path.name,
                    result.chunk_count,
                    status,
                )
            else:
                report.failed += 1
                logger.error("✗ %s: %s", file_path.name, result.error)

        except Exception as e:
            # Catch any unexpected error to continue with remaining files
            error_msg = f"Unexpected error processing '{file_path.name}': {e}"
            logger.error(error_msg)
            report.failed += 1
            report.results.append(
                IngestResult(file_path=rel_path, success=False, chunk_count=0, error=error_msg)
            )

    # Save registry after all processing
    _save_registry(registry)

    logger.info(
        "Startup ingestion complete: %d total, %d ingested, %d skipped, %d failed",
        report.total_files,
        report.ingested,
        report.skipped,
        report.failed,
    )

    return report


def ingest_file(file_path: str) -> IngestResult:
    """
    Force re-ingest a single file regardless of registry status.

    Executes the full pipeline: load → PII redact → chunk → embed → store.
    Removes old chunks if the file was previously ingested.

    Args:
        file_path: Path to the file to ingest.

    Returns:
        IngestResult with success status and chunk count.
    """
    path = Path(file_path)

    if not path.exists():
        return IngestResult(
            file_path=str(path),
            success=False,
            chunk_count=0,
            error=f"File not found: {file_path}",
        )

    if not _is_supported_file(path):
        ext = path.suffix.lstrip(".").lower()
        return IngestResult(
            file_path=str(path),
            success=False,
            chunk_count=0,
            error=f"File type .{ext} is not supported",
        )

    registry = _load_registry()
    result = _ingest_single_file(str(path), registry)
    _save_registry(registry)

    return result


def ingest_all() -> IngestReport:
    """
    Re-ingest all supported files in the knowledge_source directory.

    Ignores registry status — forces full re-ingestion of every supported file.
    Old chunks are removed before new ones are stored for each file.

    Returns:
        IngestReport with counts and per-file results.
    """
    logger.info("Starting full re-ingestion of all documents...")

    registry = _load_registry()
    supported_files = _scan_knowledge_source()

    report = IngestReport(
        total_files=len(supported_files),
        ingested=0,
        skipped=0,
        failed=0,
    )

    for file_path in supported_files:
        rel_path = str(file_path)
        logger.info("Re-ingesting: %s", file_path.name)

        result = _ingest_single_file(str(file_path), registry)
        report.results.append(result)

        if result.success:
            report.ingested += 1
            logger.info("✓ %s: %d chunks", file_path.name, result.chunk_count)
        else:
            report.failed += 1
            logger.error("✗ %s: %s", file_path.name, result.error)

    # Save registry after all processing
    _save_registry(registry)

    logger.info(
        "Full re-ingestion complete: %d total, %d ingested, %d failed",
        report.total_files,
        report.ingested,
        report.failed,
    )

    return report


def list_documents() -> list[DocumentEntry]:
    """
    Return all ingested documents with metadata from the registry.

    Returns:
        List of DocumentEntry objects for all registered documents.
    """
    registry = _load_registry()
    documents = []

    for _path, entry in registry.get("documents", {}).items():
        documents.append(
            DocumentEntry(
                document_id=entry.get("document_id", ""),
                file_path=entry.get("file_path", ""),
                file_name=entry.get("file_name", ""),
                file_type=entry.get("file_type", ""),
                content_hash=entry.get("content_hash", ""),
                chunk_count=entry.get("chunk_count", 0),
                ingested_at=entry.get("ingested_at", ""),
                file_size=entry.get("file_size", 0),
                modified_date=entry.get("modified_date", ""),
                status=entry.get("status", "unknown"),
            )
        )

    return documents


def delete_document(document_id: str) -> DeleteResult:
    """
    Remove a document and its chunks from the knowledge base.

    Removes chunks from both ChromaDB and BM25 index, then updates
    the registry to reflect the deletion.

    Args:
        document_id: The document ID to delete.

    Returns:
        DeleteResult with success status and count of chunks removed.
    """
    registry = _load_registry()

    # Find the document in the registry by document_id
    target_path = None
    for path, entry in registry.get("documents", {}).items():
        if entry.get("document_id") == document_id:
            target_path = path
            break

    if target_path is None:
        return DeleteResult(
            document_id=document_id,
            success=False,
            chunks_removed=0,
            error=f"Document not found in registry: {document_id}",
        )

    try:
        # Remove chunks from storage
        chunks_removed = _remove_chunks_by_document_id(document_id)

        # Remove from registry
        del registry["documents"][target_path]
        _save_registry(registry)

        logger.info(
            "Deleted document '%s': %d chunks removed",
            target_path,
            chunks_removed,
        )

        return DeleteResult(
            document_id=document_id,
            success=True,
            chunks_removed=chunks_removed,
        )

    except Exception as e:
        error_msg = f"Failed to delete document '{document_id}': {type(e).__name__}: {e}"
        logger.error(error_msg)
        return DeleteResult(
            document_id=document_id,
            success=False,
            chunks_removed=0,
            error=error_msg,
        )
