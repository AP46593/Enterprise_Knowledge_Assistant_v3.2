"""
MCP Server Entry Point.

Initializes the FastMCP server with Streamable HTTP transport, registers all
tools and agents, runs startup ingestion, and serves on the configured port.

The server:
- Injects truststore SSL certificates as its first operation
- Registers tools from filesystem, ingest, retriever, doc_loader, and chunker modules
- Registers agents: RAG Agent, Summarizer Agent, PII Redactor, Evaluator Agent
- Calls startup_ingest() on startup to auto-ingest new/modified documents
- Catches unhandled exceptions in tool execution and returns structured errors
- Logs all operations as structured JSONL to Server_Logs/

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 6.1, 12.3
"""

# FIRST OPERATION: Inject truststore SSL certificates so all subsequent
# HTTPS connections (to Ollama, external APIs) trust the system cert store.
import truststore

truststore.inject_into_ssl()

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from fastmcp import FastMCP

from MCP_Stack.server_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    KNOWLEDGE_SOURCE_DIR,
    MCP_SERVER_ENDPOINT,
    MCP_SERVER_PORT,
    SERVER_LOGS_DIR,
)

# Load environment variables from MCP_Stack/.env for secret values
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=str(_env_path))


# =============================================================================
# Structured JSONL Logging
# =============================================================================


class JSONLHandler(logging.Handler):
    """
    Logging handler that writes structured JSONL to Server_Logs/.

    Each log record is serialized as a single JSON line with timestamp,
    level, logger name, message, and optional exception traceback.
    A new log file is created per server session (timestamped filename).
    """

    def __init__(self, log_dir: str):
        """
        Initialize the JSONL handler and open the log file.

        Args:
            log_dir: Directory path where session log files will be created.
        """
        super().__init__()
        os.makedirs(log_dir, exist_ok=True)
        session_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = Path(log_dir) / f"session_{session_ts}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        """
        Write a single log record as a JSON line to the log file.

        Args:
            record: The log record to serialize and persist.
        """
        try:
            entry = {
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Include full exception traceback if present
            if record.exc_info and record.exc_info[1]:
                entry["exception"] = traceback.format_exception(*record.exc_info)
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
        except Exception:
            pass  # Logging must never crash the server

    def close(self) -> None:
        """Close the underlying file handle and release resources."""
        try:
            self._file.close()
        except Exception:
            pass
        super().close()


def _setup_logging() -> None:
    """
    Configure the root logger with JSONL file and console handlers.

    Attaches both a human-readable console handler (INFO level) and a
    structured JSONL file handler (DEBUG level) for comprehensive logging.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console handler for development visibility
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(console)

    # JSONL file handler for structured persistent logs
    jsonl_handler = JSONLHandler(SERVER_LOGS_DIR)
    jsonl_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(jsonl_handler)


_setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# MCP Server Initialization
# =============================================================================

mcp = FastMCP(
    name="RAG Chat Assistant",
    instructions=(
        "A RAG-based Document Q&A Chat Assistant. "
        "Supports document ingestion, PII redaction, hybrid retrieval, "
        "question answering with citations, summarization, and evaluation."
    ),
)


# =============================================================================
# Structured Error Response
# =============================================================================


class MCPErrorResponse:
    """
    Structured error response for tool invocations.

    Wraps exception details into a consistent JSON-serializable format
    that clients can parse and display meaningfully.
    """

    def __init__(self, error_type: str, message: str, details: str = ""):
        """
        Initialize an error response.

        Args:
            error_type: The exception class name (e.g., "ValueError").
            message: Human-readable error description.
            details: Full traceback string for debugging.
        """
        self.error_type = error_type
        self.message = message
        self.details = details

    def to_dict(self) -> dict:
        """
        Serialize the error response to a dictionary.

        Returns:
            Dict with error flag, type, message, and optional details.
        """
        result = {
            "error": True,
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


def _handle_tool_error(func_name: str, error: Exception) -> dict:
    """
    Create a structured error response from an unhandled exception.

    Logs the error and wraps it into a client-friendly format. This is the
    central error handler for all MCP tool invocations.

    Args:
        func_name: Name of the tool function that raised the error.
        error: The exception that was caught.

    Returns:
        Error dict suitable for returning to the MCP client.
    """
    error_response = MCPErrorResponse(
        error_type=type(error).__name__,
        message=str(error),
        details=traceback.format_exc(),
    )
    logger.error(
        "Tool '%s' raised unhandled exception: %s: %s",
        func_name,
        type(error).__name__,
        error,
    )
    return error_response.to_dict()


# =============================================================================
# Tool Registration
# =============================================================================


def register_tools() -> None:
    """
    Register all MCP tools from filesystem, ingest, retriever, doc_loader, and chunker modules.

    Each tool is wrapped with error handling via _handle_tool_error to ensure
    unhandled exceptions are returned as structured error responses rather
    than crashing the server.
    """

    # --- Filesystem Tools ---

    @mcp.tool()
    def list_directory(path: str = "") -> dict:
        """List files and subdirectories at a path within the knowledge source folder."""
        try:
            from MCP_Stack.tools.filesystem import list_directory as _list_directory

            result = _list_directory(path)
            return {
                "path": result.path,
                "success": result.success,
                "error": result.error,
                "entries": [
                    {
                        "name": e.name,
                        "is_directory": e.is_directory,
                        "size": e.size,
                        "file_type": e.file_type,
                    }
                    for e in result.entries
                ]
                if result.success
                else [],
            }
        except Exception as e:
            return _handle_tool_error("list_directory", e)

    @mcp.tool()
    def read_file(file_path: str) -> dict:
        """Read raw content of a file within the knowledge source folder."""
        try:
            from MCP_Stack.tools.filesystem import read_file as _read_file

            result = _read_file(file_path)
            return {
                "file_path": result.file_path,
                "success": result.success,
                "content": result.content if result.success else "",
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("read_file", e)

    @mcp.tool()
    def search_files(query: str) -> dict:
        """Search for files by name or glob pattern within the knowledge source folder."""
        try:
            from MCP_Stack.tools.filesystem import search_files as _search_files

            results = _search_files(query)
            return {
                "success": True,
                "matches": [
                    {
                        "file_path": m.file_path,
                        "file_name": m.file_name,
                        "relative_path": m.relative_path,
                    }
                    for m in results
                ],
                "count": len(results),
            }
        except Exception as e:
            return _handle_tool_error("search_files", e)

    @mcp.tool()
    def get_file_info(file_path: str) -> dict:
        """Get metadata (size, modified date, type) for a file in the knowledge source folder."""
        try:
            from MCP_Stack.tools.filesystem import get_file_info as _get_file_info

            result = _get_file_info(file_path)
            return {
                "file_path": result.file_path,
                "file_name": result.file_name,
                "file_type": result.file_type,
                "size": result.size,
                "modified_date": result.modified_date.isoformat()
                if result.modified_date
                else None,
                "success": result.success,
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("get_file_info", e)

    # --- Ingest Tools ---

    @mcp.tool()
    def ingest_file(file_path: str) -> dict:
        """Manually ingest a single file into the knowledge base (force re-ingestion)."""
        try:
            from MCP_Stack.tools.ingest import ingest_file as _ingest_file

            result = _ingest_file(file_path)
            return {
                "file_path": result.file_path,
                "success": result.success,
                "chunk_count": result.chunk_count,
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("ingest_file", e)

    @mcp.tool()
    def ingest_all() -> dict:
        """Re-ingest all supported files in the knowledge source directory."""
        try:
            from MCP_Stack.tools.ingest import ingest_all as _ingest_all

            report = _ingest_all()
            return {
                "success": True,
                "total_files": report.total_files,
                "ingested": report.ingested,
                "skipped": report.skipped,
                "failed": report.failed,
                "results": [
                    {
                        "file_path": r.file_path,
                        "success": r.success,
                        "chunk_count": r.chunk_count,
                        "error": r.error,
                    }
                    for r in report.results
                ],
            }
        except Exception as e:
            return _handle_tool_error("ingest_all", e)

    @mcp.tool()
    def list_documents() -> dict:
        """List all ingested documents with their metadata from the registry."""
        try:
            from MCP_Stack.tools.ingest import list_documents as _list_documents

            entries = _list_documents()
            return {
                "success": True,
                "documents": [
                    {
                        "document_id": e.document_id,
                        "file_path": e.file_path,
                        "file_name": e.file_name,
                        "file_type": e.file_type,
                        "chunk_count": e.chunk_count,
                        "ingested_at": e.ingested_at,
                        "file_size": e.file_size,
                        "status": e.status,
                    }
                    for e in entries
                ],
                "count": len(entries),
            }
        except Exception as e:
            return _handle_tool_error("list_documents", e)

    @mcp.tool()
    def delete_document(document_id: str) -> dict:
        """Remove a document and its chunks from the knowledge base."""
        try:
            from MCP_Stack.tools.ingest import delete_document as _delete_document

            result = _delete_document(document_id)
            return {
                "document_id": result.document_id,
                "success": result.success,
                "chunks_removed": result.chunks_removed,
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("delete_document", e)

    # --- Retriever Tool ---

    @mcp.tool()
    def hybrid_search(
        query: str, top_k: int = 5, semantic_weight: float = 0.7
    ) -> dict:
        """Search the knowledge base using hybrid semantic + keyword retrieval."""
        try:
            from MCP_Stack.tools.retriever import hybrid_search as _hybrid_search

            results = _hybrid_search(query, top_k=top_k, semantic_weight=semantic_weight)
            return {
                "success": True,
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "text": r.text,
                        "context_header": r.context_header,
                        "source_path": r.source_path,
                        "chunk_index": r.chunk_index,
                        "semantic_score": r.semantic_score,
                        "bm25_score": r.bm25_score,
                        "combined_score": r.combined_score,
                        "rerank_score": r.rerank_score,
                    }
                    for r in results
                ],
                "count": len(results),
            }
        except Exception as e:
            return _handle_tool_error("hybrid_search", e)

    # --- Document Loader Tool (direct access) ---

    @mcp.tool()
    def load_document(file_path: str) -> dict:
        """Load and extract text content from a document file."""
        try:
            from MCP_Stack.tools.doc_loader import load_document as _load_document

            result = _load_document(file_path)
            return {
                "file_path": file_path,
                "success": result.success,
                "text": result.text[:5000] if result.success else "",
                "text_length": len(result.text) if result.success else 0,
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("load_document", e)

    # --- Chunker Tool (direct access) ---

    @mcp.tool()
    def chunk_text(
        text: str,
        source_path: str = "unknown",
        title: str = "Untitled",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> dict:
        """Split text into semantically coherent chunks with metadata."""
        try:
            from MCP_Stack.tools.chunker import (
                DocumentMetadata,
                chunk_document as _chunk_document,
            )

            metadata = DocumentMetadata(
                file_path=source_path,
                file_name=Path(source_path).name if source_path != "unknown" else "unknown",
                file_type="txt",
                file_size=len(text),
                modified_date=datetime.now(timezone.utc),
                title=title,
            )
            chunks = _chunk_document(
                text=text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return {
                "success": True,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "context_header": c.context_header,
                        "chunk_index": c.metadata.chunk_index,
                        "total_chunks": c.metadata.total_chunks,
                    }
                    for c in chunks
                ],
            }
        except Exception as e:
            return _handle_tool_error("chunk_text", e)

    logger.info("All tools registered successfully.")


# =============================================================================
# Agent Registration
# =============================================================================


def register_agents() -> None:
    """
    Register RAG Agent, Summarizer Agent, PII Redactor, and Evaluator Agent as MCP tools.

    These agents provide higher-level functionality built on top of the
    primitive tools (retriever, chunker, etc.). Each is exposed as an
    MCP tool that clients can invoke directly.
    """

    # --- RAG Agent (ask_question) ---

    @mcp.tool()
    def ask_question(question: str) -> dict:
        """Ask a question and get an answer with citations from the knowledge base."""
        try:
            from MCP_Stack.agents.rag_agent import ask_question as _ask_question

            response = _ask_question(question)
            return {
                "success": True,
                "answer": response.answer,
                "citations": [
                    {
                        "source_path": c.source_path,
                        "document_title": c.document_title,
                        "chunk_index": c.chunk_index,
                        "relevant_passage": c.relevant_passage,
                    }
                    for c in response.citations
                ],
                "evaluation_scores": response.evaluation_scores,
                "metadata": response.metadata,
            }
        except Exception as e:
            return _handle_tool_error("ask_question", e)

    # --- Summarizer Agent (summarize_document) ---

    @mcp.tool()
    def summarize_document(document_id: str) -> dict:
        """Generate a summary of a document using iterative refinement."""
        try:
            from MCP_Stack.agents.summarizer_agent import (
                summarize_document as _summarize_document,
            )

            result = _summarize_document(document_id)
            return {
                "success": True,
                "document_id": result.document_id,
                "summary": result.summary,
                "from_cache": result.from_cache,
                "chunk_count": result.chunk_count,
            }
        except Exception as e:
            return _handle_tool_error("summarize_document", e)

    # --- PII Redactor ---

    @mcp.tool()
    def redact_pii(text: str, use_llm: bool = False) -> dict:
        """Detect and redact personally identifiable information from text."""
        try:
            from MCP_Stack.agents.pii_redactor import redact as _redact

            result = _redact(text, use_llm=use_llm)
            return {
                "success": True,
                "redacted_text": result.redacted_text,
                "entity_counts": result.entity_counts,
                "total_redacted": result.total_redacted,
            }
        except Exception as e:
            return _handle_tool_error("redact_pii", e)

    # --- Evaluator Agent (evaluate_response) ---

    @mcp.tool()
    def evaluate_response(
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = None,
    ) -> dict:
        """Evaluate a RAG response using RAGAS metrics (faithfulness, relevancy, precision, recall)."""
        try:
            from MCP_Stack.agents.evaluator_agent import (
                evaluate_response as _evaluate_response,
            )

            result = _evaluate_response(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            if result is None:
                return {
                    "success": True,
                    "evaluation_disabled": True,
                    "message": "RAGAS evaluation is disabled.",
                }
            return {
                "success": True,
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_precision": result.context_precision,
                "context_recall": result.context_recall,
            }
        except Exception as e:
            return _handle_tool_error("evaluate_response", e)

    # --- Evaluator Agent (generate_ground_truth) ---

    @mcp.tool()
    def generate_ground_truth(document_name: str) -> dict:
        """Generate ground-truth Q&A evaluation data from a knowledge source document."""
        try:
            from MCP_Stack.agents.evaluator_agent import (
                generate_ground_truth as _generate_ground_truth,
            )

            result = _generate_ground_truth(document_name)
            return {
                "success": result.success,
                "source_document": result.source_document,
                "output_path": result.output_path,
                "entry_count": result.entry_count,
                "error": result.error,
            }
        except Exception as e:
            return _handle_tool_error("generate_ground_truth", e)

    logger.info("All agents registered successfully.")


# =============================================================================
# Startup Hook
# =============================================================================


def _run_startup_ingest() -> None:
    """
    Run startup ingestion to auto-ingest new/modified documents.

    Called once when the server starts. Scans the knowledge_source directory,
    compares content hashes against the registry, and ingests any new or
    modified files. Does not crash the server if ingestion fails.
    """
    try:
        from MCP_Stack.tools.ingest import startup_ingest

        logger.info("Running startup ingestion...")
        report = startup_ingest()
        logger.info(
            "Startup ingestion complete: %d total, %d ingested, %d skipped, %d failed",
            report.total_files,
            report.ingested,
            report.skipped,
            report.failed,
        )
    except Exception as e:
        logger.error("Startup ingestion failed: %s", e)
        # Do not crash the server if ingestion fails


# =============================================================================
# Server Entry Point
# =============================================================================


def main() -> None:
    """
    Start the MCP server with Streamable HTTP transport.

    Workflow:
        1. Register all tools and agents
        2. Run startup ingestion (auto-ingest new/modified documents)
        3. Start the FastMCP server listening on the configured port
    """
    logger.info("Starting RAG Chat Assistant MCP Server...")
    logger.info("Port: %d, Endpoint: %s", MCP_SERVER_PORT, MCP_SERVER_ENDPOINT)

    # Register all tools and agents with the FastMCP instance
    register_tools()
    register_agents()

    # Run startup ingestion to ensure the knowledge base is up-to-date
    _run_startup_ingest()

    # Start the server with Streamable HTTP transport (blocking call)
    logger.info("Server ready. Listening on port %d at %s", MCP_SERVER_PORT, MCP_SERVER_ENDPOINT)
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=MCP_SERVER_PORT,
        path=MCP_SERVER_ENDPOINT,
    )


if __name__ == "__main__":
    main()
