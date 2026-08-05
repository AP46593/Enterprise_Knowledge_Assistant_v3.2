"""
Opik Tracing Module.

Provides centralized OpikTracer callback management for the RAG Chat Assistant.
Conditionally enables tracing based on the ENABLE_OPIK_TRACING config flag and
ensures tracing failures do not break normal operation (graceful degradation).

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env for secrets (OPIK_API_KEY, OPIK_WORKSPACE, OPIK_PROJECT_NAME)
load_dotenv()

# Tracing configuration loaded from environment
ENABLE_OPIK_TRACING = os.getenv("ENABLE_OPIK_TRACING", "false").lower() == "true"
OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "")
OPIK_PROJECT_NAME = os.getenv("OPIK_PROJECT_NAME", "rag-chat-assistant")

# Module-level tracer singleton (initialized lazily)
_tracer_instance = None
_tracer_initialized = False


def _initialize_tracer():
    """
    Initialize the OpikTracer singleton if tracing is enabled and credentials are available.

    This function is called lazily on first use. If initialization fails,
    tracing is silently disabled (graceful degradation).
    """
    global _tracer_instance, _tracer_initialized

    if _tracer_initialized:
        return

    _tracer_initialized = True

    if not ENABLE_OPIK_TRACING:
        logger.debug("Opik tracing is disabled (ENABLE_OPIK_TRACING=false)")
        return

    if not OPIK_API_KEY:
        logger.warning(
            "Opik tracing is enabled but OPIK_API_KEY is not set. Tracing disabled."
        )
        return

    try:
        from opik.integrations.langchain import OpikTracer

        _tracer_instance = OpikTracer(
            tags=["rag-chat-assistant"],
            metadata={
                "workspace": OPIK_WORKSPACE,
                "project": OPIK_PROJECT_NAME,
            },
        )
        logger.info(
            "Opik tracing initialized — workspace=%s, project=%s",
            OPIK_WORKSPACE,
            OPIK_PROJECT_NAME,
        )
    except ImportError:
        logger.warning(
            "opik package not installed. Tracing disabled. "
            "Install with: pip install opik"
        )
    except Exception as e:
        logger.warning("Failed to initialize Opik tracer: %s. Tracing disabled.", e)


def get_opik_tracer():
    """
    Get the OpikTracer callback instance if tracing is enabled.

    Returns:
        The OpikTracer instance, or None if tracing is disabled or unavailable.
    """
    _initialize_tracer()
    return _tracer_instance


def get_tracer_callbacks() -> list:
    """
    Get a list of LangChain callbacks including the OpikTracer (if enabled).

    This is the primary interface for agents and tools to obtain tracing callbacks.
    Returns an empty list if tracing is disabled, ensuring no impact on normal
    operation.

    Returns:
        List of callback instances. Empty if tracing is disabled.
    """
    tracer = get_opik_tracer()
    if tracer is not None:
        return [tracer]
    return []


def is_tracing_enabled() -> bool:
    """
    Check if Opik tracing is currently active.

    Returns:
        True if tracing is enabled and the tracer is initialized successfully.
    """
    _initialize_tracer()
    return _tracer_instance is not None


def trace_retrieval(query: str, results: list, top_k: int, semantic_weight: float) -> None:
    """
    Log a retrieval operation to Opik as a trace span.

    Records the query, number of results returned, top_k setting, and
    semantic_weight for observability. Failures are silently caught to
    ensure retrieval operations are not affected.

    Args:
        query: The user's search query.
        results: List of retrieval results returned.
        top_k: The top_k parameter used.
        semantic_weight: The semantic weight used for hybrid scoring.
    """
    if not is_tracing_enabled():
        return

    try:
        from opik import track

        @track(name="hybrid_retrieval")
        def _log_retrieval(query, num_results, top_k, semantic_weight, result_ids):
            return {
                "query": query,
                "num_results": num_results,
                "top_k": top_k,
                "semantic_weight": semantic_weight,
                "result_chunk_ids": result_ids,
            }

        result_ids = []
        for r in results:
            if hasattr(r, "chunk_id"):
                result_ids.append(r.chunk_id)
            elif isinstance(r, dict):
                result_ids.append(r.get("chunk_id", "unknown"))

        _log_retrieval(
            query=query,
            num_results=len(results),
            top_k=top_k,
            semantic_weight=semantic_weight,
            result_ids=result_ids[:10],  # Limit to first 10 IDs
        )

    except Exception as e:
        logger.debug("Failed to trace retrieval operation: %s", e)
