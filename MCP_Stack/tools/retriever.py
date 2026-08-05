"""
Hybrid Retriever Module.

Performs hybrid search combining semantic similarity (ChromaDB) and keyword
matching (BM25) with score normalization, combination, deduplication, and
LLM-based reranking.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from MCP_Stack.server_config import (
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_DIR,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
    SEMANTIC_WEIGHT,
    DEFAULT_MODEL,
)
from MCP_Stack.tools.tracing import get_tracer_callbacks, trace_retrieval

logger = logging.getLogger(__name__)

# =============================================================================
# Storage Paths (shared with ingest.py)
# =============================================================================

# Path to the persisted BM25 index (pickle file with corpus + chunk ID mapping)
BM25_INDEX_PATH = Path(KNOWLEDGE_BASE_DIR) / "bm25_index.pkl"

# ChromaDB persistent storage directory
CHROMADB_PATH = Path(KNOWLEDGE_BASE_DIR) / "chromadb"

# Name of the ChromaDB collection holding document chunk embeddings
COLLECTION_NAME = "rag_chunks"

# Minimum combined score threshold — results below this are discarded as noise
MIN_RELEVANCE_THRESHOLD = 0.05


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetrievalResult:
    """
    Result from hybrid retrieval containing chunk text, metadata, and all score components.

    Attributes:
        chunk_id: Unique chunk identifier.
        text: Full text content of the chunk.
        context_header: Auto-generated contextual summary header.
        source_path: Path to the source document.
        chunk_index: Position of this chunk within its source document.
        semantic_score: Normalized semantic similarity score [0, 1].
        bm25_score: Normalized BM25 keyword score [0, 1].
        combined_score: Weighted combination of semantic and BM25 scores.
        rerank_score: LLM-assigned relevance score [0, 1] (None if reranking skipped).
    """

    chunk_id: str
    text: str
    context_header: str
    source_path: str
    chunk_index: int
    semantic_score: float
    bm25_score: float
    combined_score: float
    rerank_score: Optional[float] = None


# =============================================================================
# Score Normalization
# =============================================================================


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Apply min-max normalization to a list of scores.

    Maps scores to the [0, 1] range where the minimum input score maps to 0
    and the maximum input score maps to 1.

    If all scores are equal (max == min), returns all 0.0 values.
    For a single-element list, returns [0.0] (no spread to normalize).
    For an empty list, returns an empty list.

    Args:
        scores: List of raw scores to normalize.

    Returns:
        List of normalized scores in [0, 1].
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


# =============================================================================
# Hybrid Search (Public API)
# =============================================================================


def hybrid_search(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    semantic_weight: float = SEMANTIC_WEIGHT,
) -> list[RetrievalResult]:
    """
    Execute hybrid search: semantic + BM25, normalize, combine, rerank.

    Algorithm:
      1. Embed query using nomic-embed-text via Ollama
      2. Query ChromaDB for top 2*top_k semantic results
      3. Tokenize query and search BM25 index for top 2*top_k keyword results
      4. Normalize scores from each source to [0, 1]
      5. Combine: final = semantic_weight * sem_norm + (1 - semantic_weight) * bm25_norm
      6. Deduplicate by chunk_id keeping highest score
      7. Rerank top candidates
      8. Return top_k results with chunk text, metadata, and all score components

    Args:
        query: User query string.
        top_k: Number of results to return.
        semantic_weight: Weight for semantic scores (1 - weight = BM25 weight).

    Returns:
        Ranked list of RetrievalResult with chunk text, metadata, and scores.
        Empty list if no relevant chunks are found above minimum threshold.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return []

    # Fetch candidate count (over-retrieve for better combination)
    n_candidates = 2 * top_k

    # Step 1 & 2: Semantic search via ChromaDB
    semantic_results = _semantic_search(query, n_candidates)

    # Step 3: BM25 keyword search
    bm25_results = _bm25_search(query, n_candidates)

    # If both searches returned nothing, no relevant documents exist
    if not semantic_results and not bm25_results:
        logger.info("No results from either semantic or BM25 search for query: %s", query[:100])
        return []

    # Step 4: Normalize scores
    semantic_scores_raw = [r["score"] for r in semantic_results]
    bm25_scores_raw = [r["score"] for r in bm25_results]

    semantic_scores_norm = normalize_scores(semantic_scores_raw)
    bm25_scores_norm = normalize_scores(bm25_scores_raw)

    # Attach normalized scores back to results
    for i, result in enumerate(semantic_results):
        result["norm_score"] = semantic_scores_norm[i]

    for i, result in enumerate(bm25_results):
        result["norm_score"] = bm25_scores_norm[i]

    # Step 5: Combine scores into a unified candidate map
    candidates: dict[str, dict] = {}

    for result in semantic_results:
        chunk_id = result["chunk_id"]
        candidates[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "context_header": result.get("context_header", ""),
            "source_path": result.get("source_path", ""),
            "chunk_index": result.get("chunk_index", 0),
            "semantic_score": result["norm_score"],
            "bm25_score": 0.0,
        }

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        if chunk_id in candidates:
            # Chunk found in both — keep the higher BM25 score
            candidates[chunk_id]["bm25_score"] = max(
                candidates[chunk_id]["bm25_score"], result["norm_score"]
            )
        else:
            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "context_header": result.get("context_header", ""),
                "source_path": result.get("source_path", ""),
                "chunk_index": result.get("chunk_index", 0),
                "semantic_score": 0.0,
                "bm25_score": result["norm_score"],
            }

    # Step 6: Compute combined score and filter by threshold
    for cid, candidate in candidates.items():
        candidate["combined_score"] = (
            semantic_weight * candidate["semantic_score"]
            + (1 - semantic_weight) * candidate["bm25_score"]
        )

    # Filter below minimum relevance threshold
    filtered = [c for c in candidates.values() if c["combined_score"] >= MIN_RELEVANCE_THRESHOLD]

    if not filtered:
        logger.info("No results above minimum relevance threshold for query: %s", query[:100])
        return []

    # Sort by combined score descending
    filtered.sort(key=lambda c: c["combined_score"], reverse=True)

    # Trim to top candidates for reranking
    top_candidates = filtered[: max(top_k * 2, top_k)]

    # Step 7: Rerank
    reranked = _rerank(query, top_candidates)

    # Step 8: Return top_k results
    final_results = reranked[:top_k]

    # Trace the retrieval operation for observability
    trace_retrieval(
        query=query,
        results=final_results,
        top_k=top_k,
        semantic_weight=semantic_weight,
    )

    return [
        RetrievalResult(
            chunk_id=r["chunk_id"],
            text=r["text"],
            context_header=r["context_header"],
            source_path=r["source_path"],
            chunk_index=r["chunk_index"],
            semantic_score=r["semantic_score"],
            bm25_score=r["bm25_score"],
            combined_score=r["combined_score"],
            rerank_score=r.get("rerank_score"),
        )
        for r in final_results
    ]


# =============================================================================
# Internal: Semantic Search
# =============================================================================


def _semantic_search(query: str, n_results: int) -> list[dict]:
    """
    Perform semantic similarity search against ChromaDB.

    Embeds the query using OllamaEmbeddings and queries the ChromaDB
    collection for the closest vectors.
    """
    try:
        import chromadb
        from langchain_ollama import OllamaEmbeddings

        # Initialize embedding model
        embeddings_model = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model=EMBEDDING_MODEL,
        )

        # Embed the query
        query_embedding = embeddings_model.embed_query(query)

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return []

        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        # Convert ChromaDB distances to similarity scores
        # ChromaDB cosine distance: distance = 1 - similarity
        # So similarity = 1 - distance
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else [""] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            distances = results["distances"][0] if results["distances"] else [1.0] * len(ids)

            for i, chunk_id in enumerate(ids):
                similarity = 1.0 - distances[i]  # Convert distance to similarity
                meta = metadatas[i] if i < len(metadatas) else {}
                search_results.append({
                    "chunk_id": chunk_id,
                    "text": documents[i] if i < len(documents) else "",
                    "context_header": meta.get("context_header", ""),
                    "source_path": meta.get("source_path", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": similarity,
                })

        return search_results

    except Exception as e:
        logger.error("Semantic search failed: %s", e)
        return []


# =============================================================================
# Internal: BM25 Search
# =============================================================================


def _bm25_search(query: str, n_results: int) -> list[dict]:
    """
    Perform BM25 keyword search against the persisted BM25 index.

    Tokenizes the query and scores all documents in the BM25 corpus,
    returning the top n_results.
    """
    try:
        from rank_bm25 import BM25Okapi

        # Load BM25 index from disk
        if not BM25_INDEX_PATH.exists():
            logger.warning("BM25 index not found at %s", BM25_INDEX_PATH)
            return []

        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)

        corpus = data.get("corpus", [])
        chunk_ids = data.get("chunk_ids", [])

        if not corpus:
            return []

        # Build BM25 index from corpus
        bm25 = BM25Okapi(corpus)

        # Tokenize query
        query_tokens = query.lower().split()

        if not query_tokens:
            return []

        # Get scores for all documents
        scores = bm25.get_scores(query_tokens)

        # Get top N indices by score
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        # Load chunk texts from ChromaDB for the top results
        chunk_texts = _load_chunk_texts([chunk_ids[i] for i in scored_indices])

        results = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                continue  # Skip zero-score documents

            chunk_id = chunk_ids[idx]
            chunk_data = chunk_texts.get(chunk_id, {})

            results.append({
                "chunk_id": chunk_id,
                "text": chunk_data.get("text", ""),
                "context_header": chunk_data.get("context_header", ""),
                "source_path": chunk_data.get("source_path", ""),
                "chunk_index": chunk_data.get("chunk_index", 0),
                "score": float(scores[idx]),
            })

        return results

    except Exception as e:
        logger.error("BM25 search failed: %s", e)
        return []


def _load_chunk_texts(chunk_ids: list[str]) -> dict[str, dict]:
    """
    Load chunk texts and metadata from ChromaDB for given chunk IDs.

    Used by the BM25 search to retrieve full text for top-scoring chunks
    (BM25 only stores tokenized corpus, not original text).

    Args:
        chunk_ids: List of chunk IDs to retrieve from ChromaDB.

    Returns:
        Dict mapping chunk_id -> {text, context_header, source_path, chunk_index}.
    """
    if not chunk_ids:
        return {}

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        results = collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )

        chunk_map = {}
        if results and results["ids"]:
            for i, cid in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] and i < len(results["metadatas"]) else {}
                doc = results["documents"][i] if results["documents"] and i < len(results["documents"]) else ""
                chunk_map[cid] = {
                    "text": doc,
                    "context_header": meta.get("context_header", ""),
                    "source_path": meta.get("source_path", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                }

        return chunk_map

    except Exception as e:
        logger.error("Failed to load chunk texts from ChromaDB: %s", e)
        return {}


# =============================================================================
# Internal: Reranking
# =============================================================================


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Rerank candidates using LLM-based relevance scoring.

    For each candidate, asks the LLM to rate relevance on a 0-1 scale.
    Falls back to combined_score ordering if LLM reranking fails.

    Args:
        query: The user's original query.
        candidates: List of candidate dicts with text and scores.

    Returns:
        Candidates sorted by rerank_score (descending).
    """
    try:
        from langchain_ollama import ChatOllama

        callbacks = get_tracer_callbacks()
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=DEFAULT_MODEL,
            temperature=0.0,
            callbacks=callbacks,
        )

        for candidate in candidates:
            prompt = (
                "Rate the relevance of the following passage to the query on a scale of 0 to 1, "
                "where 0 means completely irrelevant and 1 means perfectly relevant. "
                "Respond with ONLY a number between 0 and 1.\n\n"
                f"Query: {query}\n\n"
                f"Passage: {candidate['text'][:500]}\n\n"
                "Relevance score:"
            )

            try:
                response = llm.invoke(prompt)
                score_text = response.content.strip()
                # Parse the score from the response
                score = float(score_text.split()[0])
                candidate["rerank_score"] = max(0.0, min(1.0, score))
            except (ValueError, IndexError, AttributeError):
                # If parsing fails, use combined_score as fallback
                candidate["rerank_score"] = candidate["combined_score"]

        # Sort by rerank_score descending
        candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

    except Exception as e:
        logger.warning("LLM reranking failed, falling back to combined score ordering: %s", e)
        # Fallback: keep combined_score ordering
        for candidate in candidates:
            candidate["rerank_score"] = candidate["combined_score"]

    return candidates
