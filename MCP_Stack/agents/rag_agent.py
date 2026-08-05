"""
RAG Agent Module.

Orchestrates retrieval and answer generation with citations using LangGraph.
Implements a stateful graph workflow: Retrieve → Check Context → Generate/No Context.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from MCP_Stack.server_config import (
    DEFAULT_MODEL,
    ENABLE_RAGAS_EVAL,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
    SEMANTIC_WEIGHT,
    TEMPERATURE,
)
from MCP_Stack.tools.retriever import RetrievalResult, hybrid_search
from MCP_Stack.tools.tracing import get_tracer_callbacks

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Citation:
    """
    Citation reference linking a claim in the generated answer to a source chunk.

    Attributes:
        source_path: File path of the source document.
        document_title: Human-readable title derived from the filename.
        chunk_index: Zero-based index of the chunk within the document.
        relevant_passage: Snippet of the chunk text supporting the citation.
    """

    source_path: str
    document_title: str
    chunk_index: int
    relevant_passage: str


@dataclass
class RAGResponse:
    """
    Complete RAG response returned to the client.

    Attributes:
        answer: The generated natural language answer.
        citations: List of citations referencing source chunks.
        evaluation_scores: Optional RAGAS evaluation scores (if enabled).
        metadata: Additional info (model, temperature, chunk count, etc.).
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    evaluation_scores: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


# =============================================================================
# LangGraph State
# =============================================================================


class RAGState(TypedDict):
    """State dictionary for the RAG agent graph."""

    question: str
    retrieved_chunks: list[RetrievalResult]
    answer: str
    citations: list[Citation]
    evaluation_scores: Optional[dict]


# =============================================================================
# Graph Nodes
# =============================================================================


def retrieve(state: RAGState) -> dict:
    """
    Retrieve relevant chunks using hybrid search.

    Invokes the hybrid retriever with the user's question to obtain
    context chunks from the knowledge base.
    """
    question = state["question"]
    logger.info("Retrieving context for question: %s", question[:100])

    try:
        chunks = hybrid_search(
            query=question,
            top_k=RETRIEVAL_TOP_K,
            semantic_weight=SEMANTIC_WEIGHT,
        )
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        chunks = []

    return {"retrieved_chunks": chunks}


def check_context(state: RAGState) -> str:
    """
    Conditional edge: route to 'generate' if context exists, else 'no_context'.

    Returns:
        'generate' if retrieved_chunks is non-empty, 'no_context' otherwise.
    """
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        return "generate"
    return "no_context"


def generate(state: RAGState) -> dict:
    """
    Generate an answer from retrieved context using ChatOllama.

    Builds a prompt with the retrieved chunks as context, invokes the LLM,
    and parses the response to extract the answer and citations.
    """
    from langchain_ollama import ChatOllama

    question = state["question"]
    chunks = state["retrieved_chunks"]

    # Build context block from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        doc_title = _derive_document_title(chunk.source_path)
        header = f"[{doc_title}]"
        context_parts.append(f"{header}\n{chunk.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    # Build the prompt
    prompt = _build_prompt(question, context_block, chunks)

    logger.info("Generating answer with %d context chunks", len(chunks))

    try:
        callbacks = get_tracer_callbacks()
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
            callbacks=callbacks,
        )

        response = llm.invoke(prompt)
        answer_text = response.content.strip()
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        answer_text = "I encountered an error while generating a response. Please try again."

    # Parse citations from the answer
    citations = _extract_citations(answer_text, chunks)

    # Run RAGAS evaluation if enabled
    evaluation_scores = None
    if ENABLE_RAGAS_EVAL:
        try:
            from MCP_Stack.agents.evaluator_agent import evaluate_response

            contexts = [chunk.text for chunk in chunks]
            eval_result = evaluate_response(
                question=question,
                answer=answer_text,
                contexts=contexts,
            )
            evaluation_scores = {
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
            }
            if eval_result.context_recall is not None:
                evaluation_scores["context_recall"] = eval_result.context_recall
            logger.info("RAGAS evaluation: %s", evaluation_scores)
        except Exception as e:
            logger.warning("RAGAS evaluation failed (non-fatal): %s", e)

    return {
        "answer": answer_text,
        "citations": citations,
        "evaluation_scores": evaluation_scores,
    }


def no_context_response(state: RAGState) -> dict:
    """
    Return a response indicating no relevant information was found.

    This node is invoked when the retriever returns no relevant chunks,
    ensuring the agent does not hallucinate an answer.
    """
    logger.info("No relevant context found for question: %s", state["question"][:100])

    return {
        "answer": (
            "I couldn't find relevant information in the knowledge base to answer "
            "your question. Please ensure the relevant documents have been ingested, "
            "or try rephrasing your question."
        ),
        "citations": [],
    }


# =============================================================================
# Prompt Construction
# =============================================================================


def _build_prompt(question: str, context_block: str, chunks: list[RetrievalResult]) -> str:
    """
    Build the generation prompt with context and instructions.

    Args:
        question: The user's question.
        context_block: Formatted context from retrieved chunks.
        chunks: The retrieved chunks (for source reference).

    Returns:
        The complete prompt string for the LLM.
    """
    source_list = "\n".join(
        f"- [{_derive_document_title(chunk.source_path)}]: {chunk.source_path}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""You are a helpful document Q&A assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information to fully answer the question, say so clearly.

When citing information, reference the source document by its name in square brackets, e.g. [Document Name].

## Available Sources
{source_list}

## Context
{context_block}

## Question
{question}

## Instructions
1. Answer the question using ONLY the information from the context above.
2. Include citations in your answer using the document name in square brackets, e.g. [Document Name], for each claim.
3. If the context doesn't contain the answer, state that clearly.
4. Be concise and accurate.

## Answer"""

    return prompt


# =============================================================================
# Citation Extraction
# =============================================================================


def _extract_citations(answer_text: str, chunks: list[RetrievalResult]) -> list[Citation]:
    """
    Extract citations from the generated answer text.

    Looks for [Document Name] references in the answer and maps them back
    to the original retrieved chunks. Also handles legacy [Source N] format.

    Args:
        answer_text: The generated answer containing citation markers.
        chunks: The original retrieved chunks.

    Returns:
        List of Citation objects for referenced sources.
    """
    citations = []
    seen_paths = set()

    # Build a mapping of document titles to chunks
    title_to_chunks = {}
    for chunk in chunks:
        title = _derive_document_title(chunk.source_path)
        if title not in title_to_chunks:
            title_to_chunks[title] = chunk

    # Find all [Something] references in the answer (document name citations)
    pattern = re.compile(r"\[([^\]]+)\]")
    matches = pattern.finditer(answer_text)

    for match in matches:
        ref_text = match.group(1).strip()

        # Skip common markdown patterns that aren't citations
        if ref_text.startswith("http") or ref_text in {"x", "X", " ", ""} or ref_text.startswith("!"):
            continue

        # Try legacy [Source N] format
        source_match = re.match(r"Source\s+(\d+)", ref_text)
        if source_match:
            source_num = int(source_match.group(1))
            chunk_idx = source_num - 1
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                if chunk.source_path not in seen_paths:
                    seen_paths.add(chunk.source_path)
                    relevant_passage = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
                    citations.append(
                        Citation(
                            source_path=chunk.source_path,
                            document_title=_derive_document_title(chunk.source_path),
                            chunk_index=chunk.chunk_index,
                            relevant_passage=relevant_passage,
                        )
                    )
            continue

        # Match against known document titles
        matched_chunk = title_to_chunks.get(ref_text)
        if matched_chunk and matched_chunk.source_path not in seen_paths:
            seen_paths.add(matched_chunk.source_path)
            relevant_passage = matched_chunk.text[:200] + ("..." if len(matched_chunk.text) > 200 else "")
            citations.append(
                Citation(
                    source_path=matched_chunk.source_path,
                    document_title=ref_text,
                    chunk_index=matched_chunk.chunk_index,
                    relevant_passage=relevant_passage,
                )
            )
            continue

        # Fuzzy match: check if any title contains the reference text or vice versa
        for title, chunk in title_to_chunks.items():
            if (ref_text.lower() in title.lower() or title.lower() in ref_text.lower()) and chunk.source_path not in seen_paths:
                seen_paths.add(chunk.source_path)
                relevant_passage = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
                citations.append(
                    Citation(
                        source_path=chunk.source_path,
                        document_title=title,
                        chunk_index=chunk.chunk_index,
                        relevant_passage=relevant_passage,
                    )
                )
                break

    # If no citations were found from the answer text, return empty list
    return citations


def _derive_document_title(source_path: str) -> str:
    """
    Derive a human-readable document title from a file path.

    Args:
        source_path: The file path string.

    Returns:
        A title derived from the filename without extension.
    """
    from pathlib import Path

    path = Path(source_path)
    # Use stem (filename without extension), replace underscores/hyphens with spaces
    title = path.stem.replace("_", " ").replace("-", " ")
    # Title-case it
    return title.title()


# =============================================================================
# LangGraph Construction
# =============================================================================


def _build_rag_graph() -> StateGraph:
    """
    Build the RAG agent LangGraph.

    Graph structure:
        Start → retrieve → check_context → generate (if context) → End
                                         → no_context (if no context) → End
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_context", no_context_response)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Add conditional edge from retrieve based on context check
    graph.add_conditional_edges(
        "retrieve",
        check_context,
        {
            "generate": "generate",
            "no_context": "no_context",
        },
    )

    # Both generate and no_context lead to END
    graph.add_edge("generate", END)
    graph.add_edge("no_context", END)

    return graph


# Compile the graph once at module level for reuse
_rag_graph = _build_rag_graph()
_rag_app = _rag_graph.compile()


# =============================================================================
# Public API
# =============================================================================


def ask_question(question: str) -> RAGResponse:
    """
    Process a user question through the RAG pipeline.

    This is the main entry point for the RAG agent, wired as an MCP tool.
    It invokes the LangGraph workflow to retrieve context and generate
    an answer with citations.

    Args:
        question: The user's natural language question.

    Returns:
        RAGResponse with answer, citations, and optional evaluation scores.
    """
    if not question or not question.strip():
        return RAGResponse(
            answer="Please provide a question to search the knowledge base.",
            citations=[],
            metadata={"error": "empty_question"},
        )

    logger.info("Processing question: %s", question[:100])

    # Initialize state
    initial_state: RAGState = {
        "question": question.strip(),
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "evaluation_scores": None,
    }

    try:
        # Run the graph
        final_state = _rag_app.invoke(initial_state)

        return RAGResponse(
            answer=final_state.get("answer", ""),
            citations=final_state.get("citations", []),
            evaluation_scores=final_state.get("evaluation_scores"),
            metadata={
                "chunks_retrieved": len(final_state.get("retrieved_chunks", [])),
                "model": DEFAULT_MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
        )
    except Exception as e:
        logger.error("RAG agent failed: %s", e)
        return RAGResponse(
            answer="An error occurred while processing your question. Please try again.",
            citations=[],
            metadata={"error": str(e)},
        )
