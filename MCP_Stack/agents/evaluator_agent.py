"""
Evaluator Agent Module.

Computes RAGAS evaluation metrics (faithfulness, answer relevancy,
context precision, context recall) for RAG responses using local Ollama
models. Provides ground-truth generation for evaluation datasets.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 16.10
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from MCP_Stack.server_config import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    ENABLE_RAGAS_EVAL,
    GROUND_TRUTH_MAX_QUESTIONS,
    GROUND_TRUTH_MIN_QUESTIONS,
    KNOWLEDGE_SOURCE_DIR,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    TEMPERATURE,
)
from MCP_Stack.tools.tracing import get_tracer_callbacks

logger = logging.getLogger(__name__)

# Directory where ground-truth JSON files are written and read from
TEST_DATA_DIR = str(Path(KNOWLEDGE_SOURCE_DIR).parent / "ground_truth_docs")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResult:
    """Result from RAGAS evaluation of a RAG response."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float]


@dataclass
class GroundTruthResult:
    """Result from ground-truth dataset generation."""

    source_document: str
    output_path: str
    entry_count: int
    success: bool
    error: Optional[str]


# =============================================================================
# Public API
# =============================================================================


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> Optional[EvaluationResult]:
    """
    Evaluate a RAG response using RAGAS-style metrics via LLM-as-judge.

    Computes faithfulness, answer relevancy, context precision, and
    (when ground_truth is provided) context recall using an Ollama LLM
    as the evaluator.

    This function is conditionally invoked based on the ENABLE_RAGAS_EVAL
    config flag. If evaluation is disabled, returns None.

    Args:
        question: The user's original question.
        answer: The generated answer from the RAG pipeline.
        contexts: List of context strings used to generate the answer.
        ground_truth: Optional reference answer for recall computation.

    Returns:
        EvaluationResult with metric scores, or None if evaluation is
        disabled or inputs are invalid.
    """
    if not ENABLE_RAGAS_EVAL:
        logger.debug("RAGAS evaluation is disabled (ENABLE_RAGAS_EVAL=false)")
        return None

    if not question or not answer or not contexts:
        logger.warning("Invalid inputs for evaluation: missing question, answer, or contexts")
        return None

    try:
        from langchain_ollama import ChatOllama

        callbacks = get_tracer_callbacks()
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=DEFAULT_MODEL,
            temperature=0.0,
            num_predict=MAX_TOKENS,
            callbacks=callbacks,
        )

        context_text = "\n---\n".join(contexts)

        # Faithfulness: Is the answer supported by the contexts?
        faithfulness_score = _evaluate_metric(llm, "faithfulness", f"""Rate how faithful the answer is to the provided contexts. 
A faithful answer only contains information that can be verified from the contexts.

Question: {question}
Contexts: {context_text}
Answer: {answer}

Score from 0.0 to 5.0 (5.0 = fully faithful, 0.0 = completely hallucinated).
Respond with ONLY a number between 0.0 and 5.0.""")

        # Answer Relevancy: Does the answer address the question?
        relevancy_score = _evaluate_metric(llm, "answer_relevancy", f"""Rate how relevant the answer is to the question asked.
A relevant answer directly addresses what was asked.

Question: {question}
Answer: {answer}

Score from 0.0 to 5.0 (5.0 = perfectly relevant, 0.0 = completely irrelevant).
Respond with ONLY a number between 0.0 and 5.0.""")

        # Context Precision: Are the retrieved contexts relevant to the question?
        precision_score = _evaluate_metric(llm, "context_precision", f"""Rate how relevant the retrieved contexts are to the question.
High precision means most retrieved contexts are relevant and useful.

Question: {question}
Contexts: {context_text}

Score from 0.0 to 5.0 (5.0 = all contexts relevant, 0.0 = no contexts relevant).
Respond with ONLY a number between 0.0 and 5.0.""")

        # Context Recall: Is all needed info from ground truth present in contexts?
        recall_score = None
        if ground_truth is not None:
            recall_score = _evaluate_metric(llm, "context_recall", f"""Rate how well the retrieved contexts cover the information in the reference answer.
High recall means the contexts contain all the information needed to produce the reference answer.

Question: {question}
Reference Answer: {ground_truth}
Contexts: {context_text}

Score from 0.0 to 5.0 (5.0 = contexts cover everything, 0.0 = contexts cover nothing).
Respond with ONLY a number between 0.0 and 5.0.""")

        logger.info(
            "RAGAS evaluation complete — faithfulness=%.3f, relevancy=%.3f, "
            "precision=%.3f, recall=%s",
            faithfulness_score,
            relevancy_score,
            precision_score,
            f"{recall_score:.3f}" if recall_score is not None else "N/A",
        )

        return EvaluationResult(
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            context_precision=precision_score,
            context_recall=recall_score,
        )

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return EvaluationResult(
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=None,
        )


def _evaluate_metric(llm, metric_name: str, prompt: str) -> float:
    """
    Ask the LLM to score a single metric and parse the numeric response.

    Invokes the LLM with a scoring prompt, extracts the first numeric value
    from the response, and clamps it to the [0.0, 5.0] range.

    Args:
        llm: The ChatOllama LLM instance to use for scoring.
        metric_name: Name of the metric (used in log messages on failure).
        prompt: The full scoring prompt to send to the LLM.

    Returns:
        Float score in [0.0, 5.0]. Returns 0.0 if parsing fails.
    """
    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract the first float from the response
        import re
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(5.0, score))
        logger.warning("Could not parse %s score from: %s", metric_name, text[:50])
        return 0.0
    except Exception as e:
        logger.error("Failed to evaluate %s: %s", metric_name, e)
        return 0.0


def generate_ground_truth(document_name: str) -> GroundTruthResult:
    """
    Generate a ground-truth evaluation dataset from a source document.

    Loads the specified document from knowledge_source, uses an LLM to
    generate question-answer pairs covering different sections, and
    outputs a JSON file for use in RAGAS evaluation.

    Args:
        document_name: Name of the document in the knowledge_source directory.

    Returns:
        GroundTruthResult with generation status and output details.
    """
    from langchain_ollama import ChatOllama

    from MCP_Stack.tools.doc_loader import load_document

    # 1. Resolve the full path to the document
    file_path = os.path.join(KNOWLEDGE_SOURCE_DIR, document_name)

    # 2. Check file exists
    if not os.path.isfile(file_path):
        error_msg = f"Document not found: {file_path}"
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    # 3. Load document using doc_loader
    load_result = load_document(file_path)
    if not load_result.success:
        error_msg = f"Failed to load document: {load_result.error}"
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    document_text = load_result.text
    if not document_text.strip():
        error_msg = "Document loaded but contains no text content."
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    # 4. Split the document text into sections for context extraction
    sections = _split_into_sections(document_text)

    # 5. Use ChatOllama to generate Q&A pairs from the document content
    callbacks = get_tracer_callbacks()
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=DEFAULT_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        callbacks=callbacks,
    )

    num_questions = min(max(len(sections), GROUND_TRUTH_MIN_QUESTIONS), GROUND_TRUTH_MAX_QUESTIONS)

    prompt = (
        "Given the following document text, generate exactly "
        f"{num_questions} diverse question-answer pairs that cover "
        "different sections and topics in the document. For each pair, "
        "include the relevant passage from the text that supports the answer.\n\n"
        "Output ONLY valid JSON in this exact format (no other text):\n"
        "[\n"
        '  {{"question": "...", "answer": "...", "contexts": ["relevant passage from the document"]}}\n'
        "]\n\n"
        "Rules:\n"
        "- Questions should be diverse, covering different sections of the document\n"
        "- Answers must be accurate and derived from the document text\n"
        "- Each contexts list must have at least one non-empty string that is a direct quote or close paraphrase from the document\n"
        "- Do not invent information not present in the document\n\n"
        f"Document text:\n{document_text[:8000]}"  # Limit to avoid token overflow
    )

    try:
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        error_msg = f"LLM generation failed: {e}"
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    # 6. Parse the LLM response as JSON
    entries = _parse_llm_response(response_text)
    if entries is None:
        error_msg = "Failed to parse LLM response as valid JSON entries."
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    # 7. Validate each entry has non-empty question, answer, and contexts list
    validated_entries = _validate_entries(entries)
    if not validated_entries:
        error_msg = "No valid entries after validation. All entries failed schema validation."
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    # 8. Write output JSON to ground_truth_docs directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Derive output filename from source document: "report.pdf" -> "report_ground_truth.json"
    doc_stem = Path(document_name).stem
    output_filename = f"{doc_stem}_ground_truth.json"
    output_path = os.path.join(TEST_DATA_DIR, output_filename)

    output_data = {
        "source_document": f"knowledge_source/{document_name}",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": validated_entries,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        error_msg = f"Failed to write output file: {e}"
        logger.error(error_msg)
        return GroundTruthResult(
            source_document=document_name,
            output_path="",
            entry_count=0,
            success=False,
            error=error_msg,
        )

    logger.info(
        "Ground-truth generated: %d entries written to %s",
        len(validated_entries),
        output_path,
    )

    # 9. Return GroundTruthResult with success status
    return GroundTruthResult(
        source_document=document_name,
        output_path=output_path,
        entry_count=len(validated_entries),
        success=True,
        error=None,
    )


# =============================================================================
# Private Helpers
# =============================================================================


def _split_into_sections(text: str) -> list[str]:
    """
    Split document text into meaningful sections by paragraphs.

    Splits on double newlines first. If that yields very few sections,
    falls back to splitting on single newlines with minimum length filtering.
    """
    # Try splitting on double newlines (paragraphs)
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]

    # If very few sections, try single newlines
    if len(sections) < 3:
        sections = [s.strip() for s in text.split("\n") if len(s.strip()) > 50]

    # Filter out very short sections (less than 30 chars)
    sections = [s for s in sections if len(s) >= 30]

    # If still empty, treat entire text as one section
    if not sections:
        sections = [text.strip()]

    return sections


def _parse_llm_response(response_text: str) -> Optional[list[dict]]:
    """
    Parse the LLM response text as a JSON array of Q&A entries.

    Attempts to extract JSON from the response, handling cases where
    the LLM wraps output in markdown code blocks, adds extra text,
    uses trailing commas, or includes other common formatting issues.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        List of dictionaries if parsing succeeds, None otherwise.
    """
    import re

    text = response_text.strip()

    # Try to extract JSON from markdown code blocks
    if "```json" in text:
        start = text.index("```json") + len("```json")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + len("```")
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        json_candidate = text[bracket_start : bracket_end + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fix common LLM JSON issues: trailing commas, single quotes
        fixed = json_candidate
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        # Replace single quotes with double quotes (careful with apostrophes)
        # Only do this if there are no double quotes already
        if '"' not in fixed and "'" in fixed:
            fixed = fixed.replace("'", '"')

        try:
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try fixing unescaped newlines inside strings
        fixed2 = re.sub(r'(?<=": ")(.*?)(?="[,\s}])', lambda m: m.group(0).replace('\n', '\\n'), json_candidate, flags=re.DOTALL)
        fixed2 = re.sub(r",\s*([}\]])", r"\1", fixed2)
        try:
            parsed = json.loads(fixed2)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Last resort: try to parse individual JSON objects from text
    objects = []
    for match in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(match.group())
            if "question" in obj and "answer" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            # Try fixing trailing commas in individual objects
            fixed_obj = re.sub(r",\s*}", "}", match.group())
            try:
                obj = json.loads(fixed_obj)
                if "question" in obj and "answer" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

    if objects:
        return objects

    return None


def _validate_entries(entries: list[dict]) -> list[dict]:
    """
    Validate and filter entries to ensure each has the required schema:
    - question: non-empty string
    - answer: non-empty string
    - contexts: list with at least one non-empty string
    """
    validated = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        question = entry.get("question", "")
        answer = entry.get("answer", "")
        contexts = entry.get("contexts", [])

        # Validate question and answer are non-empty strings
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue

        # Validate contexts is a list with at least one non-empty string
        if not isinstance(contexts, list):
            continue

        valid_contexts = [c for c in contexts if isinstance(c, str) and c.strip()]
        if not valid_contexts:
            continue

        validated.append({
            "question": question.strip(),
            "answer": answer.strip(),
            "contexts": valid_contexts,
        })

    return validated
