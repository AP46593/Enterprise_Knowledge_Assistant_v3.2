"""
Batch RAGAS Evaluation Script. (LLM-As-A-Judge)

Runs RAGAS evaluation against ground-truth files to measure RAG pipeline quality.

For each ground-truth entry:
1. Sends the question through the hybrid retriever to get contexts
2. Generates an answer using the RAG LLM with retrieved contexts
3. Runs RAGAS metrics comparing output vs ground truth

Usage:
    # Evaluate against all ground-truth files:
    python run_ragas_eval.py

    # Evaluate against a specific ground-truth file:
    python run_ragas_eval.py "XLUTS1_documentation_v17_ground_truth.json"

Prerequisites:
    - MCP Server must be running (for retrieval from the knowledge base)
      OR run with --standalone flag to use the retriever directly (no MCP server needed)
    - Ollama must be running with the configured models
    - Ground-truth files must exist in MCP_Stack/ground_truth_docs/
    - Activate the .mcpvenv virtual environment

Output:
    - Per-question scores printed to console
    - Summary report with averages
    - Detailed results saved to MCP_Stack/ground_truth_docs/eval_results_<timestamp>.json
"""

# Inject truststore SSL certificates before any network imports
import truststore
truststore.inject_into_ssl()

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is in the path so MCP_Stack package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MCP_Stack.agents.evaluator_agent import TEST_DATA_DIR, evaluate_response
from MCP_Stack.server_config import ENABLE_RAGAS_EVAL, KNOWLEDGE_BASE_DIR
from MCP_Stack.tools.retriever import hybrid_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Ground-Truth File I/O
# =============================================================================


def load_ground_truth_file(file_path: str) -> dict:
    """
    Load a ground-truth JSON file from disk.

    Args:
        file_path: Absolute or relative path to the ground-truth JSON file.

    Returns:
        Parsed dictionary with keys: source_document, generated_at, entries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_ground_truth_files() -> list[str]:
    """
    Find all ground-truth JSON files in the ground_truth_docs directory.

    Looks for files matching the *_ground_truth.json naming convention.

    Returns:
        Sorted list of absolute file paths to ground-truth files.
    """
    gt_dir = Path(TEST_DATA_DIR)
    if not gt_dir.exists():
        return []
    return sorted([
        str(f) for f in gt_dir.glob("*_ground_truth.json")
    ])


# =============================================================================
# Single Entry Evaluation
# =============================================================================


def evaluate_single_entry(entry: dict) -> dict:
    """
    Evaluate a single ground-truth entry against the RAG pipeline.

    Workflow:
        1. Retrieve contexts using hybrid search (semantic + BM25)
        2. Generate an answer using ChatOllama with retrieved contexts
        3. Run RAGAS evaluation metrics comparing output vs ground truth

    Args:
        entry: A ground-truth entry dict with keys: question, answer, contexts.

    Returns:
        Dictionary containing the question, ground-truth answer, generated answer,
        retrieved contexts, RAGAS scores (or None), and any error message.
    """
    question = entry["question"]
    ground_truth_answer = entry["answer"]
    ground_truth_contexts = entry["contexts"]

    # Step 1: Retrieve contexts from the knowledge base using hybrid search
    retrieval_results = hybrid_search(question)
    retrieved_contexts = [r.text for r in retrieval_results]

    if not retrieved_contexts:
        logger.warning("  No contexts retrieved for: %s", question[:80])
        return {
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "retrieved_answer": "No relevant information found.",
            "retrieved_contexts": [],
            "scores": None,
            "error": "No contexts retrieved",
        }

    # Step 2: Generate an answer using the RAG LLM with retrieved contexts
    try:
        from langchain_ollama import ChatOllama
        from MCP_Stack.server_config import DEFAULT_MODEL, MAX_TOKENS, OLLAMA_BASE_URL, TEMPERATURE

        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS,
        )

        # Build a RAG-style prompt with context and question
        context_text = "\n\n".join(retrieved_contexts)
        prompt = (
            f"Based on the following context, answer the question accurately and concisely.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        response = llm.invoke(prompt)
        generated_answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("  Answer generation failed: %s", e)
        generated_answer = "Error generating answer."

    # Step 3: Run RAGAS evaluation (LLM-as-judge scoring)
    eval_result = evaluate_response(
        question=question,
        answer=generated_answer,
        contexts=retrieved_contexts,
        ground_truth=ground_truth_answer,
    )

    # Extract numeric scores from the evaluation result
    scores = None
    if eval_result:
        scores = {
            "faithfulness": eval_result.faithfulness,
            "answer_relevancy": eval_result.answer_relevancy,
            "context_precision": eval_result.context_precision,
            "context_recall": eval_result.context_recall,
        }

    return {
        "question": question,
        "ground_truth_answer": ground_truth_answer,
        "retrieved_answer": generated_answer,
        "retrieved_contexts": retrieved_contexts[:3],  # Truncate for readability
        "scores": scores,
        "error": None,
    }


# =============================================================================
# Display Helpers
# =============================================================================


def print_scores(scores: dict, prefix: str = "  "):
    """
    Pretty-print RAGAS scores on a 0.0–5.0 scale.

    Args:
        scores: Dictionary with metric names as keys and float scores as values.
        prefix: Indentation prefix for each printed line.
    """
    if not scores:
        print(f"{prefix}Scores: N/A (evaluation failed or disabled)")
        return
    print(f"{prefix}Faithfulness:      {scores['faithfulness']:.1f} / 5.0")
    print(f"{prefix}Answer Relevancy:  {scores['answer_relevancy']:.1f} / 5.0")
    print(f"{prefix}Context Precision: {scores['context_precision']:.1f} / 5.0")
    if scores.get("context_recall") is not None:
        print(f"{prefix}Context Recall:    {scores['context_recall']:.1f} / 5.0")


# =============================================================================
# Aggregation
# =============================================================================


def compute_averages(results: list[dict]) -> dict:
    """
    Compute average scores across all successfully evaluated entries.

    Only entries with non-None scores are included in the average computation.
    Context recall is averaged separately since it requires ground-truth.

    Args:
        results: List of result dicts from evaluate_single_entry.

    Returns:
        Dictionary with average score per metric, or empty dict if no
        entries had scores.
    """
    scored_results = [r for r in results if r.get("scores")]
    if not scored_results:
        return {}

    n = len(scored_results)
    avg = {
        "faithfulness": sum(r["scores"]["faithfulness"] for r in scored_results) / n,
        "answer_relevancy": sum(r["scores"]["answer_relevancy"] for r in scored_results) / n,
        "context_precision": sum(r["scores"]["context_precision"] for r in scored_results) / n,
    }

    # Context recall requires ground truth — not all entries may have it
    recall_results = [r for r in scored_results if r["scores"].get("context_recall") is not None]
    if recall_results:
        avg["context_recall"] = sum(r["scores"]["context_recall"] for r in recall_results) / len(recall_results)

    return avg


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Run batch RAGAS evaluation against ground-truth files.

    Validates prerequisites, loads ground-truth files, evaluates each entry,
    prints per-question scores and a summary, then saves detailed results
    to a timestamped JSON file.
    """
    # Check if RAGAS evaluation is enabled in config
    if not ENABLE_RAGAS_EVAL:
        print("WARNING: ENABLE_RAGAS_EVAL is set to 'false' in MCP_Stack/.env")
        print("RAGAS metrics will not be computed. Set ENABLE_RAGAS_EVAL=true to enable.")
        print("Continuing with retrieval-only evaluation...\n")

    # Verify knowledge base exists (required for retrieval)
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        print("ERROR: Knowledge base not found at:", KNOWLEDGE_BASE_DIR)
        print("Start the MCP server first to ingest documents, then run this script.")
        sys.exit(1)

    # Determine which ground-truth files to evaluate
    if len(sys.argv) > 1 and sys.argv[1] != "--standalone":
        # Specific file provided via CLI argument
        specific_file = os.path.join(TEST_DATA_DIR, sys.argv[1])
        if not os.path.isfile(specific_file):
            print(f"ERROR: File not found: {specific_file}")
            sys.exit(1)
        gt_files = [specific_file]
    else:
        gt_files = find_ground_truth_files()

    if not gt_files:
        print("No ground-truth files found in:", TEST_DATA_DIR)
        print("Run 'python generate_ground_truth.py --all' first.")
        sys.exit(1)

    # Print evaluation header
    print(f"Found {len(gt_files)} ground-truth file(s)")
    print(f"Knowledge base: {KNOWLEDGE_BASE_DIR}")
    print(f"RAGAS evaluation: {'ENABLED' if ENABLE_RAGAS_EVAL else 'DISABLED'}")
    print("=" * 70)

    all_results = []

    # Iterate over each ground-truth file and evaluate all entries
    for gt_file in gt_files:
        gt_data = load_ground_truth_file(gt_file)
        doc_name = gt_data.get("source_document", os.path.basename(gt_file))
        entries = gt_data.get("entries", [])

        print(f"\n📄 {doc_name} ({len(entries)} questions)")
        print("-" * 50)

        for i, entry in enumerate(entries, 1):
            question = entry.get("question", "")
            print(f"\n  Q{i}: {question[:100]}")

            result = evaluate_single_entry(entry)
            all_results.append(result)

            if result["error"]:
                print(f"  ⚠️  {result['error']}")
            else:
                print(f"  A:  {result['retrieved_answer'][:100]}...")
                print_scores(result["scores"])

    # ==========================================================================
    # Summary Report
    # ==========================================================================

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions evaluated: {len(all_results)}")
    print(f"Successful evaluations:   {sum(1 for r in all_results if r.get('scores'))}")
    print(f"Failed evaluations:       {sum(1 for r in all_results if r.get('error'))}")

    averages = compute_averages(all_results)
    if averages:
        print(f"\nAverage Scores:")
        print(f"  Faithfulness:      {averages['faithfulness']:.1f} / 5.0")
        print(f"  Answer Relevancy:  {averages['answer_relevancy']:.1f} / 5.0")
        print(f"  Context Precision: {averages['context_precision']:.1f} / 5.0")
        if "context_recall" in averages:
            print(f"  Context Recall:    {averages['context_recall']:.1f} / 5.0")

    # ==========================================================================
    # Save Detailed Results to JSON
    # ==========================================================================

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Name output file based on whether evaluating a single doc vs all
    if len(sys.argv) > 1 and sys.argv[1] != "--standalone":
        doc_stem = Path(sys.argv[1]).stem.replace("_ground_truth", "")
        output_file = os.path.join(TEST_DATA_DIR, f"eval_results_{doc_stem}_{timestamp}.json")
    else:
        output_file = os.path.join(TEST_DATA_DIR, f"eval_results_all_{timestamp}.json")

    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    output_data = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(all_results),
        "successful": sum(1 for r in all_results if r.get("scores")),
        "averages": averages,
        "results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
