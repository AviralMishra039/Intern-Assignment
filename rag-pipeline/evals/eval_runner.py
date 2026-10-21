"""Automated evaluation and ablation runner for the RAG pipeline.

Executes ground-truth queries against 3 pipeline configurations:
  1. Vector Only (Baseline)
  2. Hybrid Search (Vector + BM25 with RRF)
  3. Hybrid + Reranker (Full Pipeline with Cross-Encoder)

Outputs a comprehensive ablation table comparing retrieval quality
(context_precision, context_recall, keyword hit-rate) and generation
quality (faithfulness, answer_relevancy).

Usage:
  # Full evaluation using Gemini + RAGAS (requires GEMINI_API_KEY in .env):
  python evals/eval_runner.py

  # Fast offline mode (tests retrieval logic & keyword metrics without LLM calls):
  python evals/eval_runner.py --offline
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from app.config import settings
from app.services import ingestion, lexical, retrieval
from app.services.llm import generate_answer
from evals.ragas_evaluator import (
    compute_keyword_retrieval_metrics,
    run_ragas_evaluation,
)

EVAL_COLLECTION_NAME = "documents_eval"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "eval_dataset.json")
SAMPLE_DOC_PATH = os.path.join(
    os.path.dirname(__file__), "sample_docs", "security_and_operations.txt"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_dataset() -> List[Dict]:
    """Load benchmark ground-truth QA dataset."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_eval_collection(offline: bool = False):
    """Index the sample policy document into an isolated evaluation collection."""
    print("Setting up evaluation document and index...")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    # Re-create a clean evaluation collection
    try:
        chroma_client.delete_collection(EVAL_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=EVAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    with open(SAMPLE_DOC_PATH, "r", encoding="utf-8") as f:
        doc_text = f.read()

    pages = [{"text": doc_text, "page_number": 1}]
    chunks = ingestion.chunk_text(pages)
    texts = [c["text"] for c in chunks]

def _offline_embed(text: str, dim: int = settings.EMBEDDING_DIMENSIONS) -> List[float]:
    """Deterministic hash embedding for offline testing without API key."""
    import hashlib
    import math

    vec = [0.0] * dim
    for word in lexical.tokenize(text):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else [0.01] * dim


def setup_eval_collection(offline: bool = False):
    """Index the sample policy document into an isolated evaluation collection."""
    print("Setting up evaluation document and index...")
    chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    # Re-create a clean evaluation collection
    try:
        chroma_client.delete_collection(EVAL_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=EVAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    with open(SAMPLE_DOC_PATH, "r", encoding="utf-8") as f:
        doc_text = f.read()

    pages = [{"text": doc_text, "page_number": 1}]
    chunks = ingestion.chunk_text(pages)
    texts = [c["text"] for c in chunks]

    if offline:
        embeddings = [_offline_embed(t) for t in texts]
    else:
        from app.services.gemini import embed_documents

        embeddings = embed_documents(texts)

    ids = [f"eval_doc_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "document_id": 999,
            "filename": "security_and_operations.txt",
            "page_number": c["page_number"],
        }
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    lexical.invalidate_index()
    print(f"Indexed {len(chunks)} chunks into '{EVAL_COLLECTION_NAME}'.")
    return collection


def run_pipeline_for_config(
    dataset: List[Dict],
    collection,
    hybrid_enabled: bool,
    reranker_enabled: bool,
    offline: bool = False,
) -> List[Dict]:
    """Execute pipeline on dataset under specified configuration."""
    settings.HYBRID_SEARCH_ENABLED = hybrid_enabled
    settings.RERANKER_ENABLED = reranker_enabled

    # Temporarily point retrieval to evaluation collection
    orig_get_collection = retrieval.get_chroma_collection
    orig_embed_query = retrieval.embed_query
    retrieval.get_chroma_collection = lambda: collection

    if offline:
        retrieval.embed_query = lambda q: _offline_embed(q)

    results = []
    try:
        for item in dataset:
            question = item["question"]
            chunks = retrieval.retrieve_chunks(question, top_k=settings.TOP_K_RESULTS)
            contexts = [c["text"] for c in chunks]

            if offline:
                answer = (
                    "Context summary: " + chunks[0]["text"][:120]
                    if chunks
                    else "I could not find relevant information in the uploaded documents."
                )
            else:
                answer = generate_answer(question, chunks)

            results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item["ground_truth"],
                    "reference_keywords": item.get("reference_keywords", []),
                }
            )
    finally:
        retrieval.get_chroma_collection = orig_get_collection
        retrieval.embed_query = orig_embed_query

    return results


def format_markdown_table(ablation_results: Dict[str, Dict]) -> str:
    """Format evaluation comparison table as markdown."""
    headers = [
        "Configuration",
        "Keyword Hit Rate",
        "MRR",
        "Context Precision",
        "Context Recall",
        "Faithfulness",
        "Answer Relevancy",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for config_name, metrics in ablation_results.items():
        row = [
            f"**{config_name}**",
            f"{metrics.get('keyword_hit_rate', 0.0):.2%}",
            f"{metrics.get('mean_reciprocal_rank', 0.0):.4f}",
            f"{metrics.get('context_precision', 0.0):.4f}",
            f"{metrics.get('context_recall', 0.0):.4f}",
            f"{metrics.get('faithfulness', 0.0):.4f}",
            f"{metrics.get('answer_relevancy', 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation & Ablation Runner")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode using keyword metrics without LLM calls.",
    )
    args = parser.parse_args()

    offline_mode = args.offline or not settings.GEMINI_API_KEY
    if offline_mode and not args.offline:
        print("Note: GEMINI_API_KEY not found in .env. Running in offline evaluation mode.")

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} benchmark test cases.")

    collection = setup_eval_collection(offline=offline_mode)

    configurations = [
        ("Vector Only (Baseline)", False, False),
        ("Hybrid (Vector + BM25)", True, False),
        ("Hybrid + Reranker (Full)", True, True),
    ]

    ablation_summary = {}
    detailed_outputs = {}

    for name, hybrid, rerank in configurations:
        print(f"\nEvaluating: {name}...")
        start_time = time.time()
        samples = run_pipeline_for_config(
            dataset=dataset,
            collection=collection,
            hybrid_enabled=hybrid,
            reranker_enabled=rerank,
            offline=offline_mode,
        )
        elapsed = time.time() - start_time
        print(f"Completed queries in {elapsed:.2f}s.")

        # Compute retrieval metrics
        metrics = compute_keyword_retrieval_metrics(samples)

        # Compute full Ragas metrics if online
        if not offline_mode:
            print(f"Running RAGAS judge metrics for '{name}'...")
            try:
                ragas_scores = run_ragas_evaluation(samples)
                metrics.update(ragas_scores)
            except Exception as e:
                print(f"Warning: RAGAS evaluation encountered an error: {e}")

        ablation_summary[name] = metrics
        detailed_outputs[name] = samples

    # Format and display markdown report
    markdown_table = format_markdown_table(ablation_summary)
    print("\n" + "=" * 60)
    print("RAG PIPELINE ABLATION & EVALUATION RESULTS")
    print("=" * 60 + "\n")
    print(markdown_table + "\n")

    # Save to evals/results/
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_json = os.path.join(RESULTS_DIR, "latest_ablation_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "offline_mode": offline_mode,
                "summary": ablation_summary,
                "detailed": detailed_outputs,
            },
            f,
            indent=2,
        )

    summary_md = os.path.join(RESULTS_DIR, "ablation_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# RAG Pipeline Ablation Study\n\n")
        f.write(
            "Evaluation performed against benchmark test cases comparing retrieval architectures:\n\n"
        )
        f.write(markdown_table + "\n")

    print(f"Results saved to: {results_json}")
    print(f"Markdown table saved to: {summary_md}")


if __name__ == "__main__":
    main()
