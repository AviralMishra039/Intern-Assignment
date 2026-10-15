"""RAGAS evaluation helpers for the application's retrieval-and-answer flow.

This module is deliberately separate from the request path: evaluation is an
offline, opt-in operation that uses a labelled question/answer dataset and
calls the same ``retrieve_chunks`` and ``generate_answer`` functions as
``POST /query``.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from app.services.llm import generate_answer
from app.services.retrieval import retrieve_chunks


@dataclass(frozen=True)
class EvaluationCase:
    """One labelled RAG evaluation example."""

    question: str
    reference: str
    document_ids: Optional[List[int]] = None


def load_evaluation_cases(dataset_path: str | Path) -> List[EvaluationCase]:
    """Load and validate a JSON list of labelled evaluation examples.

    Each item must contain non-empty ``question`` and ``reference`` strings.
    ``document_ids`` is optional and mirrors the filter accepted by ``/query``.
    """
    path = Path(dataset_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Evaluation dataset was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation dataset is not valid JSON: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError("Evaluation dataset must be a non-empty JSON array.")

    cases: List[EvaluationCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object.")

        question = item.get("question")
        reference = item.get("reference")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset item {index} needs a non-empty 'question'.")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Dataset item {index} needs a non-empty 'reference'.")

        document_ids = item.get("document_ids")
        if document_ids is not None and (
            not isinstance(document_ids, list)
            or any(not isinstance(document_id, int) for document_id in document_ids)
        ):
            raise ValueError(
                f"Dataset item {index} has an invalid 'document_ids'; use a list of integers."
            )
        cases.append(
            EvaluationCase(
                question=question.strip(),
                reference=reference.strip(),
                document_ids=document_ids,
            )
        )
    return cases


def _create_ragas_metrics() -> Dict[str, Any]:
    """Create RAGAS v0.4 metrics backed by the project's Gemini provider.

    Imports are local so ordinary API startup and unit tests do not require the
    optional evaluation dependency to be imported until an evaluation is run.
    """
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to run a RAGAS evaluation.")

    try:
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are missing. Run 'pip install -r requirements.txt'."
        ) from exc

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    llm = llm_factory(settings.GEMINI_LLM_MODEL, provider="google", client=client)
    embeddings = GoogleEmbeddings(
        client=client,
        model=settings.GEMINI_EMBEDDING_MODEL,
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }


async def _score_case(metrics: Dict[str, Any], case: EvaluationCase, answer: str, contexts: List[str]) -> Dict[str, Any]:
    """Score one generated answer with the RAGAS v0.4 collections API."""
    common = {
        "user_input": case.question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": case.reference,
    }
    metric_inputs = {
        "faithfulness": {key: common[key] for key in ("user_input", "response", "retrieved_contexts")},
        "answer_relevancy": {key: common[key] for key in ("user_input", "response")},
        "context_precision": {key: common[key] for key in ("user_input", "retrieved_contexts", "reference")},
        "context_recall": {key: common[key] for key in ("user_input", "retrieved_contexts", "reference")},
        "answer_correctness": {key: common[key] for key in ("user_input", "response", "reference")},
    }
    scores: Dict[str, Any] = {}
    for name, metric in metrics.items():
        result = await metric.ascore(**metric_inputs[name])
        score = {"score": float(result.value)}
        if getattr(result, "reason", None):
            score["reason"] = result.reason
        scores[name] = score
    return scores


def run_evaluation(
    cases: List[EvaluationCase],
    *,
    retrieve: Callable[..., List[Dict[str, Any]]] = retrieve_chunks,
    answer_generator: Callable[[str, List[Dict[str, Any]]], str] = generate_answer,
) -> Dict[str, Any]:
    """Run the actual RAG pipeline for every case and return RAGAS scores."""
    metrics = _create_ragas_metrics()
    samples: List[Dict[str, Any]] = []
    for case in cases:
        chunks = retrieve(question=case.question, document_ids=case.document_ids)
        answer = answer_generator(case.question, chunks)
        contexts = [chunk["text"] for chunk in chunks]
        metric_scores = asyncio.run(_score_case(metrics, case, answer, contexts))
        samples.append(
            {
                "question": case.question,
                "reference": case.reference,
                "document_ids": case.document_ids,
                "answer": answer,
                "contexts": contexts,
                "sources": [
                    {
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                    }
                    for chunk in chunks
                ],
                "metrics": metric_scores,
            }
        )

    summary = {
        metric: sum(sample["metrics"][metric]["score"] for sample in samples) / len(samples)
        for metric in metrics
    }
    return {"summary": summary, "samples": samples}
