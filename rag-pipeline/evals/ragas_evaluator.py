"""RAGAS evaluation adapter configured for Google Gemini.

Evaluates RAG pipeline outputs using core metrics:
  - context_precision: Signal-to-noise ratio in retrieved contexts.
  - context_recall: Measures whether all ground-truth facts were retrieved.
  - faithfulness: Measures if the answer is grounded strictly in the context.
  - answer_relevancy: Measures how directly the answer addresses the question.
"""

import logging
import math
from typing import Dict, List, Optional

from google import genai
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings

logger = logging.getLogger(__name__)


def _patch_google_embeddings():
    """Bridge GoogleEmbeddings with langchain-style methods expected by RAGAS metrics."""
    if not hasattr(GoogleEmbeddings, "embed_query"):
        GoogleEmbeddings.embed_query = lambda self, text: self.embed_text(text)
    if not hasattr(GoogleEmbeddings, "embed_documents"):
        GoogleEmbeddings.embed_documents = lambda self, texts: self.embed_texts(texts)
    if not hasattr(GoogleEmbeddings, "aembed_query"):
        GoogleEmbeddings.aembed_query = lambda self, text: self.aembed_text(text)
    if not hasattr(GoogleEmbeddings, "aembed_documents"):
        GoogleEmbeddings.aembed_documents = lambda self, texts: self.aembed_texts(texts)


def get_ragas_gemini_judge(api_key: Optional[str] = None, model_name: Optional[str] = None):
    """Instantiate Ragas LLM and Embeddings wrappers using Google GenAI."""
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is required to run RAGAS with Gemini. "
            "Please set GEMINI_API_KEY in your .env file."
        )

    _patch_google_embeddings()

    client = genai.Client(api_key=key)

    # Use specified model, configured model, or proven stable gemini-2.5-flash
    model = model_name or settings.GEMINI_LLM_MODEL or "gemini-2.5-flash"

    try:
        eval_llm = llm_factory(
            model=model,
            provider="google",
            client=client,
        )
    except Exception as exc:
        logger.warning("Failed to initialize judge with %s (%s). Falling back to gemini-2.5-flash.", model, exc)
        eval_llm = llm_factory(
            model="gemini-2.5-flash",
            provider="google",
            client=client,
        )

    eval_embeddings = GoogleEmbeddings(
        client=client,
        model=settings.GEMINI_EMBEDDING_MODEL,
    )

    return eval_llm, eval_embeddings


def run_ragas_evaluation(
    samples_data: List[Dict],
    api_key: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, float]:
    """Execute Ragas evaluation on a list of QA samples.

    Each sample dict must contain:
      - question (str)
      - answer (str)
      - contexts (List[str])
      - ground_truth (str)

    Returns:
      Dict mapping metric name to average score (0.0 to 1.0).
    """
    eval_llm, eval_embeddings = get_ragas_gemini_judge(api_key, judge_model)

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("ground_truth", ""),
        )
        for s in samples_data
    ]

    dataset = EvaluationDataset(samples=ragas_samples)

    # Core evaluation metrics: pre-instantiated Metric objects
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    for m in metrics:
        m.llm = eval_llm
        if hasattr(m, "embeddings"):
            m.embeddings = eval_embeddings

    run_config = RunConfig(max_retries=3, max_wait=20, timeout=90)

    logger.info("Executing RAGAS evaluation with %d samples...", len(ragas_samples))
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
        show_progress=True,
    )

    # Convert results to a clean dict of float averages
    scores = {}
    raw_dict = getattr(results, "_repr_dict", None)
    if raw_dict is None:
        if hasattr(results, "items"):
            raw_dict = dict(results.items())
        elif hasattr(results, "_scores_dict"):
            raw_dict = {k: results[k] for k in getattr(results, "_scores_dict", {}).keys()}
        else:
            raw_dict = {}

    for metric_name, val in raw_dict.items():
        try:
            # Handle list of scores or single scalar
            if isinstance(val, (list, tuple)):
                v = sum(float(x) for x in val if not math.isnan(float(x))) / max(len(val), 1)
            else:
                v = float(val)
            scores[str(metric_name)] = round(v, 4) if not math.isnan(v) else 0.0
        except (TypeError, ValueError):
            scores[str(metric_name)] = 0.0

    return scores


def compute_keyword_retrieval_metrics(
    samples_data: List[Dict],
) -> Dict[str, float]:
    """Deterministic, offline retrieval metric: keyword hit rate.

    Calculates the percentage of expected reference keywords found in retrieved
    chunks. Useful for local tests, fast sanity-checks, and CI environments
    without burning LLM tokens.
    """
    if not samples_data:
        return {"keyword_hit_rate": 0.0, "mrr": 0.0}

    hit_rates = []
    reciprocal_ranks = []

    for s in samples_data:
        keywords = s.get("reference_keywords", [])
        if not keywords:
            continue

        contexts = s.get("contexts", [])
        combined_text = " ".join(contexts).lower()

        # Count keyword coverage
        hits = sum(1 for kw in keywords if kw.lower() in combined_text)
        hit_rates.append(hits / len(keywords) if keywords else 1.0)

        # Compute rank of first chunk containing any keyword
        first_rank = 0
        for rank, chunk in enumerate(contexts, start=1):
            if any(kw.lower() in chunk.lower() for kw in keywords):
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)

    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "keyword_hit_rate": round(avg_hit_rate, 4),
        "mean_reciprocal_rank": round(avg_mrr, 4),
    }
