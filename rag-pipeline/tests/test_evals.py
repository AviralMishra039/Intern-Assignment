"""Tests for the RAGAS evaluation harness and ablation runner."""

import os
from unittest.mock import MagicMock

import pytest

from evals import eval_runner, ragas_evaluator


def test_eval_dataset_schema():
    """Verify eval_dataset.json loads and matches required schema."""
    dataset = eval_runner.load_dataset()
    assert len(dataset) >= 10

    required_keys = {"id", "question", "ground_truth", "category", "reference_keywords"}
    for item in dataset:
        assert required_keys <= set(item.keys())
        assert len(item["question"].strip()) > 0
        assert len(item["ground_truth"].strip()) > 0
        assert isinstance(item["reference_keywords"], list)


def test_keyword_retrieval_metrics():
    """Verify deterministic retrieval metrics calculation."""
    sample_data = [
        {
            "question": "What is SEC-AUTH-401?",
            "contexts": [
                "Directive SEC-AUTH-401 prohibits SMS 2FA.",
                "Other unrelated content.",
            ],
            "reference_keywords": ["SEC-AUTH-401", "prohibits"],
        },
        {
            "question": "What is ERR-CRIT-911?",
            "contexts": [
                "Unrelated line.",
                "Escalation code ERR-CRIT-911 triggers calls.",
            ],
            "reference_keywords": ["ERR-CRIT-911", "triggers calls"],
        },
    ]

    metrics = ragas_evaluator.compute_keyword_retrieval_metrics(sample_data)
    assert metrics["keyword_hit_rate"] == 1.0
    # First query keyword found at rank 1 (1/1 = 1.0), second query at rank 2 (1/2 = 0.5) -> avg 0.75
    assert metrics["mean_reciprocal_rank"] == 0.75


def test_eval_runner_offline_execution(tmp_path, monkeypatch):
    """Verify eval runner executes end-to-end in offline mode without network/API."""
    monkeypatch.setattr(eval_runner, "RESULTS_DIR", str(tmp_path))

    collection = eval_runner.setup_eval_collection(offline=True)
    assert collection.count() > 0

    dataset = eval_runner.load_dataset()[:2]
    samples = eval_runner.run_pipeline_for_config(
        dataset=dataset,
        collection=collection,
        hybrid_enabled=True,
        reranker_enabled=True,
        offline=True,
    )
    assert len(samples) == 2
    assert "contexts" in samples[0]
    assert len(samples[0]["contexts"]) > 0

    metrics = ragas_evaluator.compute_keyword_retrieval_metrics(samples)
    assert "keyword_hit_rate" in metrics
    assert "mean_reciprocal_rank" in metrics
