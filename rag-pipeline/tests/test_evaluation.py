"""Tests for the local RAGAS evaluation orchestration (without network calls)."""

import asyncio
import json

import pytest

from app.services import evaluation


def test_load_evaluation_cases_validates_and_normalizes(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [{"question": "  What is RAG? ", "reference": " Retrieval augmented generation. ", "document_ids": [1]}]
        ),
        encoding="utf-8",
    )

    cases = evaluation.load_evaluation_cases(dataset)

    assert cases == [
        evaluation.EvaluationCase(
            question="What is RAG?",
            reference="Retrieval augmented generation.",
            document_ids=[1],
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{}],
        [{"question": "q"}],
        [{"question": "q", "reference": "a", "document_ids": ["1"]}],
    ],
)
def test_load_evaluation_cases_rejects_invalid_data(tmp_path, payload):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        evaluation.load_evaluation_cases(dataset)


def test_run_evaluation_uses_the_same_retrieval_and_generation_flow(monkeypatch):
    class Result:
        value = 0.8
        reason = "mock evaluation"

    class Metric:
        async def ascore(self, **kwargs):
            return Result()

    monkeypatch.setattr(evaluation, "_create_ragas_metrics", lambda: {"faithfulness": Metric()})
    seen = {}

    def retrieve(**kwargs):
        seen["retrieve"] = kwargs
        return [{"text": "RAG combines retrieval and generation.", "document_id": 7, "filename": "guide.txt", "page_number": 2}]

    def generate(question, chunks):
        seen["generate"] = (question, chunks)
        return "RAG combines retrieval and generation."

    report = evaluation.run_evaluation(
        [evaluation.EvaluationCase("What is RAG?", "RAG combines retrieval and generation.", [7])],
        retrieve=retrieve,
        answer_generator=generate,
    )

    assert seen["retrieve"] == {"question": "What is RAG?", "document_ids": [7]}
    assert seen["generate"][0] == "What is RAG?"
    assert report["summary"] == {"faithfulness": 0.8}
    assert report["samples"][0]["metrics"]["faithfulness"] == {
        "score": 0.8,
        "reason": "mock evaluation",
    }
