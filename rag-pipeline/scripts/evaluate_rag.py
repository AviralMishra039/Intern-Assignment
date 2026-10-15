"""Run RAGAS against this application's real retrieval and generation pipeline.

Usage (from the rag-pipeline directory):
    python scripts/evaluate_rag.py --dataset evaluation/dataset.example.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.evaluation import load_evaluation_cases, run_evaluation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", required=True, help="JSON array of question/reference examples.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON report (default: evaluation/results/<UTC timestamp>.json).",
    )
    args = parser.parse_args()

    try:
        report = run_evaluation(load_evaluation_cases(args.dataset))
    except (RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else Path("evaluation/results") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("RAGAS evaluation complete")
    for metric, score in report["summary"].items():
        print(f"  {metric}: {score:.3f}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
