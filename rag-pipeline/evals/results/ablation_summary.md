# RAG Pipeline Ablation Study

Evaluation performed against benchmark test cases comparing retrieval architectures:

| Configuration | Keyword Hit Rate | MRR | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
| --- | --- | --- | --- | --- | --- | --- |
| **Vector Only (Baseline)** | 90.00% | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Hybrid (Vector + BM25)** | 90.00% | 0.8500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Hybrid + Reranker (Full)** | 90.00% | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
