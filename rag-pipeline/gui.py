"""Interactive Gradio Web Interface for the RAG Pipeline.

Provides an all-in-one UI for:
  1. Document Ingestion: Upload PDFs, Word documents, or TXT files.
  2. Question Answering: Grounded generation with hybrid retrieval toggles.
  3. Live RAGAS Evaluation: Single-turn evaluation and benchmark ablation studies.

Run with:
  python gui.py
"""

import os
import shutil
import sys
import tempfile
import uuid
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
from app.config import settings
from app.database import Document, SessionLocal, get_chroma_collection, init_db
from app.services import ingestion, lexical, retrieval
from app.services.ingestion import process_document
from app.services.lexical import invalidate_index
from app.services.llm import generate_answer
from evals.eval_runner import (
    EVAL_COLLECTION_NAME,
    format_markdown_table,
    load_dataset,
    run_pipeline_for_config,
    setup_eval_collection,
)
from evals.ragas_evaluator import (
    compute_keyword_retrieval_metrics,
    run_ragas_evaluation,
)

# Ensure database tables exist
init_db()


# ============================================================================
# Helpers: Document Management
# ============================================================================


def get_documents_table():
    """Fetch current documents from SQLite for display in Gradio."""
    db = SessionLocal()
    try:
        docs = db.query(Document).order_by(Document.id.desc()).all()
        if not docs:
            return [["No documents uploaded yet.", "-", "-", "-", "-", "-"]]
        return [
            [
                d.id,
                d.filename,
                d.file_type.upper(),
                d.page_count,
                d.chunk_count,
                f"{d.file_size_kb:.1f} KB",
            ]
            for d in docs
        ]
    finally:
        db.close()


def get_document_choices():
    """Return list of (label, id) choices for document filtering dropdown."""
    db = SessionLocal()
    try:
        docs = db.query(Document).filter(Document.status == "ready").all()
        return [f"ID {d.id}: {d.filename}" for d in docs]
    finally:
        db.close()


def upload_and_index_file(file_obj):
    """Handle document upload and run ingestion pipeline."""
    if file_obj is None:
        return "Please select a file to upload.", get_documents_table(), gr.update(choices=get_document_choices())

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    original_filename = os.path.basename(file_path)
    file_ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""

    if file_ext not in {"pdf", "docx", "txt"}:
        return (
            f"Error: Unsupported format '.{file_ext}'. Please upload a PDF, DOCX, or TXT file.",
            get_documents_table(),
            gr.update(choices=get_document_choices()),
        )

    db = SessionLocal()
    try:
        doc_count = db.query(Document).count()
        if doc_count >= settings.MAX_DOCUMENTS:
            return (
                f"Error: Maximum document limit of {settings.MAX_DOCUMENTS} reached. Please delete old documents first.",
                get_documents_table(),
                gr.update(choices=get_document_choices()),
            )

        # Copy to UPLOAD_DIR
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        safe_filename = f"{uuid.uuid4().hex}.{file_ext}"
        target_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        shutil.copyfile(file_path, target_path)

        file_size_kb = os.path.getsize(target_path) / 1024.0

        # Create metadata record
        doc_record = Document(
            filename=original_filename,
            original_name=original_filename,
            file_type=file_ext,
            file_size_kb=round(file_size_kb, 2),
            status="processing",
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        # Ingest: extract -> token chunk -> embed -> ChromaDB
        chunk_count, page_count = process_document(
            file_path=target_path,
            file_type=file_ext,
            document_id=doc_record.id,
            filename=original_filename,
        )

        doc_record.chunk_count = chunk_count
        doc_record.page_count = page_count
        doc_record.status = "ready"
        db.commit()

        status_msg = (
            f"Successfully indexed '{original_filename}'!\n"
            f"- Document ID: {doc_record.id}\n"
            f"- Pages: {page_count}\n"
            f"- Token Chunks: {chunk_count}\n"
            f"- Size: {file_size_kb:.1f} KB"
        )
        return status_msg, get_documents_table(), gr.update(choices=get_document_choices())

    except Exception as exc:
        db.rollback()
        return f"Processing failed: {str(exc)}", get_documents_table(), gr.update(choices=get_document_choices())
    finally:
        db.close()


def delete_single_document(doc_id_str):
    """Delete a document by ID from SQLite and ChromaDB."""
    if not doc_id_str:
        return "Please specify a document ID to delete.", get_documents_table(), gr.update(choices=get_document_choices())

    try:
        doc_id = int(str(doc_id_str).split(":")[0].replace("ID", "").strip())
    except ValueError:
        return f"Invalid Document ID: {doc_id_str}", get_documents_table(), gr.update(choices=get_document_choices())

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return f"Document ID {doc_id} not found.", get_documents_table(), gr.update(choices=get_document_choices())

        # Delete chunks from Chroma
        collection = get_chroma_collection()
        chunk_ids = [f"{doc_id}_{i}" for i in range(doc.chunk_count or 0)]
        if chunk_ids:
            try:
                collection.delete(ids=chunk_ids)
            except Exception:
                pass

        invalidate_index()
        db.delete(doc)
        db.commit()

        return (
            f"Document {doc_id} ('{doc.filename}') deleted successfully.",
            get_documents_table(),
            gr.update(choices=get_document_choices()),
        )
    finally:
        db.close()


def clear_all_documents():
    """Reset both ChromaDB and SQLite metadata."""
    db = SessionLocal()
    try:
        db.query(Document).delete()
        db.commit()

        collection = get_chroma_collection()
        try:
            # Delete all items in collection
            data = collection.get()
            if data and data.get("ids"):
                collection.delete(ids=data["ids"])
        except Exception:
            pass

        invalidate_index()
        return "All documents and vector embeddings cleared.", get_documents_table(), gr.update(choices=[])
    finally:
        db.close()


# ============================================================================
# Helpers: Querying & RAG Pipeline
# ============================================================================


def parse_selected_doc_ids(selected_choices) -> Optional[List[int]]:
    """Convert dropdown items like 'ID 3: file.pdf' into [3]."""
    if not selected_choices:
        return None
    ids = []
    for item in selected_choices:
        try:
            val = int(str(item).split(":")[0].replace("ID", "").strip())
            ids.append(val)
        except Exception:
            continue
    return ids if ids else None


def query_rag_pipeline(
    question: str,
    selected_docs,
    hybrid_enabled: bool,
    reranker_enabled: bool,
    top_k: int,
    model_name: str = "gemini-3.5-flash",
):
    """Run RAG query through the pipeline with live configuration toggles."""
    if not question.strip():
        return "Please enter a question.", "No sources to display."

    # Apply UI toggles to current pipeline execution
    settings.HYBRID_SEARCH_ENABLED = hybrid_enabled
    settings.RERANKER_ENABLED = reranker_enabled

    doc_ids = parse_selected_doc_ids(selected_docs)

    try:
        chunks = retrieval.retrieve_chunks(
            question=question.strip(),
            document_ids=doc_ids,
            top_k=int(top_k),
        )

        answer = generate_answer(
            question=question.strip(),
            chunks=chunks,
            model_name=model_name,
        )

        if not chunks:
            sources_md = "*No chunks retrieved. The document store may be empty or no documents matched the filter.*"
        else:
            cards = []
            for idx, c in enumerate(chunks, 1):
                rerank_info = (
                    f" | **Rerank Score**: `{c['rerank_score']:.4f}`"
                    if "rerank_score" in c
                    else ""
                )
                rrf_info = (
                    f" | **RRF Score**: `{c['rrf_score']:.4f}`"
                    if "rrf_score" in c
                    else ""
                )
                card = (
                    f"### Chunk {idx}: `{c.get('filename', 'Unknown')}` (Page {c.get('page_number', 1)})\n"
                    f"*Doc ID*: `{c.get('document_id')}`{rrf_info}{rerank_info}\n\n"
                    f"> {c.get('text', '')}\n"
                )
                cards.append(card)
            sources_md = "\n---\n".join(cards)

        return answer, sources_md

    except Exception as exc:
        return f"Query failed: {str(exc)}", ""


# ============================================================================
# Helpers: RAGAS Evaluation
# ============================================================================


def evaluate_single_query_ragas(
    question: str,
    ground_truth: str,
    hybrid_enabled: bool,
    reranker_enabled: bool,
    top_k: int,
):
    """Run single-query RAGAS evaluation."""
    if not question.strip():
        return "Please provide a question to evaluate.", "N/A", "N/A", "N/A", "N/A"

    if not settings.GEMINI_API_KEY:
        return (
            "GEMINI_API_KEY is not configured in your .env file. "
            "Please add your free Gemini API key to run RAGAS metrics.",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
        )

    settings.HYBRID_SEARCH_ENABLED = hybrid_enabled
    settings.RERANKER_ENABLED = reranker_enabled

    try:
        chunks = retrieval.retrieve_chunks(question.strip(), top_k=int(top_k))
        if not chunks:
            return (
                "⚠️ No documents or matching chunks found in the database. Please upload and index documents in Tab 1 first.",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
            )

        answer = generate_answer(question.strip(), chunks)
        contexts = [c["text"] for c in chunks]

        ref = ground_truth.strip() if ground_truth.strip() else answer

        sample = [
            {
                "question": question.strip(),
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ref,
            }
        ]

        scores = run_ragas_evaluation(sample)

        f_score = f"{scores.get('faithfulness', 0.0):.2%}"
        r_score = f"{scores.get('answer_relevancy', 0.0):.2%}"
        p_score = f"{scores.get('context_precision', 0.0):.2%}" if ground_truth.strip() else "N/A (No Ground Truth)"
        c_score = f"{scores.get('context_recall', 0.0):.2%}" if ground_truth.strip() else "N/A (No Ground Truth)"

        summary_text = (
            f"**Generated Answer**:\n{answer}\n\n"
            f"**Retrieved Chunks Count**: {len(contexts)}"
        )

        return summary_text, f_score, r_score, p_score, c_score

    except Exception as exc:
        return f"RAGAS evaluation failed: {str(exc)}", "Error", "Error", "Error", "Error"


def run_benchmark_ablation_gui(mode: str):
    """Execute benchmark ablation study and return formatted markdown table."""
    offline_mode = (mode == "Fast Offline Mode (Deterministic)") or not settings.GEMINI_API_KEY

    try:
        dataset = load_dataset()
        collection = setup_eval_collection(offline=offline_mode)

        configurations = [
            ("Vector Only (Baseline)", False, False),
            ("Hybrid (Vector + BM25)", True, False),
            ("Hybrid + Reranker (Full)", True, True),
        ]

        ablation_summary = {}

        for name, hybrid, rerank in configurations:
            samples = run_pipeline_for_config(
                dataset=dataset,
                collection=collection,
                hybrid_enabled=hybrid,
                reranker_enabled=rerank,
                offline=offline_mode,
            )

            metrics = compute_keyword_retrieval_metrics(samples)

            if not offline_mode:
                try:
                    ragas_scores = run_ragas_evaluation(samples)
                    metrics.update(ragas_scores)
                except Exception:
                    pass

            ablation_summary[name] = metrics

        table_md = format_markdown_table(ablation_summary)
        mode_label = "Offline Mode (Deterministic Keyword/MRR Scoring)" if offline_mode else "Online RAGAS Gemini Judge Mode"
        return f"### Evaluation Completed ({mode_label})\n\n{table_md}"

    except Exception as exc:
        return f"Benchmark evaluation failed: {str(exc)}"


# ============================================================================
# Gradio UI Layout
# ============================================================================

custom_css = """
.status-bar {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.95rem;
}
.metric-box {
    text-align: center;
    border-radius: 8px;
    padding: 10px;
    background: #f1f5f9;
}
"""

with gr.Blocks(title="RAG Pipeline Studio") as demo:
    gr.Markdown("# 🚀 Production RAG Pipeline Studio")
    gr.Markdown(
        "A complete playground for document indexing, hybrid retrieval (Vector + BM25 + FlashRank Reranker), "
        "and automated **RAGAS** evaluations using Google Gemini."
    )

    # Status Bar
    api_status = "✅ Configured" if settings.GEMINI_API_KEY else "⚠️ Missing GEMINI_API_KEY in .env"
    gr.HTML(
        f"""
        <div class="status-bar">
            <strong>LLM:</strong> <code>{settings.GEMINI_LLM_MODEL}</code> &nbsp;|&nbsp;
            <strong>Embeddings:</strong> <code>{settings.GEMINI_EMBEDDING_MODEL} (768-d)</code> &nbsp;|&nbsp;
            <strong>Vector DB:</strong> <code>ChromaDB ({settings.CHROMA_COLLECTION})</code> &nbsp;|&nbsp;
            <strong>API Key:</strong> {api_status}
        </div>
        """
    )

    with gr.Tabs():
        # --------------------------------------------------------------------
        # Tab 1: Document Ingestion
        # --------------------------------------------------------------------
        with gr.TabItem("📄 1. Document Ingestion"):
            gr.Markdown("### Upload & Index Documents")
            gr.Markdown("Upload PDF, Word (DOCX), or plain text (TXT) files into the pipeline.")

            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Select Document",
                        file_types=[".pdf", ".docx", ".txt"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Upload & Index Document", variant="primary")
                    upload_status = gr.Textbox(
                        label="Ingestion Status",
                        interactive=False,
                        lines=5,
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Document Library (SQLite & ChromaDB)")
                    doc_table = gr.Dataframe(
                        headers=["ID", "Filename", "Type", "Pages", "Chunks", "Size"],
                        datatype=["number", "str", "str", "number", "number", "str"],
                        value=get_documents_table(),
                        interactive=False,
                    )

                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                        del_id_input = gr.Textbox(
                            label="Document ID to Delete",
                            placeholder="e.g. 1",
                            scale=1,
                        )
                        del_btn = gr.Button("🗑️ Delete Document", variant="stop", size="sm")
                        clear_all_btn = gr.Button("⚠️ Clear All", variant="stop", size="sm")

        # --------------------------------------------------------------------
        # Tab 2: Ask & Query (RAG Pipeline)
        # --------------------------------------------------------------------
        with gr.TabItem("💬 2. Query Pipeline"):
            gr.Markdown("### Ask Questions Grounded in Your Documents")

            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What is the policy for multi-factor authentication?",
                        lines=2,
                    )
                    ask_btn = gr.Button("Ask Question", variant="primary")

                    answer_output = gr.Markdown(label="Generated Answer")

                    with gr.Accordion("Retrieved Sources & Chunks", open=True):
                        sources_output = gr.Markdown(label="Sources")

                with gr.Column(scale=1):
                    gr.Markdown("### Pipeline Architecture Controls")
                    doc_filter = gr.Dropdown(
                        label="Filter by Documents (Optional)",
                        choices=get_document_choices(),
                        multiselect=True,
                        info="Omit to search across all uploaded documents.",
                    )
                    hybrid_toggle = gr.Checkbox(
                        label="Enable Hybrid Search (BM25 + Dense RRF)",
                        value=True,
                        info="Combines keyword matching with semantic vector similarity.",
                    )
                    reranker_toggle = gr.Checkbox(
                        label="Enable Cross-Encoder Reranker (FlashRank)",
                        value=True,
                        info="Re-scores top candidates using local ONNX cross-encoder.",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=15,
                        value=5,
                        step=1,
                        label="Top-K Chunks to LLM",
                    )
                    model_selector = gr.Dropdown(
                        label="LLM Model",
                        choices=[
                            "gemini-3.5-flash",
                            "gemini-3.5-flash-lite",
                            "gemini-flash-lite-latest",
                            "gemini-3.6-flash",
                            "gemini-3.8-flash",
                        ],
                        value="gemini-3.5-flash",
                        info="Switch model on the fly if one is experiencing high demand (503).",
                    )

        # --------------------------------------------------------------------
        # Tab 3: RAGAS Evaluation & Ablation
        # --------------------------------------------------------------------
        with gr.TabItem("📊 3. RAGAS Evaluation"):
            gr.Markdown("### Live RAGAS Evaluation & Architectural Ablations")

            with gr.Tabs():
                # Sub-tab: Single-Query Live Eval
                with gr.TabItem("Single-Turn Evaluation"):
                    gr.Markdown(
                        "Evaluate the quality of a single answer using Ragas metrics. "
                        "Optionally provide a reference ground-truth answer."
                    )
                    with gr.Row():
                        eval_q = gr.Textbox(
                            label="Evaluation Question",
                            placeholder="Enter query to evaluate...",
                            lines=2,
                        )
                        eval_gt = gr.Textbox(
                            label="Ground Truth Reference Answer (Optional)",
                            placeholder="Expected reference answer (needed for Context Recall/Precision)...",
                            lines=2,
                        )

                    eval_single_btn = gr.Button("Run RAGAS Single-Turn Evaluation", variant="primary")

                    eval_summary = gr.Markdown(label="Answer & Context Details")

                    with gr.Row():
                        m_faith = gr.Textbox(label="Faithfulness", value="-", interactive=False)
                        m_relevancy = gr.Textbox(label="Answer Relevancy", value="-", interactive=False)
                        m_precision = gr.Textbox(label="Context Precision", value="-", interactive=False)
                        m_recall = gr.Textbox(label="Context Recall", value="-", interactive=False)

                # Sub-tab: Benchmark Ablation Study
                with gr.TabItem("Benchmark Ablation Study"):
                    gr.Markdown(
                        "Run the benchmark test set against all 3 architectural variations "
                        "(Vector Only vs. Hybrid vs. Hybrid+Reranker) to demonstrate performance gains."
                    )
                    with gr.Row():
                        eval_mode = gr.Radio(
                            label="Evaluation Mode",
                            choices=[
                                "Fast Offline Mode (Deterministic)",
                                "Full Online Mode (Google Gemini LLM Judge)",
                            ],
                            value="Fast Offline Mode (Deterministic)",
                        )
                        run_ablation_btn = gr.Button("Run Benchmark Ablation Study", variant="primary")

                    ablation_table_output = gr.Markdown(label="Ablation Results")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    # Tab 1: Upload & Delete
    upload_btn.click(
        fn=upload_and_index_file,
        inputs=[file_input],
        outputs=[upload_status, doc_table, doc_filter],
    )
    refresh_btn.click(
        fn=lambda: (get_documents_table(), gr.update(choices=get_document_choices())),
        inputs=[],
        outputs=[doc_table, doc_filter],
    )
    del_btn.click(
        fn=delete_single_document,
        inputs=[del_id_input],
        outputs=[upload_status, doc_table, doc_filter],
    )
    clear_all_btn.click(
        fn=clear_all_documents,
        inputs=[],
        outputs=[upload_status, doc_table, doc_filter],
    )

    # Tab 2: Query RAG
    ask_btn.click(
        fn=query_rag_pipeline,
        inputs=[
            question_input,
            doc_filter,
            hybrid_toggle,
            reranker_toggle,
            top_k_slider,
            model_selector,
        ],
        outputs=[answer_output, sources_output],
    )

    # Tab 3: Ragas Evals
    eval_single_btn.click(
        fn=evaluate_single_query_ragas,
        inputs=[
            eval_q,
            eval_gt,
            hybrid_toggle,
            reranker_toggle,
            top_k_slider,
        ],
        outputs=[eval_summary, m_faith, m_relevancy, m_precision, m_recall],
    )
    run_ablation_btn.click(
        fn=run_benchmark_ablation_gui,
        inputs=[eval_mode],
        outputs=[ablation_table_output],
    )


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    print(f"\nStarting RAG Pipeline Studio on http://127.0.0.1:{port} ...\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
