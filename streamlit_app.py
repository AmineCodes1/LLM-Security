from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_rag.config import AppSettings
from aegis_rag.guardrails import Shield
from aegis_rag.ingestion import DocumentIngestionService
from aegis_rag.logging_config import setup_logging
from aegis_rag.rag_pipeline import RagPipeline

setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def build_services() -> tuple[RagPipeline, Shield, AppSettings, DocumentIngestionService]:
    settings = AppSettings.from_env()
    # Reuse long-lived clients in Streamlit to avoid reloading models every interaction.
    shield = Shield()
    ingestor = DocumentIngestionService(settings)
    return RagPipeline(settings, shield), shield, settings, ingestor


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _log_event(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.system_logs.append(f"{timestamp} | {message}")


def main() -> None:
    st.set_page_config(page_title="Project Aegis-RAG", layout="wide")
    st.title("Project Aegis-RAG")
    st.caption("Local RAG playground with optional guardrails")

    _ensure_state()
    rag_pipeline, shield, settings, ingestor = build_services()

    st.sidebar.header("Runtime")
    shield_enabled = st.sidebar.toggle("Enable Shield", value=settings.shield_enabled)
    use_llm_judge = st.sidebar.toggle(
        "Use LLM Judge (flagged only)",
        value=settings.shield_use_llm_judge_on_flag,
    )
    sanitize_policy = st.sidebar.selectbox(
        "Sanitize policy",
        options=["redact", "drop", "replace"],
        index=["redact", "drop", "replace"].index(settings.shield_sanitize_policy),
    )
    st.sidebar.write(f"Model: {settings.model_name}")
    st.sidebar.write(f"Top-k retrieval: {settings.retrieval_k}")

    st.sidebar.header("Ingestion")
    uploads = st.sidebar.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if st.sidebar.button("Add to knowledge base"):
        if not uploads:
            st.sidebar.warning("Select at least one file to ingest.")
        else:
            upload_dir = Path("data/raw/uploads") / uuid4().hex
            upload_dir.mkdir(parents=True, exist_ok=True)
            for uploaded in uploads:
                filename = Path(uploaded.name).name
                target_path = upload_dir / filename
                target_path.write_bytes(uploaded.getbuffer())
            result = ingestor.ingest_directory_with_stats(upload_dir)
            _log_event(
                "Ingested files: "
                f"docs={result.source_documents}, chunks={result.total_chunks}, "
                f"indexed={result.indexed_chunks}, dupes={result.skipped_duplicates}"
            )
            st.sidebar.success("Files ingested into ChromaDB")

    # Apply runtime shield settings
    settings.shield_enabled = shield_enabled
    settings.shield_use_llm_judge_on_flag = use_llm_judge
    settings.shield_sanitize_policy = sanitize_policy

    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_input = st.chat_input("Ask a question")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    try:
                        result = rag_pipeline.query(user_input)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Query failed")
                        st.error(f"Query failed: {exc}")
                        return

                    st.session_state.last_result = result
                    answer = result.answer
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    shield_report = result.shield_report
                    if shield_report and shield_report.actions_taken:
                        _log_event(
                            "Shield actions: "
                            + ", ".join(shield_report.actions_taken)
                        )

    with side_col:
        st.subheader("Retrieved Context")
        last_result = st.session_state.last_result
        if last_result is None:
            st.info("Run a query to see retrieved context.")
        else:
            shield_report = last_result.shield_report
            threat_detected = bool(
                shield_report and (shield_report.flagged_chunks or shield_report.actions_taken)
            )
            if threat_detected:
                st.warning("Threat detected in retrieved context")
            else:
                st.success("No threat detected")

            if shield_report and shield_report.reasons:
                st.caption("Threat reasons")
                for reason in shield_report.reasons:
                    st.write(f"- {reason}")

            for index, chunk in enumerate(last_result.retrieved_context, start=1):
                label = f"Chunk {index}"
                if shield_report and (index - 1) in (shield_report.flagged_chunks or []):
                    st.warning(f"{label} flagged")
                with st.expander(label, expanded=False):
                    st.write(chunk.content)

        st.subheader("System Logs")
        if st.session_state.system_logs:
            for entry in reversed(st.session_state.system_logs[-50:]):
                st.write(entry)
        else:
            st.info("No system events yet.")


if __name__ == "__main__":
    main()
