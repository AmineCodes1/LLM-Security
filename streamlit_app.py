from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_rag.config import AppSettings
from aegis_rag.guardrails import Shield
from aegis_rag.logging_config import setup_logging
from aegis_rag.rag_pipeline import RagPipeline

setup_logging()
logger = logging.getLogger(__name__)


@st.cache_resource
def build_services() -> tuple[RagPipeline, Shield, AppSettings]:
    settings = AppSettings.from_env()
    # Reuse long-lived clients in Streamlit to avoid reloading models every interaction.
    return RagPipeline(settings), Shield(), settings


def main() -> None:
    st.set_page_config(page_title="Project Aegis-RAG", layout="wide")
    st.title("Project Aegis-RAG")
    st.caption("Local RAG playground with optional guardrails")

    rag_pipeline, shield, settings = build_services()

    st.sidebar.header("Runtime")
    shield_enabled = st.sidebar.toggle("Enable Shield", value=True)
    st.sidebar.write(f"Model: {settings.model_name}")
    st.sidebar.write(f"Top-k retrieval: {settings.retrieval_k}")

    question = st.text_input("Ask a question", placeholder="What did the document say about ...?")

    if st.button("Run Query", type="primary") and question.strip():
        try:
            result = rag_pipeline.answer_question(question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Query failed")
            st.error(f"Query failed: {exc}")
            return

        st.subheader("Retrieved Context")
        scan_result = shield.scan_context(result.documents)

        for index, doc in enumerate(result.documents, start=1):
            label = f"Chunk {index}"
            if index - 1 in scan_result.flagged_chunks:
                st.warning(f"{label} flagged by shield")
            with st.expander(label, expanded=False):
                st.write(doc.page_content)

        st.subheader("Generated Answer")
        if shield_enabled and scan_result.flagged_chunks:
            st.error("Response blocked: context contains suspicious instructions.")
            for reason in scan_result.reasons:
                st.write(f"- {reason}")
            return

        output_check = shield.validate_output(result.answer) if shield_enabled else None
        if shield_enabled and output_check and not output_check.is_safe:
            st.error("Response blocked: output validation failed.")
            for reason in output_check.reasons:
                st.write(f"- {reason}")
            return

        st.success(result.answer)


if __name__ == "__main__":
    main()
