from __future__ import annotations

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import AppSettings

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Ingestion is isolated so indexing can run offline and on-demand."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def ingest_directory(self, input_dir: Path) -> int:
        documents = self._load_documents(input_dir)
        if not documents:
            logger.warning("No documents found in %s", input_dir)
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        vector_store = self._vector_store()
        vector_store.add_documents(chunks)

        logger.info(
            "Ingestion complete: %s documents loaded and %s chunks indexed.",
            len(documents),
            len(chunks),
        )
        return len(chunks)

    def _load_documents(self, input_dir: Path) -> list[Document]:
        documents: list[Document] = []

        for pdf_path in input_dir.rglob("*.pdf"):
            documents.extend(PyPDFLoader(str(pdf_path)).load())

        for txt_path in input_dir.rglob("*.txt"):
            documents.extend(TextLoader(str(txt_path), encoding="utf-8").load())

        logger.info("Loaded %s source documents from %s", len(documents), input_dir)
        return documents

    def _vector_store(self) -> Chroma:
        embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        return Chroma(
            collection_name=self.settings.collection_name,
            persist_directory=str(self.settings.chroma_persist_dir),
            embedding_function=embeddings,
        )
