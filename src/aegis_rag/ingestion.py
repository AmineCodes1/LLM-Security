from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import AppSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    source_documents: int
    total_chunks: int
    indexed_chunks: int
    skipped_duplicates: int


class ChromaDocumentStore:
    """Thin abstraction around Chroma for indexing and top-k similarity search."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k if k is not None else self.settings.retrieval_k
        return self._vector_store().similarity_search(query, k=top_k)

    def add_documents(self, chunks: list[Document]) -> tuple[int, int]:
        if not chunks:
            return 0, 0

        ids = [self._chunk_id(chunk) for chunk in chunks]
        vector_store = self._vector_store()

        existing = vector_store.get(ids=ids, include=[])
        existing_ids = set(existing.get("ids", []))

        new_documents: list[Document] = []
        new_ids: list[str] = []
        skipped_duplicates = 0

        for doc, chunk_id in zip(chunks, ids):
            if chunk_id in existing_ids or chunk_id in new_ids:
                skipped_duplicates += 1
                continue
            new_documents.append(doc)
            new_ids.append(chunk_id)

        if new_documents:
            vector_store.add_documents(new_documents, ids=new_ids)

        return len(new_documents), skipped_duplicates

    def _vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.settings.collection_name,
            persist_directory=str(self.settings.chroma_persist_dir),
            embedding_function=self._embeddings,
        )

    def _chunk_id(self, chunk: Document) -> str:
        source = str(chunk.metadata.get("source", ""))
        page = str(chunk.metadata.get("page", ""))
        start_index = str(chunk.metadata.get("start_index", ""))
        payload = f"{source}|{page}|{start_index}|{chunk.page_content.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DocumentIngestionService:
    """Ingestion is isolated so indexing can run offline and on-demand."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.store = ChromaDocumentStore(settings)

    def ingest_directory(self, input_dir: Path) -> int:
        result = self.ingest_directory_with_stats(input_dir)
        return result.indexed_chunks

    def ingest_directory_with_stats(self, input_dir: Path) -> IngestionResult:
        documents = self._load_documents(input_dir)
        if not documents:
            logger.warning("No documents found in %s", input_dir)
            return IngestionResult(
                source_documents=0,
                total_chunks=0,
                indexed_chunks=0,
                skipped_duplicates=0,
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)

        indexed_chunks, skipped_duplicates = self.store.add_documents(chunks)

        result = IngestionResult(
            source_documents=len(documents),
            total_chunks=len(chunks),
            indexed_chunks=indexed_chunks,
            skipped_duplicates=skipped_duplicates,
        )

        logger.info(
            "Ingestion complete: %s source docs, %s chunks, %s indexed, %s duplicates skipped.",
            result.source_documents,
            result.total_chunks,
            result.indexed_chunks,
            result.skipped_duplicates,
        )
        return result

    def _load_documents(self, input_dir: Path) -> list[Document]:
        documents: list[Document] = []

        for pdf_path in input_dir.rglob("*.pdf"):
            documents.extend(PyPDFLoader(str(pdf_path)).load())

        for txt_path in input_dir.rglob("*.txt"):
            documents.extend(TextLoader(str(txt_path), encoding="utf-8").load())

        logger.info("Loaded %s source documents from %s", len(documents), input_dir)
        return documents
