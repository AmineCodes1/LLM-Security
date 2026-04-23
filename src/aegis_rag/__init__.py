"""Project Aegis-RAG package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .attack_simulation import AttackDocumentSet, IndirectPromptInjectionSimulator
    from .config import AppSettings
    from .ingestion import ChromaDocumentStore, DocumentIngestionService, IngestionResult
    from .rag_pipeline import RagPipeline

__all__ = [
    "AttackDocumentSet",
    "AppSettings",
    "ChromaDocumentStore",
    "DocumentIngestionService",
    "IndirectPromptInjectionSimulator",
    "IngestionResult",
    "RagPipeline",
]


def __getattr__(name: str) -> Any:
    if name == "AttackDocumentSet":
        from .attack_simulation import AttackDocumentSet

        return AttackDocumentSet
    if name == "IndirectPromptInjectionSimulator":
        from .attack_simulation import IndirectPromptInjectionSimulator

        return IndirectPromptInjectionSimulator
    if name == "AppSettings":
        from .config import AppSettings

        return AppSettings
    if name == "ChromaDocumentStore":
        from .ingestion import ChromaDocumentStore

        return ChromaDocumentStore
    if name == "DocumentIngestionService":
        from .ingestion import DocumentIngestionService

        return DocumentIngestionService
    if name == "IngestionResult":
        from .ingestion import IngestionResult

        return IngestionResult
    if name == "RagPipeline":
        from .rag_pipeline import RagPipeline

        return RagPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
