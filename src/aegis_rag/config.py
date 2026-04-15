from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppSettings:
    """Centralized runtime settings to keep app wiring explicit and testable."""

    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral:7b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: Path = Path("data/chroma")
    collection_name: str = "aegis_rag"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 3
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model_name=os.getenv("OLLAMA_MODEL", "mistral:7b"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")),
            collection_name=os.getenv("CHROMA_COLLECTION", "aegis_rag"),
            chunk_size=_int_from_env("CHUNK_SIZE", 500),
            chunk_overlap=_int_from_env("CHUNK_OVERLAP", 50),
            retrieval_k=_int_from_env("RETRIEVAL_K", 3),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc
