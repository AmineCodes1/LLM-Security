from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_rag.config import AppSettings
from aegis_rag.ingestion import DocumentIngestionService
from aegis_rag.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing PDF/TXT files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings.from_env()
    setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    ingestor = DocumentIngestionService(settings)

    result = ingestor.ingest_directory_with_stats(args.input_dir)
    logger.info(
        "Ingestion summary: source_docs=%s, chunks=%s, indexed=%s, skipped_duplicates=%s",
        result.source_documents,
        result.total_chunks,
        result.indexed_chunks,
        result.skipped_duplicates,
    )


if __name__ == "__main__":
    main()
