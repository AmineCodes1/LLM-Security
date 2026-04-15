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

    chunks = ingestor.ingest_directory(args.input_dir)
    logger.info("Indexed %s chunks.", chunks)


if __name__ == "__main__":
    main()
