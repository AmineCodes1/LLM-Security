from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_rag.attack_simulation import IndirectPromptInjectionSimulator
from aegis_rag.config import AppSettings
from aegis_rag.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean and poisoned documents for indirect prompt injection testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where generated documents will be written",
    )
    parser.add_argument(
        "--stem",
        default="vendor_maintenance_bulletin",
        help="Base name for output files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings.from_env()
    setup_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    simulator = IndirectPromptInjectionSimulator()

    generated = simulator.generate(
        output_dir=args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
    )

    logger.info("Created clean document: %s", generated.clean_path)
    logger.info("Created poisoned document: %s", generated.poisoned_path)


if __name__ == "__main__":
    main()
