"""CLI runner for ingesting lecture assets into the Multimodal AI Tutor."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR
from src.ingestion.pipeline import IngestionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest lecture assets into Qdrant.")
    parser.add_argument(
        "--lecture-dir",
        type=str,
        default=None,
        help="Path to specific lecture folder (e.g. data/raw/lecture-03)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Scan and ingest all lecture-* folders in data/raw/",
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline()

    if args.lecture_dir:
        target_dir = Path(args.lecture_dir)
        pipeline.ingest_lecture(target_dir, lecture_id=target_dir.name)
    elif args.auto or True:
        # Default to auto scanning if no specific dir provided
        lecture_dirs = sorted(list(RAW_DIR.glob("lecture-*")))
        if not lecture_dirs:
            logger.error("No lecture-* directories found in %s", RAW_DIR)
            sys.exit(1)
        for l_dir in lecture_dirs:
            logger.info("Auto-ingesting %s...", l_dir.name)
            pipeline.ingest_lecture(l_dir, lecture_id=l_dir.name)


if __name__ == "__main__":
    main()

