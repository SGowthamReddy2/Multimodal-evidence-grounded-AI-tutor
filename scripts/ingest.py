"""CLI runner for ingesting multi-course lecture assets into the Multimodal AI Tutor."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure Windows terminal prints unicode mathematical symbols without charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import DEFAULT_COURSE_ID, RAW_DIR
from src.ingestion.pipeline import (
    IngestionPipeline,
    discover_courses_and_lectures,
    resolve_lecture_dir,
)

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
        help="Path or name of specific lecture folder (e.g. data/raw/mit-6.006/lecture-03, mit-6.006/lecture-03, or lecture-03)",
    )
    parser.add_argument(
        "--course",
        type=str,
        default=DEFAULT_COURSE_ID,
        help=f"Course identifier (default: {DEFAULT_COURSE_ID})",
    )
    parser.add_argument(
        "--lecture",
        type=str,
        default=None,
        help="Lecture identifier (e.g. lecture-03)",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM image captioning (skips API calls)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover and ingest all courses and lectures in data/raw/",
    )
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    enable_vlm = not args.no_vlm

    try:
        if args.lecture_dir or args.lecture:
            raw_target = args.lecture_dir or args.lecture
            target_dir = resolve_lecture_dir(raw_target, course_id=args.course)
            lec_id = args.lecture or target_dir.name
            pipeline.ingest_lecture(
                lecture_dir=target_dir,
                course_id=args.course,
                lecture_id=lec_id,
                enable_vlm=enable_vlm,
            )
        else:
            # Default: auto-discover and ingest all courses & lectures
            logger.info("Auto-discovering courses and lectures in %s...", RAW_DIR)
            pipeline.ingest_all(base_dir=RAW_DIR, enable_vlm=enable_vlm)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
