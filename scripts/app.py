"""Application launcher for the Grounded Multimodal AI Tutor."""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, FRAMES_DIR, PROJECT_ROOT, RAW_DIR, SLIDES_DIR
from src.ui.app import build_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


def main() -> None:
    logger.info("Starting Grounded Multimodal AI Tutor Web Interface...")
    demo = build_app()
    allowed_dirs = [
        str(PROJECT_ROOT),
        str(DATA_DIR),
        str(FRAMES_DIR),
        str(SLIDES_DIR),
        str(RAW_DIR),
    ]
    # Share=False for local development
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=allowed_dirs,
    )


if __name__ == "__main__":
    main()

