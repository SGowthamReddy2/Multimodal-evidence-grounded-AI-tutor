"""CLI runner for filtering raw keyframes into mature pedagogical blackboard/slide frames."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.keyframe_filter import main

if __name__ == "__main__":
    main()
