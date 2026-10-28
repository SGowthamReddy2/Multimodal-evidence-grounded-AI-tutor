"""CLI query runner for testing retrieval and grounded responses from the terminal."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.service import TutorService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    # Ensure stdout handles UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query the Multimodal AI Tutor via CLI.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Question to ask.")
    parser.add_argument("--config", "-c", type=str, default="A", choices=["A", "B"], help="Config profile (A or B)")
    args = parser.parse_args()

    service = TutorService()
    result = service.answer_query(query=args.query, config=args.config)

    print("\n" + "=" * 60)
    print(f"QUERY: {args.query}")
    print(f"DECISION: {result['decision']} (Score: {result['top_score']:.4f})")
    print("=" * 60)
    print("\nTUTOR RESPONSE:\n")
    print(result["answer"])
    print("\n" + "=" * 60)
    print("RETRIEVED CITATIONS:")
    for b in result["evidence_badges"]:
        print(f" - [{b['evidence_id']}] ({b['modality']}, score={b['score']}) span: {b['snippet']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
