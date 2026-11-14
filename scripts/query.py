"""CLI query runner for testing retrieval and grounded responses from the terminal."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

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
    parser.add_argument("--query", "-q", type=str, default=None, help="Question to ask (omit if in interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive multi-turn dialogue session")
    parser.add_argument("--session", "-s", type=str, default=None, help="Existing session ID to resume")
    parser.add_argument("--learner", "-l", type=str, default="student_01", help="Learner ID (default: student_01)")
    parser.add_argument("--config", "-c", type=str, default="A", choices=["A", "B"], help="Config profile (A or B)")
    parser.add_argument("--course", type=str, default=None, help="Filter retrieval by Course ID (e.g. mit-6.006)")
    parser.add_argument("--lecture", type=str, default=None, help="Filter retrieval by Lecture ID (e.g. lecture-03)")
    args = parser.parse_args()

    service = TutorService()

    def run_single(query_text: str, sess_id: Optional[str]) -> str:
        res = service.answer_query(
            query=query_text,
            learner_id=args.learner,
            session_id=sess_id,
            config=args.config,
            course_id=args.course,
            lecture_id=args.lecture,
        )

        active_sess = res.get("session_id", sess_id)
        decision = res.get("decision", "FULL")
        score = res.get("top_score", 0.0)
        intent = res.get("intent", "academic_rag")
        memories = res.get("proposed_memories", [])

        print("\n" + "=" * 60)
        print(f"QUERY:    {query_text}")
        print(f"SESSION:  {active_sess} | INTENT: {intent.upper()} | DECISION: {decision} (Score: {score:.4f})")
        if memories:
            print(f"MEMORIES: {len(memories)} trait(s) captured -> {[m['value'] for m in memories]}")
        print("=" * 60)
        print("\nTUTOR RESPONSE:\n")
        print(res["answer"])
        print("\n" + "=" * 60)
        if res.get("evidence_badges"):
            print("RETRIEVED CITATIONS:")
            for b in res["evidence_badges"]:
                print(f" - [{b['evidence_id']}] ({b['modality']}, score={b['score']}) span: {b['snippet']}")
            print("=" * 60 + "\n")
        return active_sess

    if args.interactive:
        current_sess = args.session or service.memory.create_session(args.learner, "CLI Interactive Session")
        print(f"\n🎓 Starting Interactive Multi-Turn Tutor Session ({current_sess})")
        print("Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_msg = input("\nStudent: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ("exit", "quit", "q"):
                    print("Exiting session.")
                    break
                current_sess = run_single(user_msg, current_sess)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        if not args.query:
            parser.error("--query is required unless using --interactive mode")
        run_single(args.query, args.session)


if __name__ == "__main__":
    main()
