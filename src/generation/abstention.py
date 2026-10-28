"""Tri-State Epistemic Abstention Policy Manager.

Evaluates retrieval confidence to enforce:
1. State 1: Full Confident Answer (Score >= 0.65)
2. State 2: Qualified Answer with Hedging / Warning (0.45 <= Score < 0.65)
3. State 3: Explicit Abstention (Score < 0.45 or empty retrieval)
"""

import logging
from typing import Dict, List, Tuple
from src.config import ABSTENTION_FULL_THRESHOLD, ABSTENTION_QUALIFIED_THRESHOLD
from src.models import RetrievedEvidence

logger = logging.getLogger(__name__)

ABSTENTION_MESSAGE = (
    "I cannot verify this from the approved course material. "
    "This question appears to fall outside the scope of MIT 6.006 (Introduction to Algorithms), "
    "or the provided lecture records do not contain sufficient evidence to answer with certainty."
)

QUALIFIED_PREFIX = (
    "> ℹ️ **Note**: The available lecture evidence has moderate confidence for this question. "
    "Here is what the course material indicates:\n\n"
)


class AbstentionPolicy:
    """Evaluates evidence scores against calibrated epistemic thresholds."""

    def __init__(
        self,
        full_threshold: float = ABSTENTION_FULL_THRESHOLD,
        qualified_threshold: float = ABSTENTION_QUALIFIED_THRESHOLD,
    ) -> None:
        self.full_threshold = full_threshold
        self.qualified_threshold = qualified_threshold

    def evaluate(self, evidence_list: List[RetrievedEvidence]) -> Tuple[str, float]:
        """Evaluate retrieval candidates and return (decision, max_score).

        Returns
        -------
        Tuple[str, float]
            decision: 'FULL', 'QUALIFIED', or 'ABSTAIN'
            max_score: Highest retrieval/rerank score among candidates
        """
        if not evidence_list:
            return "ABSTAIN", 0.0

        max_score = max(ev.score for ev in evidence_list)
        logger.info("Abstention evaluation: max_score=%.4f (full=%.2f, qual=%.2f)", max_score, self.full_threshold, self.qualified_threshold)

        if max_score >= self.full_threshold:
            return "FULL", max_score
        elif max_score >= self.qualified_threshold:
            return "QUALIFIED", max_score
        else:
            return "ABSTAIN", max_score

