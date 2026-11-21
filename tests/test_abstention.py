"""Unit tests for AbstentionPolicy and Epistemic Humility (Milestone M6).

Tests:
1. Empty evidence list produces explicit ABSTAIN.
2. Retrieval score >= 0.65 produces FULL answer state.
3. 0.45 <= score < 0.65 produces QUALIFIED hedge state.
4. Score < 0.45 produces explicit ABSTAIN state.
"""

import unittest
from src.generation.abstention import AbstentionPolicy, ABSTENTION_MESSAGE
from src.models import RetrievedEvidence


class TestAbstentionPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = AbstentionPolicy(full_threshold=0.65, qualified_threshold=0.45)

    def test_empty_evidence_abstains(self):
        decision, score = self.policy.evaluate([])
        self.assertEqual(decision, "ABSTAIN")
        self.assertEqual(score, 0.0)

    def test_high_confidence_full(self):
        ev = [
            RetrievedEvidence(evidence_id="ev-1", score=0.88, modality="transcript"),
            RetrievedEvidence(evidence_id="ev-2", score=0.72, modality="slide"),
        ]
        decision, score = self.policy.evaluate(ev)
        self.assertEqual(decision, "FULL")
        self.assertAlmostEqual(score, 0.88)

    def test_moderate_confidence_qualified(self):
        ev = [
            RetrievedEvidence(evidence_id="ev-1", score=0.55, modality="transcript"),
            RetrievedEvidence(evidence_id="ev-2", score=0.48, modality="slide"),
        ]
        decision, score = self.policy.evaluate(ev)
        self.assertEqual(decision, "QUALIFIED")
        self.assertAlmostEqual(score, 0.55)

    def test_low_confidence_abstains(self):
        ev = [
            RetrievedEvidence(evidence_id="ev-1", score=0.32, modality="transcript"),
            RetrievedEvidence(evidence_id="ev-2", score=0.20, modality="slide"),
        ]
        decision, score = self.policy.evaluate(ev)
        self.assertEqual(decision, "ABSTAIN")
        self.assertAlmostEqual(score, 0.32)

    def test_abstention_message_content(self):
        self.assertIn("cannot verify this", ABSTENTION_MESSAGE.lower())
        self.assertIn("mit 6.006", ABSTENTION_MESSAGE.lower())


if __name__ == "__main__":
    unittest.main()
