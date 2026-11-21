"""Unit tests for QueryIntentClassifier (Milestones M5 & M6).

Tests:
1. Greetings and pleasantries routed to GREETING_CHITCHAT.
2. Personal study disclosures routed to LEARNER_PROFILE_UPDATE.
3. Compound queries routed to ACADEMIC_WITH_MEMORY.
4. Technical questions routed to ACADEMIC_RAG.
"""

import unittest
from src.generation.intent import QueryIntent, QueryIntentClassifier


class TestQueryIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = QueryIntentClassifier()

    def test_greeting_chitchat(self):
        self.assertEqual(
            self.classifier.classify("Hello! How can you help me today?"),
            QueryIntent.GREETING_CHITCHAT,
        )
        self.assertEqual(
            self.classifier.classify("Good morning, who are you?"),
            QueryIntent.GREETING_CHITCHAT,
        )
        self.assertEqual(
            self.classifier.classify("Thanks for the help!"),
            QueryIntent.GREETING_CHITCHAT,
        )

    def test_learner_profile_update(self):
        # Query with learner signals and no explicit technical query
        self.assertEqual(
            self.classifier.classify(
                "I am struggling with recursion trees and my midterm is next Monday.",
                has_learner_signals=True,
            ),
            QueryIntent.LEARNER_PROFILE_UPDATE,
        )

    def test_compound_query(self):
        # Query with learner signals and an explicit technical algorithmic question
        self.assertEqual(
            self.classifier.classify(
                "I have an exam tomorrow, can you explain what is the worst-case runtime of merge sort?",
                has_learner_signals=True,
            ),
            QueryIntent.ACADEMIC_WITH_MEMORY,
        )

    def test_academic_rag(self):
        self.assertEqual(
            self.classifier.classify("What is the recurrence relation for merge sort?"),
            QueryIntent.ACADEMIC_RAG,
        )
        self.assertEqual(
            self.classifier.classify("Why is insertion sort in-place?"),
            QueryIntent.ACADEMIC_RAG,
        )


if __name__ == "__main__":
    unittest.main()
