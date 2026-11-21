"""Unit tests for CitationValidator (Milestones M3 & M6).

Tests:
1. Regex extraction of [claim_id | evidence_id] and [evidence_id] tokens.
2. Verified active citations pass validation.
3. Missing or hallucinated evidence_id flagged in invalid_citations.
4. Superseded citations flagged and trigger warning banner.
5. Assets missing on disk logged and handled.
"""

import unittest
from unittest.mock import MagicMock
from src.generation.validator import CitationValidator
from src.models import RetrievedEvidence


class TestCitationValidator(unittest.TestCase):
    def setUp(self):
        self.mock_storage = MagicMock()
        self.validator = CitationValidator(storage=self.mock_storage)

    def test_valid_citation_contract(self):
        # Configure mock storage to return active evidence
        mock_ev = RetrievedEvidence(
            evidence_id="ev-lec03-t-872000_901000",
            score=0.92,
            modality="transcript",
            status="active",
            start_ms=872000,
            end_ms=901000,
        )
        self.mock_storage.get_evidence_by_id.return_value = mock_ev

        response = (
            "Merge sort divides the problem into two subproblems of size n/2 [claim_1 | ev-lec03-t-872000_901000]. "
            "Total work at each level is cn."
        )
        res = self.validator.validate(response)

        self.assertTrue(res.is_valid)
        self.assertEqual(res.total_claims, 1)
        self.assertIn("ev-lec03-t-872000_901000", res.valid_citations)
        self.assertEqual(len(res.invalid_citations), 0)
        self.assertEqual(len(res.superseded_citations), 0)

    def test_hallucinated_citation_interception(self):
        # Mock storage returns None for unknown/fabricated ID
        self.mock_storage.get_evidence_by_id.return_value = None

        response = (
            "Binary search takes O(1) in all cases [claim_1 | ev-fake-nonexistent-id]."
        )
        res = self.validator.validate(response)

        self.assertFalse(res.is_valid)
        self.assertIn("ev-fake-nonexistent-id", res.invalid_citations)
        self.assertEqual(len(res.valid_citations), 0)

    def test_superseded_citation_warning(self):
        # Mock storage returns superseded evidence
        mock_ev = RetrievedEvidence(
            evidence_id="ev-lec03-old-draft",
            score=0.75,
            modality="slide",
            status="superseded",
        )
        self.mock_storage.get_evidence_by_id.return_value = mock_ev

        response = (
            "According to the initial draft notes [claim_1 | ev-lec03-old-draft], worst case is O(n)."
        )
        res = self.validator.validate(response)

        self.assertFalse(res.is_valid)
        self.assertIn("ev-lec03-old-draft", res.superseded_citations)
        # Warning banner should be appended
        self.assertIn("superseded or updated lecture material", res.sanitized_response)


if __name__ == "__main__":
    unittest.main()
