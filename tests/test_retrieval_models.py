"""Unit tests for models, chunk contracts, and retrieval formatting (Milestones M1 & M2).

Tests:
1. EvidenceIdentity and AlignmentProvenance dataclass serialization.
2. MultimodalChunk to Qdrant payload conversion.
3. Context assembly and evidence badge formatting.
"""

import unittest
from src.models import EvidenceIdentity, AlignmentProvenance, MultimodalChunk, RetrievedEvidence
from src.retrieval.formatter import format_context_for_llm, format_evidence_badges


class TestRetrievalModels(unittest.TestCase):
    def test_chunk_to_payload(self):
        ident = EvidenceIdentity(
            source_id="6.006-lec03-video",
            source_url="https://ocw.mit.edu/lecture-3",
            content_version="v1.0",
            content_hash="abc123hash",
            evidence_id="ev-lec03-t-100_200",
            modality="transcript",
            status="active",
            course_id="mit-6.006",
            lecture_id="lecture-03",
        )
        prov = AlignmentProvenance(
            evidence_id="ev-lec03-t-100_200",
            start_ms=100000,
            end_ms=200000,
            transcript_span="Merge sort recursive step",
        )
        chunk = MultimodalChunk(
            identity=ident,
            provenance=prov,
            text_for_embedding="Merge sort recursive step",
        )

        payload = chunk.to_payload()
        self.assertEqual(payload["source_id"], "6.006-lec03-video")
        self.assertEqual(payload["course_id"], "mit-6.006")
        self.assertEqual(payload["lecture_id"], "lecture-03")
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["start_ms"], 100000)
        self.assertEqual(payload["end_ms"], 200000)
        self.assertEqual(payload["transcript_span"], "Merge sort recursive step")

    def test_format_context_for_llm(self):
        evidence_list = [
            RetrievedEvidence(
                evidence_id="ev-lec03-t-100_200",
                score=0.91,
                modality="transcript",
                transcript_span="The work at level 2 of the recursion tree is cn.",
                start_ms=100000,
                end_ms=200000,
                source_id="lec03",
            ),
            RetrievedEvidence(
                evidence_id="ev-lec03-s-p12",
                score=0.85,
                modality="slide",
                visual_caption="Slide 12: Merge sort tree diagram with cn total work.",
                page_number=12,
                source_id="lec03-slides",
            ),
        ]

        context_str = format_context_for_llm(evidence_list)
        self.assertIn("ev-lec03-t-100_200", context_str)
        self.assertIn("The work at level 2 of the recursion tree is cn.", context_str)
        self.assertIn("ev-lec03-s-p12", context_str)
        self.assertIn("Slide 12: Merge sort tree diagram with cn total work.", context_str)

    def test_format_evidence_badges(self):
        evidence_list = [
            RetrievedEvidence(
                evidence_id="ev-lec03-t-100_200",
                score=0.91,
                modality="transcript",
                transcript_span="The work at level 2 is cn.",
                start_ms=65000,
                end_ms=90000,
            )
        ]
        badges = format_evidence_badges(evidence_list)
        self.assertEqual(len(badges), 1)
        self.assertEqual(badges[0]["start_ms"], 65000)
        self.assertEqual(badges[0]["evidence_id"], "ev-lec03-t-100_200")


if __name__ == "__main__":
    unittest.main()
