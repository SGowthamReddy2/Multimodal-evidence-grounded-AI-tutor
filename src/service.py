"""TutorService: Core business logic coordinator for the Multimodal AI Tutor."""

import logging
from typing import Any, Dict, List, Optional

from src.generation.abstention import (
    ABSTENTION_MESSAGE,
    QUALIFIED_PREFIX,
    AbstentionPolicy,
)
from src.generation.llm import GroundedGenerator
from src.generation.validator import CitationValidator, ValidationResult
from src.memory.controller import LearnerMemoryController
from src.retrieval.formatter import format_context_for_llm, format_evidence_badges
from src.retrieval.retriever import HybridRetriever
from src.storage import QdrantStorageManager

logger = logging.getLogger(__name__)


class TutorService:
    """Coordinates retrieval, generation, citation validation, and memory."""

    def __init__(
        self,
        storage: Optional[QdrantStorageManager] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.storage = storage or QdrantStorageManager()
        self.retriever = HybridRetriever(storage=self.storage)
        self.generator = GroundedGenerator(api_key=api_key)
        self.validator = CitationValidator(storage=self.storage)
        self.abstention_policy = AbstentionPolicy()
        self.memory = LearnerMemoryController()

    def answer_query(
        self,
        query: str,
        learner_id: str = "student_01",
        config: str = "A",
    ) -> Dict[str, Any]:
        """Execute end-to-end tutor query answering."""
        logger.info("Answering query: '%s' (learner=%s, config=%s)", query, learner_id, config)

        # 1. Fetch active learner profile
        profile = self.memory.get_active_profile(learner_id)

        # 2. Retrieve multimodal evidence
        evidence_list = self.retriever.retrieve(query=query, config=config)

        # 3. Evaluate Abstention Policy
        decision, top_score = self.abstention_policy.evaluate(evidence_list)
        logger.info("Abstention decision: %s (score=%.4f)", decision, top_score)

        if decision == "ABSTAIN":
            return {
                "answer": ABSTENTION_MESSAGE,
                "decision": decision,
                "top_score": top_score,
                "evidence_badges": [],
                "validation": ValidationResult(
                    is_valid=True,
                    total_claims=0,
                    valid_citations=[],
                    invalid_citations=[],
                    superseded_citations=[],
                    sanitized_response=ABSTENTION_MESSAGE,
                ),
                "primary_video_timestamp_ms": None,
                "primary_asset_path": None,
                "active_profile": profile,
            }

        # 4. Format context block for LLM
        context_block = format_context_for_llm(evidence_list)

        # 5. Generate Grounded Explanation
        raw_response = self.generator.generate_response(
            query=query,
            context_block=context_block,
            learner_profile=profile,
            config=config,
        )

        # 6. Validate Citations
        validation = self.validator.validate(raw_response)

        # 7. Apply qualification prefix if needed
        final_answer = validation.sanitized_response
        if decision == "QUALIFIED":
            final_answer = QUALIFIED_PREFIX + final_answer

        # 8. Find primary media asset to navigate UI
        primary_ts = None
        primary_asset = None
        for ev in evidence_list:
            if ev.start_ms > 0 and primary_ts is None:
                primary_ts = ev.start_ms
            if ev.slide_asset_path and primary_asset is None:
                primary_asset = ev.slide_asset_path
            elif ev.frame_asset_path and primary_asset is None:
                primary_asset = ev.frame_asset_path

        badges = format_evidence_badges(evidence_list)

        return {
            "answer": final_answer,
            "decision": decision,
            "top_score": top_score,
            "evidence_badges": badges,
            "validation": validation,
            "primary_video_timestamp_ms": primary_ts,
            "primary_asset_path": primary_asset,
            "active_profile": profile,
        }

