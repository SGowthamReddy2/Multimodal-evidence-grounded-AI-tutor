"""TutorService: Core business logic coordinator for the Multimodal AI Tutor.

Implements:
1. Multi-turn ephemeral dialogue sessions and persistent chat history.
2. Selective durable learner memory extraction and student confirmation gating.
3. Humane query intent routing (greetings, profile updates, academic RAG).
4. Epistemic tri-state abstention with grounded citations and source verification.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from src.generation.abstention import (
    ABSTENTION_MESSAGE,
    QUALIFIED_PREFIX,
    AbstentionPolicy,
)
from src.generation.intent import QueryIntent, QueryIntentClassifier
from src.generation.llm import GroundedGenerator
from src.generation.validator import CitationValidator, ValidationResult
from src.memory.controller import LearnerMemoryController
from src.memory.engine import LearnerMemoryEngine
from src.memory.extractor import LearnerSignalExtractor
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
        self.memory_engine = LearnerMemoryEngine(controller=self.memory)
        self.signal_extractor = LearnerSignalExtractor()
        self.intent_classifier = QueryIntentClassifier()

    def list_available_courses_and_lectures(self) -> Dict[str, List[str]]:
        """Return dict of course_id -> list of lecture_ids from storage and raw directory."""
        return self.storage.list_available_courses_and_lectures()

    def answer_query(
        self,
        query: str,
        learner_id: str = "student_01",
        session_id: Optional[str] = None,
        config: str = "A",
        course_id: Optional[Union[str, List[str]]] = None,
        lecture_id: Optional[Union[str, List[str]]] = None,
        include_visual: bool = True,
    ) -> Dict[str, Any]:
        """Execute end-to-end tutor query answering with multi-turn memory and intent routing."""
        active_session = self.memory.get_or_create_session(learner_id, session_id=session_id)

        logger.info(
            "Answering query: '%s' (learner=%s, session=%s, config=%s, course=%s, lecture=%s)",
            query,
            learner_id,
            active_session,
            config,
            course_id,
            lecture_id,
        )

        # 1. Autonomous Letta Core Memory Planning & Execution
        active_blocks = self.memory.get_active_core_memory_blocks(learner_id)
        session_history = self.memory.get_session_messages(active_session, limit=6)
        planned_actions = self.memory_engine.plan_memory_actions(
            query=query,
            active_blocks=active_blocks,
            recent_messages=session_history,
        )
        executed_actions = self.memory_engine.execute_memory_actions(planned_actions, learner_id)

        proposed_memories: List[Dict[str, Any]] = []
        for act in executed_actions:
            block_type = act.get("block", "goal")
            if block_type.endswith("s") and block_type != "mastery":
                block_type = block_type[:-1]
            proposed_memories.append({
                "memory_id": act.get("memory_id", ""),
                "type": block_type,
                "value": act.get("value", ""),
                "authority": act.get("authority", "student_authored"),
                "confidence": act.get("confidence", 0.9),
                "action": act.get("action", "CREATE"),
                "status": act.get("status", "created"),
            })

        # Fallback to signal_extractor if engine found nothing but extractor might
        if not proposed_memories:
            extracted_signals = self.signal_extractor.extract_signals(query)
            for sig in extracted_signals:
                if sig.get("authority") == "student_authored":
                    mem_id = self.memory.add_student_memory(
                        learner_id=learner_id,
                        memory_type=sig["type"],
                        value=sig["value"],
                    )
                else:
                    mem_id = self.memory.propose_memory(
                        learner_id=learner_id,
                        memory_type=sig["type"],
                        value=sig["value"],
                        provenance=f"query_inference:{query[:60]}",
                        evidence=query,
                        confidence=sig.get("confidence", 0.85),
                    )
                proposed_memories.append({**sig, "memory_id": mem_id, "action": "CREATE", "status": "created"})

        # 2. Classify high-level query intent
        intent = self.intent_classifier.classify(
            query=query,
            has_learner_signals=bool(proposed_memories),
        )

        # 3. Handle Greeting & Pleasantries (Bypasses cold RAG abstention)
        if intent == QueryIntent.GREETING_CHITCHAT:
            profile = self.memory.get_active_profile(learner_id)
            history = self.memory.get_session_messages(active_session, limit=4)
            reply = self.generator.generate_conversational_response(
                query=query,
                learner_profile=profile,
                conversation_history=history,
                context_note="Warmly welcome the student to MIT 6.006 Multimodal AI Tutor. Explain you can help with lecture videos, blackboard math, and slides.",
                config=config,
            )

            # Record turn in SQLite ephemeral buffer
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="user",
                content=query,
                intent="greeting",
            )
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="assistant",
                content=reply,
                intent="greeting",
            )

            return {
                "answer": reply,
                "decision": "CONVERSATIONAL",
                "intent": intent.value,
                "top_score": 1.0,
                "evidence_badges": [],
                "validation": ValidationResult(
                    is_valid=True,
                    total_claims=0,
                    valid_citations=[],
                    invalid_citations=[],
                    superseded_citations=[],
                    sanitized_response=reply,
                ),
                "primary_video_timestamp_ms": None,
                "primary_asset_path": None,
                "active_profile": profile,
                "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
                "executed_memory_actions": executed_actions,
                "session_id": active_session,
                "proposed_memories": proposed_memories,
            }

        # 4. Handle Learner State & Personal Study Disclosures
        if intent == QueryIntent.LEARNER_PROFILE_UPDATE:
            profile = self.memory.get_active_profile(learner_id)
            history = self.memory.get_session_messages(active_session, limit=4)
            added_traits_summary = ", ".join([f"{s['type']}: {s['value']}" for s in proposed_memories])
            note = (
                f"Student disclosed personal study state ({added_traits_summary}). "
                "Acknowledge empathetically, confirm their profile is updated, and encourage them by suggesting "
                "key topics to review (e.g. Lecture 3: Merge Sort recursion tree, Theta(n log n) runtime)."
            )
            reply = self.generator.generate_conversational_response(
                query=query,
                learner_profile=profile,
                conversation_history=history,
                context_note=note,
                config=config,
            )

            # Record turn in SQLite ephemeral buffer
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="user",
                content=query,
                intent="profile_update",
                metadata={"proposed_memories": proposed_memories},
            )
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="assistant",
                content=reply,
                intent="profile_update",
            )

            return {
                "answer": reply,
                "decision": "PROFILE_UPDATED",
                "intent": intent.value,
                "top_score": 1.0,
                "evidence_badges": [],
                "validation": ValidationResult(
                    is_valid=True,
                    total_claims=0,
                    valid_citations=[],
                    invalid_citations=[],
                    superseded_citations=[],
                    sanitized_response=reply,
                ),
                "primary_video_timestamp_ms": None,
                "primary_asset_path": None,
                "active_profile": profile,
                "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
                "executed_memory_actions": executed_actions,
                "session_id": active_session,
                "proposed_memories": proposed_memories,
            }

        # 5. Academic RAG: Retrieve multimodal evidence
        profile = self.memory.get_active_profile(learner_id)
        session_history = self.memory.get_session_messages(active_session, limit=6)

        # Multi-turn context resolution for retrieval query:
        retrieval_query = query
        prev_user_turns = [m["content"] for m in session_history if m.get("role") == "user"]
        if prev_user_turns:
            prev_q = prev_user_turns[-1]
            anaphoric_cues = {"that", "it", "this", "those", "these", "same", "again", "above", "previous", "step", "detail", "more"}
            words = set(query.lower().split())
            if words.intersection(anaphoric_cues) or len(words) < 6:
                # Clean filler from prev_q to leave core topic keywords
                clean_prev = prev_q
                for prefix in ["i have a mid examination tomorrow and i am weak in", "i am weak in", "can you explain", "what is"]:
                    if clean_prev.lower().startswith(prefix):
                        clean_prev = clean_prev[len(prefix):].strip()
                retrieval_query = f"{clean_prev} {query}".strip()
                logger.info("Enriched multi-turn retrieval query: '%s'", retrieval_query)

        evidence_list = self.retriever.retrieve(
            query=retrieval_query,
            config=config,
            include_visual=include_visual,
            course_id=course_id,
            lecture_id=lecture_id,
        )
        if not include_visual and evidence_list:
            # Ablation mode: strip all visual slide/keyframe chunks
            evidence_list = [ev for ev in evidence_list if ev.modality in ("transcript", "reading")]

        # 6. Evaluate Abstention Policy
        decision, top_score = self.abstention_policy.evaluate(evidence_list)
        logger.info("Abstention decision: %s (score=%.4f)", decision, top_score)

        if decision == "ABSTAIN":
            # Check 6A: If user disclosed profile traits, soften abstention with supportive coaching
            if proposed_memories:
                logger.info("Softening abstention for query with captured learner signals: %s", [m["value"] for m in proposed_memories])
                traits_summary = ", ".join([f"{s['type']}: {s['value']}" for s in proposed_memories])
                note = (
                    f"Student disclosed personal study state ({traits_summary}). "
                    "Their question was broad or did not match a specific factual slide statement. "
                    "Acknowledge their goals/weaknesses warmly, confirm their profile is updated, "
                    "and guide them to the relevant course topics in MIT 6.006 (e.g. Lecture 3: Sorting, Merge Sort, Recurrences)."
                )
                reply = self.generator.generate_conversational_response(
                    query=query,
                    learner_profile=profile,
                    conversation_history=session_history,
                    context_note=note,
                    config=config,
                )
                self.memory.add_chat_message(
                    session_id=active_session,
                    learner_id=learner_id,
                    role="user",
                    content=query,
                    intent="academic_rag",
                    metadata={"proposed_memories": proposed_memories},
                )
                self.memory.add_chat_message(
                    session_id=active_session,
                    learner_id=learner_id,
                    role="assistant",
                    content=reply,
                    intent="academic_rag",
                    metadata={"decision": "PROFILE_UPDATED"},
                )
                return {
                    "answer": reply,
                    "decision": "PROFILE_UPDATED",
                    "intent": intent.value,
                    "top_score": top_score,
                    "evidence_badges": [],
                    "validation": ValidationResult(
                        is_valid=True,
                        total_claims=0,
                        valid_citations=[],
                        invalid_citations=[],
                        superseded_citations=[],
                        sanitized_response=reply,
                    ),
                    "primary_video_timestamp_ms": None,
                    "primary_asset_path": None,
                    "active_profile": profile,
                    "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
                    "executed_memory_actions": executed_actions,
                    "session_id": active_session,
                    "proposed_memories": proposed_memories,
                }

            # Check 6B: If this is an anaphoric follow-up about the ongoing dialogue in this session
            followup_patterns = [
                r"\bstep\s*\d+\b",
                r"\b(?:that|this|above|previous)\s+(?:step|point|algorithm|recurrence|tree|concept|part|level)\b",
                r"\b(?:explain\s+more|elaborate|clarify|can you explain|give an example)\b",
                r"\bwhat\s+(?:did|do)\s+you\s+mean\b",
                r"\bwhy\s+is\s+that\b",
            ]
            is_dialogue_followup = bool(
                session_history
                and len(session_history) >= 2
                and any(re.search(pat, query, re.IGNORECASE) for pat in followup_patterns)
            )

            if is_dialogue_followup:
                logger.info("Answering follow-up query using ongoing conversation history context.")
                reply = self.generator.generate_conversational_response(
                    query=query,
                    learner_profile=profile,
                    conversation_history=session_history,
                    context_note="Student is asking a follow-up question about the previous explanation in this session. Clarify and elaborate directly on that step or concept in detail.",
                    config=config,
                )
                self.memory.add_chat_message(
                    session_id=active_session,
                    learner_id=learner_id,
                    role="user",
                    content=query,
                    intent="academic_followup",
                )
                self.memory.add_chat_message(
                    session_id=active_session,
                    learner_id=learner_id,
                    role="assistant",
                    content=reply,
                    intent="academic_followup",
                    metadata={"decision": "CONVERSATIONAL_FOLLOWUP"},
                )
                return {
                    "answer": reply,
                    "decision": "CONVERSATIONAL_FOLLOWUP",
                    "intent": intent.value,
                    "top_score": top_score,
                    "evidence_badges": [],
                    "validation": ValidationResult(
                        is_valid=True,
                        total_claims=0,
                        valid_citations=[],
                        invalid_citations=[],
                        superseded_citations=[],
                        sanitized_response=reply,
                    ),
                    "primary_video_timestamp_ms": None,
                    "primary_asset_path": None,
                    "active_profile": profile,
                    "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
                    "executed_memory_actions": executed_actions,
                    "session_id": active_session,
                    "proposed_memories": proposed_memories,
                }

            # Standard explicit abstention for unsupported or off-corpus queries
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="user",
                content=query,
                intent="academic_rag",
            )
            self.memory.add_chat_message(
                session_id=active_session,
                learner_id=learner_id,
                role="assistant",
                content=ABSTENTION_MESSAGE,
                intent="academic_rag",
                metadata={"decision": decision, "top_score": top_score},
            )
            return {
                "answer": ABSTENTION_MESSAGE,
                "decision": decision,
                "intent": intent.value,
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
                "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
                "executed_memory_actions": executed_actions,
                "session_id": active_session,
                "proposed_memories": proposed_memories,
            }

        # 7. Format context block and dialogue history for LLM
        context_block = format_context_for_llm(evidence_list)

        # 8. Generate Grounded Explanation with Multi-turn Ephemeral Context
        raw_response = self.generator.generate_response(
            query=query,
            context_block=context_block,
            learner_profile=profile,
            conversation_history=session_history,
            config=config,
        )

        # 9. Validate Citations
        validation = self.validator.validate(raw_response)

        # 10. Apply qualification prefix if needed
        final_answer = validation.sanitized_response
        lower_ans = final_answer.lower()
        if any(phrase in lower_ans for phrase in [
            "cannot verify this from the approved course material",
            "outside the scope of mit 6.006",
            "not covered in the provided lecture",
            "not covered in lecture",
            "cannot be established from the provided",
            "cannot answer based on the provided material",
            "cannot be verified from the approved",
            "i cannot answer",
        ]):
            decision = "ABSTAIN"
        elif decision == "QUALIFIED":
            final_answer = QUALIFIED_PREFIX + final_answer

        # 11. Find primary media asset to navigate UI
        primary_ts = None
        primary_asset = None
        primary_course = None
        primary_lecture = None
        for ev in evidence_list:
            if ev.start_ms > 0 and primary_ts is None:
                primary_ts = ev.start_ms
            if ev.slide_asset_path and primary_asset is None:
                primary_asset = ev.slide_asset_path
            elif ev.frame_asset_path and primary_asset is None:
                primary_asset = ev.frame_asset_path
            if primary_lecture is None and ev.lecture_id:
                primary_lecture = ev.lecture_id
                primary_course = ev.course_id

        badges = format_evidence_badges(evidence_list)

        # 12. Record turn in SQLite ephemeral buffer
        self.memory.add_chat_message(
            session_id=active_session,
            learner_id=learner_id,
            role="user",
            content=query,
            intent="academic_rag",
        )
        self.memory.add_chat_message(
            session_id=active_session,
            learner_id=learner_id,
            role="assistant",
            content=final_answer,
            intent="academic_rag",
            metadata={
                "decision": decision,
                "top_score": top_score,
                "badges_count": len(badges),
                "proposed_memories": proposed_memories,
            },
        )

        return {
            "answer": final_answer,
            "decision": decision,
            "intent": intent.value,
            "top_score": top_score,
            "evidence_badges": badges,
            "validation": validation,
            "primary_video_timestamp_ms": primary_ts,
            "primary_asset_path": primary_asset,
            "primary_course": primary_course,
            "primary_lecture": primary_lecture,
            "active_profile": profile,
            "core_memory_blocks": self.memory.get_active_core_memory_blocks(learner_id),
            "executed_memory_actions": executed_actions,
            "session_id": active_session,
            "proposed_memories": proposed_memories,
        }
