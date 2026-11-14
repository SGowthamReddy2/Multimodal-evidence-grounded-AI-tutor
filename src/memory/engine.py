"""Learner Memory Reasoner & Autonomous CRUD Engine (Letta / MemGPT Architecture).

Fulfills Milestone M5:
1. Contextual Intent & Memory Reasoning across dialogue turns.
2. Self-Directed Core Memory Actions: CREATE, UPDATE, DELETE, RESOLVE, NOOP.
3. Multi-Session entity resolution (e.g. updating an existing exam date instead of duplicating).
4. Sovereign Confirmation Gate preservation: direct student statements are 'student_authored',
   tutor-inferred traits are 'tutor_proposed'.
"""

import datetime
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.memory.controller import LearnerMemoryController

logger = logging.getLogger(__name__)

# Heuristic patterns for robust, offline/zero-latency detection & entity matching
EXAM_PATTERNS = [
    r"\b(?:mid\s*(?:term|sem|semester|examination)|exam|examination|quiz|test|finals?)\b",
]

GOAL_INTENT_PATTERNS = [
    (r"(?:i have|i've got|upcoming|there is|scheduled)\s+(?:a\s+)?(?:mid\s*(?:term|sem|semester|examination)|exam|examination|quiz|test|finals?)\s*([^.!?\n]*)", "Upcoming assessment: {}"),
    (r"(?:my goal is to|i want to|aiming to|hoping to)\s+([^.!?\n]+)", "Goal: {}"),
    (r"(?:preparing for|studying for)\s+(?:my\s+)?(?:mid\s*(?:term|sem|semester|examination)|exam|examination|quiz|test|finals?|interview)\s*([^.!?\n]*)", "Target preparation: {}"),
    (r"(?:target(?:ing)?|objective is)\s+([^.!?\n]+)", "Target: {}"),
]

RESCHEDULE_PATTERNS = [
    r"(?:postponed|rescheduled|moved|delayed|pushed back|shifted)\s+(?:to|until)\s+([^.!?\n]+)",
    r"(?:exam|test|midterm|midsem)\s+is\s+(?:now\s+)?(?:on|at)\s+([^.!?\n]+)",
]

MASTERY_WEAK_PATTERNS = [
    (r"(?:i am|i'm)\s+(?:really\s+)?(?:weak|bad|struggling|confused|not good)\s+(?:in|at|with)\s+([^.!?\n]+)", "Weakness / Needs support with: {}"),
    (r"(?:i struggle with|i have trouble with|i don't understand|i don't get)\s+([^.!?\n]+)", "Weakness / Difficulty with: {}"),
    (r"(?:hard for me to understand|difficult to follow)\s+([^.!?\n]+)", "Weakness / Conceptual hurdle: {}"),
    (r"(?:confused about|unclear about)\s+([^.!?\n]+)", "Weakness / Confusion: {}"),
]

MASTERY_STRONG_PATTERNS = [
    (r"(?:i am|i'm)\s+(?:good|confident|strong)\s+(?:in|at|with)\s+([^.!?\n]+)", "Mastered / confident: {}"),
    (r"(?:i already know|i have mastered|i understand|i got it|i understand now)\s+([^.!?\n]+)", "Mastered: {}"),
    (r"(?:clear to me now|finally make sense|now makes sense)\s*[:,-]?\s*([^.!?\n]*)", "Mastered: {}"),
]

MISCONCEPTION_PATTERNS = [
    (r"(?:i thought|i assumed|i believed)\s+([^.!?\n]+)", "Misconception: {}"),
    (r"(?:isn't|is not)\s+([^.!?\n]+)\s+(?:always|in-place|constant time|faster)", "Misconception / False belief: {}"),
]

RESOLVE_MISCONCEPTION_PATTERNS = [
    r"(?:i understand now|i get it now|i realize(?:d)?|my mistake|i see why)\s*[,:-]?\s*([^.!?\n]+)",
    r"(?:is not|isn't)\s+(?:actually\s+)?(?:in-place|o\(1\)|constant)\s*[,:-]?\s*([^.!?\n]*)",
]

PREFERENCE_PATTERNS = [
    (r"(?:i prefer|please explain with|can you show|show me|i want)\s+(diagrams?|visuals?|drawings?|charts?|visual intuition)", "Prefers visual diagram explanations"),
    (r"(?:i prefer|give me|show me|explain with|i want)\s+(code|pseudocode|python|implementation)", "Prefers code and pseudocode implementations"),
    (r"(?:keep it|be)\s+(concise|brief|short|simple|step by step)", "Prefers {} explanations"),
    (r"(?:more|focus on|i want|prefer)\s+(math|formal proofs?|derivations?)", "Prefers formal mathematical derivations and proofs"),
]

DELETE_PREFERENCE_PATTERNS = [
    r"(?:remove|delete|clear|stop|don't)\s+(?:my\s+)?(?:preference|prefer)\s+(?:for\s+)?([^.!?\n]+)",
    r"(?:don't give me|no more|stop giving)\s+(diagrams?|code|visuals?|math)",
]


@dataclass
class MemoryAction:
    """Letta / MemGPT self-directed memory mutation operation."""

    action: Literal["CREATE", "UPDATE", "DELETE", "RESOLVE", "NOOP"]
    block: Literal["goals", "mastery", "misconceptions", "preferences"]
    target_id: Optional[str] = None  # Existing memory_id for UPDATE, DELETE, or RESOLVE
    value: str = ""
    confidence: float = 0.85
    authority: Literal["student_authored", "tutor_proposed"] = "student_authored"
    reasoning: str = ""
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearnerMemoryEngine:
    """Reasoning engine that determines intended memory mutations from student dialogue."""

    def __init__(self, controller: Optional[LearnerMemoryController] = None) -> None:
        self.controller = controller or LearnerMemoryController()

    def plan_memory_actions(
        self,
        query: str,
        active_blocks: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MemoryAction]:
        """Analyze dialogue turn and emit structured CRUD operations over Core Memory Blocks.

        Args:
            query: Raw student input text.
            active_blocks: Current in-context Core Memory Blocks from controller.
            recent_messages: Prior dialogue turns in active chat session.

        Returns:
            List of MemoryAction items.
        """
        clean_text = query.strip()
        actions: List[MemoryAction] = []
        blocks = active_blocks or {
            "goals": [],
            "mastery": [],
            "misconceptions": [],
            "preferences": [],
        }

        # --------------------------------------------------------------
        # 1. Goals & Assessment Deadlines (CREATE or UPDATE / Reschedule)
        # --------------------------------------------------------------
        # Check for rescheduling an existing exam goal
        is_reschedule = False
        for resched_pat in RESCHEDULE_PATTERNS:
            m_resched = re.search(resched_pat, clean_text, re.IGNORECASE)
            if m_resched:
                new_date = m_resched.group(1).strip()
                # Find existing exam goal in active blocks
                for item in blocks.get("goals", []):
                    item_val = item.get("value", "").lower()
                    if any(re.search(ep, item_val) for ep in EXAM_PATTERNS):
                        actions.append(
                            MemoryAction(
                                action="UPDATE",
                                block="goals",
                                target_id=item["memory_id"],
                                value=f"Upcoming assessment: {new_date}",
                                confidence=0.95,
                                authority="student_authored",
                                reasoning=f"Student rescheduled assessment to {new_date}",
                                evidence=clean_text,
                            )
                        )
                        is_reschedule = True
                        break
            if is_reschedule:
                break

        if not is_reschedule:
            for pat, template in GOAL_INTENT_PATTERNS:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    raw_val = match.group(0).strip()
                    val = template.format(raw_val) if "{}" in template else template
                    val = re.sub(r"^(i have|i've got)\s+", "", val, flags=re.IGNORECASE).strip()
                    val = re.sub(r"\s+", " ", val)

                    # Deduplicate: check if this goal already exists or matches closely
                    existing_match = None
                    for item in blocks.get("goals", []):
                        if any(term in item.get("value", "").lower() for term in ["exam", "midterm", "midsem", "test"]) and any(term in val.lower() for term in ["exam", "midterm", "midsem", "test"]):
                            existing_match = item
                            break

                    if existing_match:
                        actions.append(
                            MemoryAction(
                                action="UPDATE",
                                block="goals",
                                target_id=existing_match["memory_id"],
                                value=val[:120],
                                confidence=0.95,
                                authority="student_authored",
                                reasoning="Student updated active goal/assessment timeline",
                                evidence=clean_text,
                            )
                        )
                    else:
                        actions.append(
                            MemoryAction(
                                action="CREATE",
                                block="goals",
                                value=val[:120],
                                confidence=0.95,
                                authority="student_authored",
                                reasoning="Student declared new learning goal or assessment target",
                                evidence=clean_text,
                            )
                        )
                    break

        # --------------------------------------------------------------
        # 2. Misconceptions: Detection & Resolution (Checked before mastery)
        # --------------------------------------------------------------
        is_resolved = False
        for res_pat in RESOLVE_MISCONCEPTION_PATTERNS:
            m_res = re.search(res_pat, clean_text, re.IGNORECASE)
            if m_res and blocks.get("misconceptions"):
                matched_snippet = m_res.group(1).strip() if m_res.groups() else clean_text
                # Find matching misconception item
                matched_item = None
                query_words = set(re.findall(r"\w+", clean_text.lower()))
                for item in blocks.get("misconceptions", []):
                    item_words = set(re.findall(r"\w+", item.get("value", "").lower()))
                    if query_words & item_words:
                        matched_item = item
                        break
                if not matched_item and blocks.get("misconceptions"):
                    matched_item = blocks["misconceptions"][0]

                if matched_item:
                    actions.append(
                        MemoryAction(
                            action="RESOLVE",
                            block="misconceptions",
                            target_id=matched_item["memory_id"],
                            value=matched_item.get("value", ""),
                            confidence=0.95,
                            authority="student_authored",
                            reasoning=f"Student resolved misconception through dialogue: '{matched_snippet[:60]}'",
                            evidence=clean_text,
                        )
                    )
                    is_resolved = True
                    break

        # Declaration of new misconception / false assumption
        if not is_resolved:
            for pat, template in MISCONCEPTION_PATTERNS:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    raw_statement = match.group(1).strip()
                    val = template.format(raw_statement) if "{}" in template else template
                    actions.append(
                        MemoryAction(
                            action="CREATE",
                            block="misconceptions",
                            value=val[:120],
                            confidence=0.85,
                            authority="student_authored",
                            reasoning="Student dialogue revealed a potential false mental model needing correction",
                            evidence=clean_text,
                        )
                    )
                    break

        # --------------------------------------------------------------
        # 3. Concept Mastery Updates (Weakness vs Strong Mastery)
        # (Skipped if current turn was resolving a misconception)
        # --------------------------------------------------------------
        if not is_resolved:
            # A) Check Weakness / Needs Support
            for pat, template in MASTERY_WEAK_PATTERNS:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    raw_topic = match.group(1).strip()
                    raw_topic = re.split(r"\b(can you|please|and|so|because)\b", raw_topic, flags=re.IGNORECASE)[0].strip()
                    val = f"Weakness / Needs support with: {raw_topic}"

                    # Check if this topic already exists in mastery
                    existing_item = None
                    topic_words = set(re.findall(r"\w+", raw_topic.lower()))
                    for item in blocks.get("mastery", []):
                        item_words = set(re.findall(r"\w+", item.get("value", "").lower()))
                        if topic_words & item_words:
                            existing_item = item
                            break

                    if existing_item:
                        actions.append(
                            MemoryAction(
                                action="UPDATE",
                                block="mastery",
                                target_id=existing_item["memory_id"],
                                value=val[:120],
                                confidence=0.90,
                                authority="student_authored",
                                reasoning=f"Student clarified weakness regarding {raw_topic}",
                                evidence=clean_text,
                            )
                        )
                    else:
                        actions.append(
                            MemoryAction(
                                action="CREATE",
                                block="mastery",
                                value=val[:120],
                                confidence=0.90,
                                authority="student_authored",
                                reasoning=f"Student disclosed conceptual hurdle on {raw_topic}",
                                evidence=clean_text,
                            )
                        )
                    break

            # B) Check Mastery Upgrade (Student learned or mastered a concept)
            for pat, template in MASTERY_STRONG_PATTERNS:
                match = re.search(pat, clean_text, re.IGNORECASE)
                if match:
                    raw_topic = match.group(1).strip() if match.groups() else match.group(0).strip()
                    raw_topic = re.split(r"\b(can you|please|and|so|but|however)\b", raw_topic, flags=re.IGNORECASE)[0].strip()
                    val = f"Mastered: {raw_topic}"

                    # Match against existing weak mastery to upgrade it
                    existing_item = None
                    topic_words = set(re.findall(r"\w+", raw_topic.lower()))
                    for item in blocks.get("mastery", []):
                        item_words = set(re.findall(r"\w+", item.get("value", "").lower()))
                        if topic_words & item_words:
                            existing_item = item
                            break

                    if existing_item:
                        actions.append(
                            MemoryAction(
                                action="UPDATE",
                                block="mastery",
                                target_id=existing_item["memory_id"],
                                value=val[:120],
                                confidence=0.95,
                                authority="student_authored",
                                reasoning=f"Student indicated progression from struggling to mastered on {raw_topic}",
                                evidence=clean_text,
                            )
                        )
                    else:
                        actions.append(
                            MemoryAction(
                                action="CREATE",
                                block="mastery",
                                value=val[:120],
                                confidence=0.95,
                                authority="student_authored",
                                reasoning=f"Student reported mastery of {raw_topic}",
                                evidence=clean_text,
                            )
                        )
                    break

        # --------------------------------------------------------------
        # 4. Learning Preferences: Addition & Deletion
        # --------------------------------------------------------------
        # A) Deletion / Cancellation of preference
        is_pref_del = False
        for del_pat in DELETE_PREFERENCE_PATTERNS:
            m_del = re.search(del_pat, clean_text, re.IGNORECASE)
            if m_del and blocks.get("preferences"):
                target_pref = m_del.group(1).strip().lower()
                for item in blocks.get("preferences", []):
                    if any(w in item.get("value", "").lower() for w in target_pref.split()):
                        actions.append(
                            MemoryAction(
                                action="DELETE",
                                block="preferences",
                                target_id=item["memory_id"],
                                value=item.get("value", ""),
                                confidence=0.95,
                                authority="student_authored",
                                reasoning=f"Student requested removal of preference for {target_pref}",
                                evidence=clean_text,
                            )
                        )
                        is_pref_del = True
                        break
            if is_pref_del:
                break

        # B) Addition of new pedagogical preference
        for pat, template in PREFERENCE_PATTERNS:
            match = re.search(pat, clean_text, re.IGNORECASE)
            if match:
                raw_pref = match.group(1).strip() if match.groups() else match.group(0).strip()
                val = template.format(raw_pref) if "{}" in template else template

                # Deduplicate with existing preferences
                existing = any(val.lower() in p.get("value", "").lower() for p in blocks.get("preferences", []))
                if not existing:
                    actions.append(
                        MemoryAction(
                            action="CREATE",
                            block="preferences",
                            value=val[:120],
                            confidence=0.95,
                            authority="student_authored",
                            reasoning="Student expressed pedagogical learning mode preference",
                            evidence=clean_text,
                        )
                    )
                break

        return actions

    def execute_memory_actions(
        self,
        actions: List[MemoryAction],
        learner_id: str,
    ) -> List[Dict[str, Any]]:
        """Commit planned memory actions against the SQLite LearnerMemoryController.

        Returns:
            List of operation execution summary dicts.
        """
        results: List[Dict[str, Any]] = []

        for act in actions:
            if act.action == "CREATE":
                mem_id = self.controller.core_memory_append(
                    learner_id=learner_id,
                    block=act.block,
                    value=act.value,
                    authority=act.authority,
                    confidence=act.confidence,
                    evidence=act.evidence,
                    provenance="dialogue_turn",
                )
                results.append({**act.to_dict(), "memory_id": mem_id, "status": "created"})

            elif act.action == "UPDATE":
                if act.target_id:
                    ok = self.controller.core_memory_replace(
                        memory_id=act.target_id,
                        new_value=act.value,
                        reason=act.reasoning,
                    )
                    results.append({**act.to_dict(), "memory_id": act.target_id, "status": "updated" if ok else "failed"})

            elif act.action == "DELETE":
                if act.target_id:
                    ok = self.controller.core_memory_delete(
                        memory_id=act.target_id,
                        reason=act.reasoning,
                    )
                    results.append({**act.to_dict(), "memory_id": act.target_id, "status": "deleted" if ok else "failed"})

            elif act.action == "RESOLVE":
                if act.target_id:
                    ok = self.controller.core_memory_resolve(
                        memory_id=act.target_id,
                        resolution_note=act.reasoning,
                    )
                    results.append({**act.to_dict(), "memory_id": act.target_id, "status": "resolved" if ok else "failed"})

        return results
