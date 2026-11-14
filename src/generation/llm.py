"""Grounded generation client using LiteLLMGateway.

Enforces:
1. Strict inline citation syntax: [claim_id | evidence_id]
2. Strict adherence to retrieved course evidence context
3. Configurable quality vs. speed profiles (Config A vs Config B)
4. Dynamic rate-limit failover with automatic reset tracking
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.config import (
    GROQ_MAX_TOKENS_A,
    GROQ_MAX_TOKENS_B,
    GROQ_MAX_TOKENS_CONVERSATIONAL,
    PRIMARY_LLM_MODEL,
    PROJECT_ROOT,
    SPEED_LLM_MODEL,
)
from src.generation.gateway import LiteLLMGateway

# Load environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """You are the Grounded Multimodal AI Tutor for university computer science courses (including MIT 6.006).

CRITICAL PEDAGOGICAL & GROUNDING RULES:
1. You must answer the student's question accurately using ONLY the provided Course Evidence context.
2. CITATION CONTRACT: Every factual claim, algorithm property, formula, recurrence, or runtime you state MUST be immediately followed by an inline citation token in the format:
   [claim_id | evidence_id]
   Where:
   - claim_id is a short identifier (e.g. c1, c2, c3)
   - evidence_id MUST EXACTLY MATCH one of the Evidence IDs provided in the context (e.g. ev-6.006-lec03-video-t-872000_901000 or ev-6.006-lec03-notes-p2).
3. If the provided evidence does not contain sufficient information to answer the question with certainty, you MUST state:
   "I cannot verify this from the approved course material."
   Do NOT guess, hallucinate, or bring in external knowledge not present in the evidence.
4. If an active learner profile is provided (goals, misconceptions, mastery), adapt your pedagogical style accordingly (e.g. use visual intuition if preferred, address known misconceptions gently).
5. Maintain a clear, encouraging, university-level instructional tone.
6. Address the student directly as "you". Never invent or assume a name for the student (do NOT call or greet the student as "Letta" or any assumed name).
7. PEDAGOGICAL COMPLETENESS & DEPTH CONTRACT:
   Ensure you explain the concept completely, rigorously, and self-containedly within your response:
   - Do NOT give surface-level, clipped, or prematurely truncated summaries.
   - When explaining algorithms or mathematical runtimes:
     * Explain the core intuition and mechanism step-by-step (e.g. divide, conquer, combine).
     * State the formal recurrence relation (e.g. T(n) = 2T(n/2) + O(n)) and walk through the tree/derivation levels.
     * Clearly state the base cases and final asymptotic bounds (e.g. Theta(n log n)).
     * Integrate blackboard and slide visual citations seamlessly into the narrative.
     * Conclude with a clear concluding takeaway tying the concept together so the student has a complete, fully resolved explanation from start to finish.
"""

TUTOR_HUMANE_CONVERSATIONAL_PROMPT = """You are the Grounded Multimodal AI Tutor for MIT 6.006 (Introduction to Algorithms) and university computer science courses.
You are communicating directly with a university student.

PEDAGOGICAL & CONVERSATIONAL GUIDELINES:
1. Be warm, humane, encouraging, and academically supportive.
2. For greetings or introductions:
   - Introduce yourself warmly as their Multimodal AI Tutor.
   - Explain what you can help with: lecture video moments, blackboard derivations, slide diagrams, recursion trees, and algorithm proofs.
   - Mention currently indexed course material (MIT 6.006: Sorting, Insertion Sort, Merge Sort, Recurrences).
3. For student personal learning disclosures (e.g. exams tomorrow, feeling weak or anxious about a topic):
   - Acknowledge their situation empathetically.
   - Reassure them that you have updated their learner profile to tailor your teaching to their goals and priorities.
   - Recommend key foundational topics to focus on (e.g. Lecture 3: Merge Sort recursion tree, divide-and-conquer master method).
4. Thoroughness & Completeness:
   - When explaining concepts or answering questions during conversational or profile-update turns, provide a thorough, complete explanation without cutting off prematurely.
   - Ensure all thoughts, advice, and pedagogical recommendations are concluded clearly with an actionable next study step.
5. Address the student directly as "you". Never invent or assume a name for the student (do NOT call or greet the student as "Letta" or any assumed name).
"""


class GroundedGenerator:
    """Wraps LiteLLMGateway for grounded pedagogical explanation generation."""

    def __init__(
        self,
        gateway: Optional[LiteLLMGateway] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        self.gateway = gateway or LiteLLMGateway()

    def set_api_key(self, api_key: str) -> None:
        """Update API key in the environment for LiteLLM."""
        os.environ["GROQ_API_KEY"] = api_key
        logger.info("API key updated for generation.")

    def generate_response(
        self,
        query: str,
        context_block: str,
        learner_profile: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        config: str = "A",
    ) -> str:
        """Generate a grounded, cited pedagogical explanation with multi-turn context."""
        model_name = PRIMARY_LLM_MODEL if config.upper() == "A" else SPEED_LLM_MODEL
        max_tokens = GROQ_MAX_TOKENS_A if config.upper() == "A" else GROQ_MAX_TOKENS_B

        profile_str = ""
        if learner_profile:
            profile_lines = ["Active Learner Profile Context (Active Working Memory):"]
            for k, v in learner_profile.items():
                if not v:
                    continue
                if isinstance(v, list):
                    profile_lines.append(f"- {k.capitalize()}:")
                    for item in v:
                        profile_lines.append(f"  * {item}")
                elif k not in ("goals", "masteries", "misconceptions", "preferences"):
                    profile_lines.append(f"- {k.capitalize()}: {v}")
            profile_str = "\n".join(profile_lines) + "\n\n"

        history_str = ""
        if conversation_history:
            h_lines = ["Recent Dialogue Context:"]
            for turn in conversation_history[-6:]:
                role = "Student" if turn.get("role") == "user" else "Tutor"
                c = turn.get("content", "").strip()
                if c:
                    h_lines.append(f"{role}: {c}")
            if len(h_lines) > 1:
                history_str = "\n".join(h_lines) + "\n\n"

        user_content = (
            f"{profile_str}{history_str}Course Evidence Context:\n{context_block}\n\n"
            f"Student Question:\n{query}\n\nProvide your cited explanation:"
        )

        try:
            return self.gateway.complete(
                prompt=user_content,
                system_prompt=TUTOR_SYSTEM_PROMPT,
                model=model_name,
                max_tokens=max_tokens,
                temperature=0.1,
            )
        except Exception as e:
            logger.exception("LLM generation error: %s", e)
            return f"⚠️ Generation error: {e}"

    def generate_conversational_response(
        self,
        query: str,
        learner_profile: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
        context_note: Optional[str] = None,
        config: str = "A",
    ) -> str:
        """Generate a warm, humane conversational reply for greetings or learner state updates."""
        model_name = SPEED_LLM_MODEL if config.upper() == "B" else PRIMARY_LLM_MODEL
        max_tokens = GROQ_MAX_TOKENS_CONVERSATIONAL

        profile_str = ""
        if learner_profile:
            profile_lines = ["Active Learner Profile (Active Working Memory):"]
            for k, v in learner_profile.items():
                if not v:
                    continue
                if isinstance(v, list):
                    profile_lines.append(f"- {k.capitalize()}:")
                    for item in v:
                        profile_lines.append(f"  * {item}")
                elif k not in ("goals", "masteries", "misconceptions", "preferences"):
                    profile_lines.append(f"- {k.capitalize()}: {v}")
            profile_str = "\n".join(profile_lines) + "\n\n"

        history_str = ""
        if conversation_history:
            h_lines = ["Recent Dialogue:"]
            for turn in conversation_history[-4:]:
                role = "Student" if turn.get("role") == "user" else "Tutor"
                c = turn.get("content", "").strip()
                if c:
                    h_lines.append(f"{role}: {c}")
            if len(h_lines) > 1:
                history_str = "\n".join(h_lines) + "\n\n"

        extra_note = f"Guidance note: {context_note}\n\n" if context_note else ""

        user_content = (
            f"{profile_str}{history_str}{extra_note}"
            f"Student Message:\n{query}\n\n"
            "Provide your supportive response:"
        )

        try:
            return self.gateway.complete(
                prompt=user_content,
                system_prompt=TUTOR_HUMANE_CONVERSATIONAL_PROMPT,
                model=model_name,
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except Exception as e:
            logger.exception("Conversational LLM generation error: %s", e)
            return f"Hello! I am your Multimodal AI Tutor for algorithms. How can I help you today? (Note: {e})"

