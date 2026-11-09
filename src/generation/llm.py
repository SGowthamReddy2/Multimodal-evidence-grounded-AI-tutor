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
        config: str = "A",
    ) -> str:
        """Generate a grounded, cited pedagogical explanation."""
        model_name = PRIMARY_LLM_MODEL if config.upper() == "A" else SPEED_LLM_MODEL
        max_tokens = GROQ_MAX_TOKENS_A if config.upper() == "A" else GROQ_MAX_TOKENS_B

        profile_str = ""
        if learner_profile:
            profile_lines = ["Active Learner Profile Context:"]
            for k, v in learner_profile.items():
                profile_lines.append(f"- {k.capitalize()}: {v}")
            profile_str = "\n".join(profile_lines) + "\n\n"

        user_content = (
            f"{profile_str}Course Evidence Context:\n{context_block}\n\n"
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
