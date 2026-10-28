"""Grounded generation client using Groq API (Qwen 2.5 72B / Llama 3.3 70B).

Enforces:
1. Strict inline citation syntax: [claim_id | evidence_id]
2. Strict adherence to retrieved course evidence context
3. Configurable quality vs. speed profiles (Config A vs Config B)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq

from src.config import (
    GROQ_MAX_TOKENS_A,
    GROQ_MAX_TOKENS_B,
    GROQ_MODEL_QUALITY,
    GROQ_MODEL_SPEED,
    PROJECT_ROOT,
)

# Load environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """You are the Grounded Multimodal AI Tutor for MIT 6.006 (Introduction to Algorithms, Spring 2020).

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
    """Wraps the Groq Cloud API for frontier open-source LLM generation."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not set. LLM generation calls will fail until the key is provided."
            )
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    def set_api_key(self, api_key: str) -> None:
        """Dynamically update the Groq API key."""
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        logger.info("Groq API key updated successfully.")

    def generate_response(
        self,
        query: str,
        context_block: str,
        learner_profile: Optional[Dict] = None,
        config: str = "A",
    ) -> str:
        """Generate a grounded, cited pedagogical explanation.

        Parameters
        ----------
        query:
            Student inquiry.
        context_block:
            Formatted evidence block from Retriever.
        learner_profile:
            Optional active student memory profile attributes.
        config:
            'A' selects Qwen 2.5 72B (Quality), 'B' selects Llama 3.3 70B (Speed).
        """
        if not self.client:
            return (
                "⚠️ Groq API key is missing. Please set your GROQ_API_KEY in the environment or .env file to enable tutor generation."
            )

        model_name = GROQ_MODEL_QUALITY if config.upper() == "A" else GROQ_MODEL_SPEED
        max_tokens = GROQ_MAX_TOKENS_A if config.upper() == "A" else GROQ_MAX_TOKENS_B

        profile_str = ""
        if learner_profile:
            profile_lines = ["Active Learner Profile Context:"]
            for k, v in learner_profile.items():
                profile_lines.append(f"- {k.capitalize()}: {v}")
            profile_str = "\n".join(profile_lines) + "\n\n"

        user_content = f"{profile_str}Course Evidence Context:\n{context_block}\n\nStudent Question:\n{query}\n\nProvide your cited explanation:"

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,  # Low temperature for deterministic grounding
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.exception("Groq LLM generation error: %s", e)
            return f"⚠️ Generation error: {e}"

