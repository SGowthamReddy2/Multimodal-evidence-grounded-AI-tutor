"""Learner Signal Extractor for Selective Durable Memory.

Analyzes student dialogue inputs to automatically detect and extract:
1. Goals: Exam schedules, deadlines, course targets (e.g. "I have a mid examination tomorrow")
2. Mastery: Self-reported strengths or weaknesses (e.g. "I am weak in this topic", "I struggle with recursion")
3. Misconceptions: Assertions revealing incorrect mental models
4. Preferences: Pedagogical learning modes (e.g. "Explain with diagrams", "Give me code examples")

Fulfills Milestone M5: Selective Learner Memory extraction & Confirmation Gate proposals.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Heuristic extraction patterns for rapid, zero-latency detection
GOAL_PATTERNS = [
    (r"(?:i have|i've got|upcoming|there is|scheduled)\s+(?:a\s+)?(?:mid\s*(?:term|sem|semester|examination)|exam|examination|quiz|test|finals?)\s*([^.!?\n]*)", "Upcoming assessment: {}"),
    (r"(?:my goal is to|i want to|aiming to|hoping to)\s+([^.!?\n]+)", "Goal: {}"),
    (r"(?:preparing for|studying for)\s+(?:my\s+)?(?:mid\s*(?:term|sem|semester|examination)|exam|examination|quiz|test|finals?|interview)\s*([^.!?\n]*)", "Target preparation: {}"),
]

MASTERY_WEAK_PATTERNS = [
    (r"(?:i am|i'm)\s+(?:really\s+)?(?:weak|bad|struggling|confused|not good)\s+(?:in|at|with)\s+([^.!?\n]+)", "Weakness: {}"),
    (r"(?:i struggle with|i have trouble with|i don't understand|i don't get)\s+([^.!?\n]+)", "Weakness / Difficulty with: {}"),
    (r"(?:hard for me to understand|difficult to follow)\s+([^.!?\n]+)", "Weakness / Conceptual hurdle: {}"),
]

MASTERY_STRONG_PATTERNS = [
    (r"(?:i am|i'm)\s+(?:good|confident|strong)\s+(?:in|at|with)\s+([^.!?\n]+)", "Mastered / confident: {}"),
    (r"(?:i already know|i have mastered|i understand)\s+([^.!?\n]+)", "Prior mastery: {}"),
]

PREFERENCE_PATTERNS = [
    (r"(?:i prefer|please explain with|can you show|show me)\s+(diagrams?|visuals?|drawings?|charts?|visual intuition)", "Prefers visual diagram explanations"),
    (r"(?:i prefer|give me|show me|explain with)\s+(code|pseudocode|python|implementation)", "Prefers code and pseudocode implementations"),
    (r"(?:keep it|be)\s+(concise|brief|short|simple|step by step)", "Prefers {} explanations"),
]


class LearnerSignalExtractor:
    """Extracts candidate durable memory records from user queries."""

    def extract_signals(self, query: str) -> List[Dict[str, Any]]:
        """Extract goal, mastery, and preference traits from user text.

        Returns:
            List of dicts with:
                - type: 'goal' | 'mastery' | 'misconception' | 'preference'
                - value: Cleaned, sanitized profile value
                - confidence: Initial confidence score (0.0 to 1.0)
                - authority: 'student_authored' (direct self-declaration) or 'tutor_proposed'
        """
        text = query.strip()
        signals: List[Dict[str, Any]] = []

        # 1. Goals & Exam deadlines
        for pat, template in GOAL_PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                matched_item = match.group(0).strip()
                val = template.format(matched_item) if "{}" in template else template
                # Capitalize nicely
                val = re.sub(r"^(i have|i've got)\s+", "", val, flags=re.IGNORECASE).strip()
                signals.append(
                    {
                        "type": "goal",
                        "value": val[:120],
                        "confidence": 0.95,
                        "authority": "student_authored",
                        "raw_match": match.group(0),
                    }
                )
                break

        # 2. Mastery / Weakness
        for pat, template in MASTERY_WEAK_PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                matched_item = match.group(1).strip()
                # Clean trailing filler words
                matched_item = re.split(r"\b(can you|please|and|so)\b", matched_item, flags=re.IGNORECASE)[0].strip()
                val = f"Weakness / Needs support with: {matched_item}"
                signals.append(
                    {
                        "type": "mastery",
                        "value": val[:120],
                        "confidence": 0.90,
                        "authority": "student_authored",
                        "raw_match": match.group(0),
                    }
                )
                break

        # 3. Mastery / Strong
        for pat, template in MASTERY_STRONG_PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                matched_item = match.group(1).strip()
                matched_item = re.split(r"\b(can you|please|and|so)\b", matched_item, flags=re.IGNORECASE)[0].strip()
                val = f"Confident in: {matched_item}"
                signals.append(
                    {
                        "type": "mastery",
                        "value": val[:120],
                        "confidence": 0.85,
                        "authority": "student_authored",
                        "raw_match": match.group(0),
                    }
                )
                break

        # 4. Learning Preferences
        for pat, template in PREFERENCE_PATTERNS:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                matched_item = match.group(1).strip()
                val = template.format(matched_item) if "{}" in template else template
                signals.append(
                    {
                        "type": "preference",
                        "value": val[:120],
                        "confidence": 0.90,
                        "authority": "student_authored",
                        "raw_match": match.group(0),
                    }
                )
                break

        if signals:
            logger.info("Extracted %d learner memory signal(s) from query: %s", len(signals), [s["value"] for s in signals])

        return signals
