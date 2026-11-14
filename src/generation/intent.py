"""Query Intent Classifier for Humane Dialogue & Epistemic Boundaries.

Classifies incoming student queries into:
1. GREETING_CHITCHAT: Conversational pleasantries, introductions, and scope questions ("Hi", "Hello", "Who are you?").
   -> Bypasses RAG retrieval and strict abstention; responds with warm, humane orientation.
2. LEARNER_PROFILE_UPDATE: Statements of exam schedules, weaknesses, or goals without a specific technical query.
   -> Updates durable learner memory, provides supportive coaching, without triggering cold RAG abstention.
3. ACADEMIC_RAG: Course-specific algorithmic and technical questions.
   -> Routes through hybrid retrieval, reranking, and calibrated tri-state epistemic abstention.
4. OFF_CORPUS: Irrelevant general knowledge or non-CS queries.
   -> Evaluated against retrieval thresholds to trigger standard epistemic abstention.
"""

import logging
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GREETING_REGEXES = [
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening|day))\b",
    r"^(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|help|introduce\s+yourself)\b",
    r"^(thanks|thank\s+you|great|awesome|bye|goodbye|see\s+you)\b",
]

# Course-specific keywords indicating genuine algorithm questions
ACADEMIC_KEYWORDS = [
    "sort", "sorting", "merge", "insertion", "heap", "quick", "recursion",
    "recurrence", "tree", "divide", "conquer", "algorithm", "complexity",
    "asymptotic", "theta", "big-o", "omega", "runtime", "in-place",
    "stable", "comparison", "binary", "search", "linear", "array", "pointer",
    "lecture", "slide", "blackboard", "notes", "proof", "derivation", "master"
]


class QueryIntent(str, Enum):
    GREETING_CHITCHAT = "greeting_chitchat"
    LEARNER_PROFILE_UPDATE = "learner_profile_update"
    ACADEMIC_RAG = "academic_rag"
    ACADEMIC_WITH_MEMORY = "academic_with_memory"
    OFF_CORPUS = "off_corpus"


class QueryIntentClassifier:
    """Fast, deterministic intent classifier for incoming student queries."""

    def classify(self, query: str, has_learner_signals: bool = False) -> QueryIntent:
        """Determine the routing intent for a student query.

        Args:
            query: Raw query string from user.
            has_learner_signals: True if LearnerSignalExtractor or LearnerMemoryEngine detected goals/mastery.

        Returns:
            QueryIntent enum.
        """
        clean = query.strip().lower()

        # 1. Check for pure greetings or chit-chat
        for pattern in GREETING_REGEXES:
            if re.search(pattern, clean):
                # If there are also technical terms, treat as academic
                has_academic_term = any(kw in clean for kw in ACADEMIC_KEYWORDS)
                if not has_academic_term:
                    logger.info("Classified query '%s' as GREETING_CHITCHAT", query)
                    return QueryIntent.GREETING_CHITCHAT

        # 2. Check if the query has learner signals (exam deadline, weakness, preference)
        if has_learner_signals:
            has_academic_term = any(kw in clean for kw in ACADEMIC_KEYWORDS)
            explicit_interrogative = any(
                re.search(rf"\b{starter}\b", clean)
                for starter in [
                    "how do", "how does", "how can", "how to", "what is", "what are",
                    "why does", "why is", "explain", "derive", "prove", "calculate",
                    "solve", "walk me through", "can you show", "can you explain"
                ]
            )
            # If student asked an explicit technical algorithm question while sharing personal study state:
            if has_academic_term and explicit_interrogative:
                logger.info("Classified query '%s' as ACADEMIC_WITH_MEMORY (Compound query)", query)
                return QueryIntent.ACADEMIC_WITH_MEMORY

            # If no explicit technical question is being asked, prioritize the learner profile update
            logger.info("Classified query '%s' as LEARNER_PROFILE_UPDATE", query)
            return QueryIntent.LEARNER_PROFILE_UPDATE

        # 3. Default to ACADEMIC_RAG for full evidence retrieval and tri-state abstention
        return QueryIntent.ACADEMIC_RAG
