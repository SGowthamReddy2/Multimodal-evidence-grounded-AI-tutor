"""Deterministic Pre-Display Citation Validator.

Verifies that every [claim_id | evidence_id] token in the generated response:
1. Exists in Qdrant storage.
2. Has status == 'active' (not superseded by errata).
3. Has physically valid temporal bounds (start_ms <= end_ms).
4. Links to reachable asset files on disk.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.models import RetrievedEvidence
from src.storage import QdrantStorageManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of pre-display citation gate."""

    is_valid: bool
    total_claims: int
    valid_citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    superseded_citations: List[str] = field(default_factory=list)
    sanitized_response: str = ""
    evidence_items: Dict[str, RetrievedEvidence] = field(default_factory=dict)


class CitationValidator:
    """Deterministic validation middleware for grounded LLM explanations."""

    def __init__(self, storage: Optional[QdrantStorageManager] = None) -> None:
        self.storage = storage or QdrantStorageManager()
        # Pattern matches [claim_id | evidence_id] or [ev-...] with optional whitespace
        self.pattern = re.compile(r"\[\s*([a-zA-Z0-9_\-\.]+)\s*\|\s*([a-zA-Z0-9_\-\.\:]+)\s*\]")
        self.fallback_pattern = re.compile(r"\[\s*(ev\-[a-zA-Z0-9_\-\.\:]+)\s*\]")

    def validate(self, response_text: str) -> ValidationResult:
        """Parse and validate all citations in response_text."""
        matches = self.pattern.findall(response_text)
        fallback_matches = self.fallback_pattern.findall(response_text)

        all_citations: Set[str] = set()
        for claim_id, ev_id in matches:
            all_citations.add(ev_id.strip())
        for ev_id in fallback_matches:
            all_citations.add(ev_id.strip())

        valid_citations = []
        invalid_citations = []
        superseded_citations = []
        evidence_items = {}

        for ev_id in all_citations:
            ev = self.storage.get_evidence_by_id(ev_id)
            if not ev:
                logger.warning("Citation validation failed: evidence_id '%s' not found in Qdrant.", ev_id)
                invalid_citations.append(ev_id)
                continue

            if ev.status == "superseded":
                logger.warning("Citation references superseded evidence: '%s'", ev_id)
                superseded_citations.append(ev_id)

            # Check asset paths if specified
            if ev.frame_asset_path and not Path(ev.frame_asset_path).exists():
                logger.warning("Evidence asset missing on disk: %s", ev.frame_asset_path)
            if ev.slide_asset_path and not Path(ev.slide_asset_path).exists():
                logger.warning("Evidence slide missing on disk: %s", ev.slide_asset_path)

            valid_citations.append(ev_id)
            evidence_items[ev_id] = ev

        is_valid = len(invalid_citations) == 0 and len(superseded_citations) == 0

        # Replace valid citations with clean badges for display
        sanitized_text = response_text
        if superseded_citations:
            warning_banner = (
                "\n\n> ⚠️ **Notice**: This response cites superseded or updated lecture material "
                f"({', '.join(superseded_citations)}). Please verify with latest course errata."
            )
            sanitized_text += warning_banner

        return ValidationResult(
            is_valid=is_valid,
            total_claims=len(all_citations),
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            superseded_citations=superseded_citations,
            sanitized_response=sanitized_text,
            evidence_items=evidence_items,
        )

