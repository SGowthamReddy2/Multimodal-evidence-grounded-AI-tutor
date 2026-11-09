"""Vision-Language Model (VLM) Image Captioner for Lecture Slides and Keyframes.

Uses LiteLLMGateway to generate rich pedagogical descriptions of blackboard
derivations, recursion trees, algorithm diagrams, and slide code.
Includes a persistent disk cache to avoid redundant API calls across ingestion runs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.config import EVIDENCE_DIR, PRIMARY_VLM_MODEL
from src.generation.gateway import LiteLLMGateway

logger = logging.getLogger(__name__)

VLM_PEDAGOGICAL_PROMPT = """You are an expert university algorithms teaching assistant.
Analyze this lecture image (blackboard or slide).
In 2-3 concise, information-dense sentences:
1. Describe any algorithmic structures, recursion trees, data structures, graphs, tables, or code.
2. Transcribe any exact mathematical formulas, recurrence equations, asymptotic bounds (e.g. O(n log n)), and variable names shown.
3. State the core algorithmic concept being explained.
Do not describe the lecturer, physical room, or lighting. Focus purely on technical content."""


class VLMCaptioner:
    """Generates and caches pedagogical VLM captions for visual assets."""

    def __init__(
        self,
        gateway: Optional[LiteLLMGateway] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.gateway = gateway or LiteLLMGateway()
        self.cache_path = cache_path or (EVIDENCE_DIR / "vlm_captions_cache.json")
        self._cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Load persistent caption cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read VLM caption cache: %s", e)
        return {}

    def _save_cache(self) -> None:
        """Persist caption cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Could not save VLM caption cache: %s", e)

    def caption_image(
        self,
        image_path: Union[str, Path],
        content_hash: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate a pedagogical caption for an image, using cache if available."""
        p = Path(image_path)
        if not p.exists():
            return ""

        cache_key = content_hash or p.name
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.info("Calling VLM for image caption: %s", p.name)
        active_prompt = prompt or VLM_PEDAGOGICAL_PROMPT

        try:
            caption = self.gateway.caption_image(
                image_path=p,
                prompt=active_prompt,
                max_tokens=300,
                temperature=0.1,
            )
            if caption:
                self._cache[cache_key] = caption
                self._save_cache()
            return caption
        except Exception as e:
            logger.error("Failed to generate VLM caption for %s: %s", p.name, e)
            return ""

    def caption_batch(
        self,
        images_and_hashes: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """Process a list of {'path': ..., 'hash': ...} dicts and return mapping of path -> caption."""
        results = {}
        for item in images_and_hashes:
            path = item.get("path")
            chash = item.get("hash")
            if path:
                results[path] = self.caption_image(path, content_hash=chash)
        return results
