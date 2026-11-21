"""Hybrid Multi-Vector Retriever with RRF Fusion and Cross-Encoder Reranking.

Implements the multi-stage retrieval architecture:
1. Multi-vector prefetch (dense BGE + sparse BM25 + visual CLIP)
2. Qdrant server-side Reciprocal Rank Fusion (RRF, k=30)
3. Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) for Config A
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import logging
from typing import List, Optional, Union

from sentence_transformers import CrossEncoder

from src.config import (
    FINAL_TOP_K,
    PREFETCH_LIMIT_A,
    PREFETCH_LIMIT_B,
    RERANK_TOP_N,
    RERANKER_MODEL,
)
from src.ingestion.embedder import MultimodalEmbedder
from src.models import RetrievedEvidence
from src.storage import QdrantStorageManager

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Multi-channel retriever combining dense, sparse, and visual search."""

    def __init__(
        self,
        storage: Optional[QdrantStorageManager] = None,
        embedder: Optional[MultimodalEmbedder] = None,
    ) -> None:
        self.storage = storage or QdrantStorageManager()
        self.embedder = embedder or MultimodalEmbedder()
        logger.info("Loading Cross-Encoder reranker: %s", RERANKER_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Cross-Encoder reranker ready.")

    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        config: str = "A",  # 'A' = Quality-Optimized, 'B' = Speed-Optimized
        include_visual: bool = True,
        course_id: Optional[Union[str, List[str]]] = None,
        lecture_id: Optional[Union[str, List[str]]] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve top-K evidence chunks matching the student query with optional course/lecture scoping."""
        is_config_a = config.upper() == "A"
        prefetch_limit = PREFETCH_LIMIT_A if is_config_a else PREFETCH_LIMIT_B
        use_reranker = is_config_a

        # 1. Embed query
        dense_vec, sparse_vec, visual_vec = self.embedder.embed_query(
            query=query, include_visual=include_visual
        )

        # 2. Qdrant multi-vector prefetch & server-side RRF with optional course/lecture scope
        candidates = self.storage.search_multi_vector(
            dense_query=dense_vec,
            sparse_query=sparse_vec,
            visual_query=visual_vec,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            use_reranker=use_reranker,
            filter_course_id=course_id,
            filter_lecture_id=lecture_id,
        )

        if not candidates:
            logger.info("No candidates returned from Qdrant.")
            return []

        # Store initial multi-vector RRF scores for hybrid calibration
        for c in candidates:
            setattr(c, "rrf_score", float(c.score))

        # 3. Rerank if Config A (Quality-focused Cross-Encoder pipeline)
        if use_reranker and len(candidates) > 1:
            pairs = []
            for c in candidates:
                # Build rich multimodal passage incorporating slide page, transcript, and visual diagram captions
                passage_parts = []
                if c.page_number:
                    passage_parts.append(f"Slide/Page {c.page_number}")
                if c.transcript_span:
                    passage_parts.append(c.transcript_span.strip())
                if c.visual_caption and c.visual_caption.strip() != (c.transcript_span or "").strip():
                    passage_parts.append(f"Visual Diagram: {c.visual_caption.strip()}")
                passage = " | ".join(passage_parts) if passage_parts else (c.transcript_span or c.visual_caption or "")
                pairs.append((query, passage))

            scores = self.reranker.predict(pairs)
            import math
            q_lower = query.lower()
            is_visual_q = any(w in q_lower for w in ["slide", "diagram", "table", "chalkboard", "drawn", "visual", "figure", "board"])

            for c, s in zip(candidates, scores):
                raw_s = float(s)
                # Calibrate ms-marco cross-encoder logit into [0.0, 1.0] probability matching epistemic thresholds:
                # - Highly relevant (s >= 0.0): score in [0.69, 0.99] -> State: FULL
                # - Borderline / hedged (-1.0 <= s < 0.0): score in [0.45, 0.69) -> State: QUALIFIED
                # - Irrelevant / off-corpus (s < -1.0): score < 0.45 -> State: ABSTAIN
                try:
                    calibrated_score = float(1.0 / (1.0 + math.exp(-(raw_s + 0.8))))
                except OverflowError:
                    calibrated_score = 1.0 if raw_s > 0 else 0.0

                # Multimodal visual query boost (+0.08) for slide/diagram evidence on visual queries
                if is_visual_q and c.modality in ["slide", "diagram", "keyframe"]:
                    calibrated_score = min(1.0, calibrated_score + 0.08)

                c.score = float(calibrated_score)

            # Sort descending by quality-calibrated score
            candidates.sort(key=lambda x: x.score, reverse=True)

        final_results = candidates[:top_k]
        logger.info(
            "Retrieved %d final chunks (config=%s, top_score=%.4f)",
            len(final_results),
            config,
            final_results[0].score if final_results else 0.0,
        )
        return final_results

