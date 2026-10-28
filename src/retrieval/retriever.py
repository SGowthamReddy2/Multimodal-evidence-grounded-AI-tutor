"""Hybrid Multi-Vector Retriever with RRF Fusion and Cross-Encoder Reranking.

Implements the multi-stage retrieval architecture:
1. Multi-vector prefetch (dense BGE + sparse BM25 + visual CLIP)
2. Qdrant server-side Reciprocal Rank Fusion (RRF, k=30)
3. Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) for Config A
"""

import logging
from typing import List, Optional

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
    ) -> List[RetrievedEvidence]:
        """Retrieve top-K evidence chunks matching the student query.

        Parameters
        ----------
        query:
            Student inquiry text.
        top_k:
            Number of final evidence items to return (default 5).
        config:
            'A' (Quality) uses prefetch=50 and cross-encoder reranking.
            'B' (Speed) uses prefetch=20 and skips reranking.
        include_visual:
            Whether to include cross-modal visual embedding in prefetch.
        """
        is_config_a = config.upper() == "A"
        prefetch_limit = PREFETCH_LIMIT_A if is_config_a else PREFETCH_LIMIT_B
        use_reranker = is_config_a

        # 1. Embed query
        dense_vec, sparse_vec, visual_vec = self.embedder.embed_query(
            query=query, include_visual=include_visual
        )

        # 2. Qdrant multi-vector prefetch & server-side RRF
        candidates = self.storage.search_multi_vector(
            dense_query=dense_vec,
            sparse_query=sparse_vec,
            visual_query=visual_vec,
            top_k=top_k,
            prefetch_limit=prefetch_limit,
            use_reranker=use_reranker,
        )

        if not candidates:
            logger.info("No candidates returned from Qdrant.")
            return []

        # 3. Rerank if Config A
        if use_reranker and len(candidates) > 1:
            pairs = []
            for c in candidates:
                # Combine transcript span or visual caption as passage
                passage = c.transcript_span or c.visual_caption or ""
                pairs.append((query, passage))

            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c.score = float(s)

            # Sort descending by cross-encoder score
            candidates.sort(key=lambda x: x.score, reverse=True)

        final_results = candidates[:top_k]
        logger.info(
            "Retrieved %d final chunks (config=%s, top_score=%.4f)",
            len(final_results),
            config,
            final_results[0].score if final_results else 0.0,
        )
        return final_results

