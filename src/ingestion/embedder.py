"""Multi-vector embedding pipeline using FastEmbed.

Generates:
1. Dense semantic embeddings via BAAI/bge-base-en-v1.5 (768d)
2. Sparse lexical embeddings via FastEmbed BM25 (Qdrant/bm25)
3. Visual image embeddings via CLIP ViT-B/32 (512d)
4. Cross-modal text query embeddings for visual vector retrieval (512d)
"""

import logging
from typing import List, Optional, Tuple, Union
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding, ImageEmbedding
from qdrant_client.models import SparseVector

from src.config import (
    DENSE_MODEL,
    SPARSE_MODEL,
    CLIP_MODEL,
    CLIP_TEXT_MODEL,
)

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Singleton wrapper around FastEmbed dense, sparse, and CLIP models."""

    def __init__(self) -> None:
        logger.info("Initializing MultimodalEmbedder with FastEmbed models...")
        self.dense_model = TextEmbedding(DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        self.clip_vision_model = ImageEmbedding(CLIP_MODEL)
        self.clip_text_model = TextEmbedding(CLIP_TEXT_MODEL)
        logger.info("All FastEmbed models successfully loaded.")

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generate 768-dimensional dense embeddings for a list of texts."""
        if not texts:
            return []
        embeddings = list(self.dense_model.embed(texts))
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_sparse(self, texts: List[str]) -> List[SparseVector]:
        """Generate native BM25 sparse vectors for a list of texts."""
        if not texts:
            return []
        sparse_results = list(self.sparse_model.embed(texts))
        vectors: List[SparseVector] = []
        for res in sparse_results:
            vectors.append(
                SparseVector(
                    indices=res.indices.tolist() if hasattr(res.indices, "tolist") else list(res.indices),
                    values=res.values.tolist() if hasattr(res.values, "tolist") else list(res.values),
                )
            )
        return vectors

    def embed_images(self, image_paths: List[Union[str, Path]]) -> List[List[float]]:
        """Generate 512-dimensional CLIP embeddings for images."""
        if not image_paths:
            return []
        str_paths = [str(p) for p in image_paths]
        embeddings = list(self.clip_vision_model.embed(str_paths))
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(
        self, query: str, include_visual: bool = True
    ) -> Tuple[List[float], SparseVector, Optional[List[float]]]:
        """Embed a single query into dense, sparse, and cross-modal visual vectors."""
        dense_vec = self.embed_dense([query])[0]
        sparse_vec = self.embed_sparse([query])[0]
        visual_vec = None
        if include_visual:
            visual_res = list(self.clip_text_model.embed([query]))[0]
            visual_vec = visual_res.tolist() if hasattr(visual_res, "tolist") else list(visual_res)

        return dense_vec, sparse_vec, visual_vec

