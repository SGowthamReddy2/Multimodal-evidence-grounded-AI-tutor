"""Qdrant storage manager for the Multimodal AI Tutor evidence collection.

Manages an embedded Qdrant collection with named vectors (``dense``,
``sparse``, ``visual``) for hybrid multi-vector retrieval with server-side
RRF fusion.  All points carry rich evidence payloads produced by
:py:class:`src.models.MultimodalChunk`.
"""

import logging
import uuid
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import (
    COLLECTION_NAME,
    DENSE_DIM,
    QDRANT_PATH,
    RERANK_TOP_N,
    VISUAL_DIM,
)
from src.models import MultimodalChunk, RetrievedEvidence

logger = logging.getLogger(__name__)

# Maximum number of points per upsert batch.
_UPSERT_BATCH_SIZE = 64


class QdrantStorageManager:
    """Embedded Qdrant manager for the ``course_evidence`` collection.

    On construction the client opens (or creates) a persistent storage
    directory at :pydata:`QDRANT_PATH` and ensures the collection schema
    exists with the correct named-vector configuration and payload indexes.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise the embedded Qdrant client and ensure the collection."""
        logger.info("Opening Qdrant embedded storage at %s", QDRANT_PATH)
        self.client = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()

    def close(self) -> None:
        """Release the embedded Qdrant client resources."""
        try:
            self.client.close()
            logger.info("Qdrant client closed.")
        except Exception:
            logger.exception("Error while closing Qdrant client")

    # ------------------------------------------------------------------
    # Collection bootstrap
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection and payload indexes if they do not exist."""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                logger.info(
                    "Collection '%s' already exists – skipping creation.",
                    COLLECTION_NAME,
                )
                return

            vectors_config = {
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
                "visual": VectorParams(size=VISUAL_DIM, distance=Distance.COSINE),
            }
            sparse_vectors_config = {
                "sparse": SparseVectorParams(),
            }

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
            logger.info("Created collection '%s'.", COLLECTION_NAME)

            # Payload indexes for fast filtered queries.
            for field_name in ("status", "modality", "source_id", "evidence_id", "course_id", "lecture_id"):
                try:
                    self.client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            logger.info("Payload indexes ensured for '%s'.", COLLECTION_NAME)

        except Exception:
            logger.exception(
                "Failed to ensure collection '%s'", COLLECTION_NAME
            )
            raise

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[MultimodalChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[SparseVector],
        visual_vectors: List[List[float]],
    ) -> None:
        """Upsert multimodal chunks with their pre-computed vectors.

        Parameters
        ----------
        chunks:
            Evidence chunks whose ``to_payload()`` output becomes the point
            payload.
        dense_vectors:
            Dense embedding vectors (one per chunk, length ``DENSE_DIM``).
        sparse_vectors:
            Sparse BM25 vectors (one per chunk).  Each element must expose
            ``.indices`` and ``.values`` attributes.
        visual_vectors:
            CLIP visual embedding vectors (one per chunk, length
            ``VISUAL_DIM``).
        """
        points: List[PointStruct] = []
        for chunk, dense_vec, sparse_vec, visual_vec in zip(
            chunks, dense_vectors, sparse_vectors, visual_vectors
        ):
            point_id = uuid.uuid5(
                uuid.NAMESPACE_DNS, chunk.identity.evidence_id
            )
            point = PointStruct(
                id=str(point_id),
                vector={
                    "dense": dense_vec,
                    "visual": visual_vec,
                    "sparse": SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                },
                payload=chunk.to_payload(),
            )
            points.append(point)

        # Batch upsert.
        total_upserted = 0
        for i in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[i : i + _UPSERT_BATCH_SIZE]
            try:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch,
                )
                total_upserted += len(batch)
            except Exception:
                logger.exception(
                    "Upsert failed for batch starting at index %d", i
                )
                raise

        logger.info(
            "Upserted %d points into '%s'.", total_upserted, COLLECTION_NAME
        )

    # ------------------------------------------------------------------
    # Read path – multi-vector hybrid search
    # ------------------------------------------------------------------

    def search_multi_vector(
        self,
        dense_query: List[float],
        sparse_query: SparseVector,
        visual_query: Optional[List[float]] = None,
        top_k: int = 5,
        prefetch_limit: int = 50,
        use_reranker: bool = True,
        filter_course_id: Optional[str] = None,
        filter_lecture_id: Optional[str] = None,
    ) -> List[RetrievedEvidence]:
        """Hybrid multi-vector search with server-side RRF fusion and course/lecture filtering."""
        must_conditions = [
            FieldCondition(
                key="status", match=MatchValue(value="active")
            )
        ]
        if filter_course_id:
            must_conditions.append(
                FieldCondition(key="course_id", match=MatchValue(value=filter_course_id))
            )
        if filter_lecture_id:
            must_conditions.append(
                FieldCondition(key="lecture_id", match=MatchValue(value=filter_lecture_id))
            )

        active_filter = Filter(must=must_conditions)

        prefetches: List[Prefetch] = [
            Prefetch(
                query=dense_query,
                using="dense",
                limit=prefetch_limit,
                filter=active_filter,
            ),
            Prefetch(
                query=sparse_query,
                using="sparse",
                limit=prefetch_limit,
                filter=active_filter,
            ),
        ]
        if visual_query is not None:
            prefetches.append(
                Prefetch(
                    query=visual_query,
                    using="visual",
                    limit=prefetch_limit,
                    filter=active_filter,
                )
            )

        limit = RERANK_TOP_N if use_reranker else top_k

        try:
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=prefetches,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        except Exception:
            logger.exception("Multi-vector search failed")
            return []

        return [
            self._point_to_evidence(point)
            for point in results.points
        ]

    # ------------------------------------------------------------------
    # Point-level reads
    # ------------------------------------------------------------------

    def get_evidence_by_id(
        self, evidence_id: str
    ) -> Optional[RetrievedEvidence]:
        """Retrieve a single evidence chunk by its ``evidence_id`` payload.

        Returns ``None`` if no matching point is found.
        """
        try:
            results, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="evidence_id",
                            match=MatchValue(value=evidence_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
        except Exception:
            logger.exception(
                "Failed to retrieve evidence_id='%s'", evidence_id
            )
            return None

        if not results:
            logger.debug("No point found for evidence_id='%s'", evidence_id)
            return None

        return self._point_to_evidence(results[0])

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_source_version(self, source_id: str) -> None:
        """Mark all points for *source_id* as ``superseded``.

        Uses a server-side ``set_payload`` update so that no vectors need to
        be re-uploaded.
        """
        source_filter = Filter(
            must=[
                FieldCondition(
                    key="source_id", match=MatchValue(value=source_id)
                )
            ]
        )

        try:
            result = self.client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"status": "superseded"},
                points=source_filter,
            )
            logger.info(
                "Invalidated points for source_id='%s' (result=%s).",
                source_id,
                result,
            )
        except Exception:
            logger.exception(
                "Failed to invalidate source_id='%s'", source_id
            )
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_collection_info(self) -> Dict:
        """Return basic collection diagnostics (point count & status)."""
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            return {
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            logger.exception("Failed to get collection info")
            return {"points_count": 0, "status": "unknown"}

    def list_available_courses_and_lectures(self) -> Dict[str, List[str]]:
        """Return a mapping of course_id -> list of distinct lecture_ids in the collection."""
        try:
            records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["course_id", "lecture_id"],
            )
            mapping: Dict[str, set] = {}
            for r in records:
                p = r.payload or {}
                c = p.get("course_id", "MIT_6.006")
                l = p.get("lecture_id", "lecture-03")
                if c not in mapping:
                    mapping[c] = set()
                mapping[c].add(l)
            if not mapping:
                return {"MIT_6.006": ["lecture-03"]}
            return {c: sorted(list(lects)) for c, lects in mapping.items()}
        except Exception:
            return {"MIT_6.006": ["lecture-03"]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_to_evidence(point) -> RetrievedEvidence:
        """Convert a Qdrant scored/record point to a ``RetrievedEvidence``."""
        payload = point.payload or {}
        score = getattr(point, "score", 0.0) or 0.0
        return RetrievedEvidence(
            evidence_id=payload.get("evidence_id", ""),
            score=float(score),
            modality=payload.get("modality", ""),
            transcript_span=payload.get("transcript_span", ""),
            visual_caption=payload.get("visual_caption", ""),
            start_ms=payload.get("start_ms", 0),
            end_ms=payload.get("end_ms", 0),
            frame_asset_path=payload.get("frame_asset_path"),
            slide_asset_path=payload.get("slide_asset_path"),
            page_number=payload.get("page_number"),
            source_id=payload.get("source_id", ""),
            course_id=payload.get("course_id", "MIT_6.006"),
            lecture_id=payload.get("lecture_id", "lecture-03"),
            status=payload.get("status", "active"),
            content_version=payload.get("content_version", ""),
        )
