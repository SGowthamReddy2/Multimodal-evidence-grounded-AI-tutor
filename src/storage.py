"""Qdrant storage manager for the Multimodal AI Tutor evidence collection.

Manages an embedded Qdrant collection with named vectors (``dense``,
``sparse``, ``visual``) for hybrid multi-vector retrieval with server-side
RRF fusion.  All points carry rich evidence payloads produced by
:py:class:`src.models.MultimodalChunk`.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
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
    DEFAULT_COURSE_ID,
    DENSE_DIM,
    EVIDENCE_DIR,
    PROJECT_ROOT,
    QDRANT_PATH,
    RAW_DIR,
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
        self.is_replica = False
        try:
            self.client = QdrantClient(path=QDRANT_PATH)
            self._ensure_collection()
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                logger.warning(
                    "Qdrant storage folder is locked by another process (e.g. running UI). "
                    "Falling back to synchronized read replica 'qdrant_data_backup'."
                )
                import shutil
                replica_path = Path(PROJECT_ROOT) / "qdrant_data_backup"
                replica_path.mkdir(exist_ok=True)
                for item in Path(QDRANT_PATH).iterdir():
                    if item.name != ".lock":
                        dest = replica_path / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
                self.client = QdrantClient(path=str(replica_path))
                self.is_replica = True
                self._ensure_collection()
            else:
                raise

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
        if getattr(self, "is_replica", False):
            raise RuntimeError(
                "Cannot upsert chunks into read replica 'qdrant_data_backup'. "
                "The primary Qdrant storage ('qdrant_data') is currently locked by another process "
                "(e.g., a running Streamlit UI). Please stop the UI process before running ingestion."
            )

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
        filter_course_id: Optional[Union[str, List[str]]] = None,
        filter_lecture_id: Optional[Union[str, List[str]]] = None,
    ) -> List[RetrievedEvidence]:
        """Hybrid multi-vector search with server-side RRF fusion and course/lecture filtering."""
        must_conditions = [
            FieldCondition(
                key="status", match=MatchValue(value="active")
            )
        ]
        if filter_course_id:
            if isinstance(filter_course_id, str):
                c_list = [filter_course_id.strip()]
            else:
                c_list = [c.strip() for c in filter_course_id if c and c.strip()]
            # Filter out 'all courses' / 'all' wildcards
            c_list = [c for c in c_list if c.lower() not in ("all courses", "all")]
            if len(c_list) == 1:
                must_conditions.append(
                    FieldCondition(key="course_id", match=MatchValue(value=c_list[0]))
                )
            elif len(c_list) > 1:
                must_conditions.append(
                    FieldCondition(key="course_id", match=MatchAny(any=c_list))
                )

        if filter_lecture_id:
            if isinstance(filter_lecture_id, str):
                l_list = [filter_lecture_id.strip()]
            else:
                l_list = [l.strip() for l in filter_lecture_id if l and l.strip()]
            # Filter out 'all lectures' / 'all' wildcards
            l_list = [l for l in l_list if l.lower() not in ("all lectures", "all")]
            if len(l_list) == 1:
                must_conditions.append(
                    FieldCondition(key="lecture_id", match=MatchValue(value=l_list[0]))
                )
            elif len(l_list) > 1:
                must_conditions.append(
                    FieldCondition(key="lecture_id", match=MatchAny(any=l_list))
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
            alt_id = None
            if evidence_id.startswith("ev-6.006-"):
                alt_id = "ev-mit-" + evidence_id[3:]
            elif evidence_id.startswith("ev-mit-6.006-"):
                alt_id = "ev-" + evidence_id[7:]
            if alt_id:
                try:
                    results, _ = self.client.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="evidence_id",
                                    match=MatchValue(value=alt_id),
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=True,
                    )
                except Exception:
                    pass

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
        if getattr(self, "is_replica", False):
            raise RuntimeError(
                "Cannot invalidate points in read replica 'qdrant_data_backup'. "
                "The primary Qdrant storage is locked by another process."
            )

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
        """Return a mapping of course_id -> list of distinct lecture_ids in the collection and RAW_DIR."""
        mapping: Dict[str, set] = {}

        # 1. Discover from data/raw/ directory hierarchy
        try:
            if RAW_DIR.exists():
                for course_dir in RAW_DIR.iterdir():
                    if course_dir.is_dir() and not course_dir.name.startswith("."):
                        for lec_dir in course_dir.iterdir():
                            if lec_dir.is_dir() and not lec_dir.name.startswith("."):
                                mapping.setdefault(course_dir.name, set()).add(lec_dir.name)
        except Exception:
            pass

        # 2. Augment from Qdrant storage points if available
        try:
            records, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["course_id", "lecture_id"],
            )
            for r in records:
                p = r.payload or {}
                c = p.get("course_id", DEFAULT_COURSE_ID)
                l = p.get("lecture_id", "lecture-03")
                mapping.setdefault(c, set()).add(l)
        except Exception:
            pass

        if not mapping:
            return {DEFAULT_COURSE_ID: ["lecture-03", "lecture-04", "lecture-06", "lecture-07"]}

        return {c: sorted(list(lects)) for c, lects in mapping.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_asset_path(path_str: Optional[str]) -> Optional[str]:
        if not path_str:
            return None
        p = Path(path_str)
        if p.exists():
            return str(p)
        norm = path_str.replace("\\", "/")
        if "data/evidence/" in norm:
            rel = norm.split("data/evidence/", 1)[1]
            cand = EVIDENCE_DIR / rel
            if cand.exists():
                return str(cand)
        return path_str

    @classmethod
    def _point_to_evidence(cls, point) -> RetrievedEvidence:
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
            frame_asset_path=cls._resolve_asset_path(payload.get("frame_asset_path")),
            slide_asset_path=cls._resolve_asset_path(payload.get("slide_asset_path")),
            page_number=payload.get("page_number"),
            source_id=payload.get("source_id", ""),
            course_id=payload.get("course_id", DEFAULT_COURSE_ID),
            lecture_id=payload.get("lecture_id", "lecture-03"),
            status=payload.get("status", "active"),
            content_version=payload.get("content_version", ""),
        )
