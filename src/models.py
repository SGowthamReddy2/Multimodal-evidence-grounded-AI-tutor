"""Core data contracts for Multimodal AI Tutor evidence identity, alignment, and retrieval.

Defines the immutable data structures used across ingestion, storage, retrieval,
and generation layers.  Every evidence chunk flowing through the pipeline is
represented as a ``MultimodalChunk`` containing its ``EvidenceIdentity`` and
``AlignmentProvenance``.
"""

from dataclasses import dataclass, asdict, field
from typing import Literal, Optional, Dict, Any, List


@dataclass
class EvidenceIdentity:
    """Immutable evidence identity and cryptographic state."""

    source_id: str           # Canonical source identifier (e.g., '6.006-lec03-video')
    source_url: str          # Canonical public URL or source path
    content_version: str     # Version string (e.g., 'v1.0')
    content_hash: str        # SHA-256 cryptographic digest of raw text/image bytes
    evidence_id: str         # Globally unique identifier (e.g., 'ev-lec03-t-142000_165000')
    modality: Literal["transcript", "slide", "keyframe", "reading", "code"]
    status: Literal["active", "superseded"] = "active"


@dataclass
class AlignmentProvenance:
    """Temporal and spatial alignment linking evidence to physical instructional moments."""

    evidence_id: str
    start_ms: int                              # Start millisecond offset in video/audio
    end_ms: int                                # End millisecond offset in video/audio
    frame_timestamp_ms: Optional[int] = None   # Aligned keyframe timestamp in ms
    frame_asset_path: Optional[str] = None     # Local path to extracted keyframe PNG
    slide_asset_path: Optional[str] = None     # Local path to extracted slide PNG
    page_number: Optional[int] = None          # 1-indexed slide/page number
    crop_coordinates: Optional[Dict[str, int]] = None
    transcript_span: str = ""                  # Verbatim transcript text covering this span
    visual_caption: str = ""                   # VLM-generated caption for visual assets


@dataclass
class MultimodalChunk:
    """Unified multimodal evidence chunk ready for embedding and indexing."""

    identity: EvidenceIdentity
    provenance: AlignmentProvenance
    text_for_embedding: str = ""               # Concatenated text used for dense/sparse embedding
    extraction_method: str = "whisperx+learnet_tib"
    extraction_version: str = "v1.0"
    attribution: str = "MIT OCW CC BY-NC-SA 4.0 - Prof. Erik Demaine"

    def to_payload(self) -> Dict[str, Any]:
        """Flatten into a Qdrant-compatible payload dictionary."""
        return {
            "source_id": self.identity.source_id,
            "source_url": self.identity.source_url,
            "content_version": self.identity.content_version,
            "content_hash": self.identity.content_hash,
            "evidence_id": self.identity.evidence_id,
            "modality": self.identity.modality,
            "status": self.identity.status,
            "start_ms": self.provenance.start_ms,
            "end_ms": self.provenance.end_ms,
            "frame_timestamp_ms": self.provenance.frame_timestamp_ms,
            "frame_asset_path": self.provenance.frame_asset_path,
            "slide_asset_path": self.provenance.slide_asset_path,
            "page_number": self.provenance.page_number,
            "crop_coordinates": self.provenance.crop_coordinates,
            "transcript_span": self.provenance.transcript_span,
            "visual_caption": self.provenance.visual_caption,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "attribution": self.attribution,
        }


@dataclass
class RetrievedEvidence:
    """A retrieved evidence chunk with its relevance score for context assembly."""

    evidence_id: str
    score: float
    modality: str
    transcript_span: str = ""
    visual_caption: str = ""
    start_ms: int = 0
    end_ms: int = 0
    frame_asset_path: Optional[str] = None
    slide_asset_path: Optional[str] = None
    page_number: Optional[int] = None
    source_id: str = ""
    status: str = "active"
    content_version: str = ""

