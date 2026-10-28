"""PyMuPDF slide renderer and vector text extractor.

Renders crisp 150 DPI PNGs for multimodal inspection and extracts clean text
for dense and sparse retrieval.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import pymupdf

from src.config import ATTRIBUTION, EXTRACTION_METHOD, SLIDE_DPI, SLIDES_DIR
from src.models import AlignmentProvenance, EvidenceIdentity, MultimodalChunk

logger = logging.getLogger(__name__)


class PDFSlideProcessor:
    """Extracts slides and text from lecture PDF documents."""

    def __init__(self, dpi: int = SLIDE_DPI, output_dir: Path = SLIDES_DIR) -> None:
        self.dpi = dpi
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(
        self,
        pdf_path: str,
        source_id: str = "6.006-lec03-notes",
        source_url: str = "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/",
        prefix: str = "lec03",
    ) -> List[MultimodalChunk]:
        """Render slide pages to PNG and return MultimodalChunk evidence objects."""
        doc = pymupdf.open(str(pdf_path))
        num_pages = len(doc)
        logger.info("Processing PDF %s (%d pages)...", pdf_path, num_pages)

        chunks: List[MultimodalChunk] = []

        # 150 DPI matrix
        zoom = self.dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            # 1. Render PNG
            pix = page.get_pixmap(matrix=mat)
            slide_filename = f"{prefix}_slide_p{page_num}.png"
            slide_path = self.output_dir / slide_filename
            pix.save(str(slide_path))

            # 2. Extract Text
            text = page.get_text("text").strip()

            # 3. Hash
            with open(slide_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            evidence_id = f"ev-{source_id}-p{page_num}"

            identity = EvidenceIdentity(
                source_id=source_id,
                source_url=source_url,
                content_version="v1.0",
                content_hash=content_hash,
                evidence_id=evidence_id,
                modality="slide",
                status="active",
            )

            provenance = AlignmentProvenance(
                evidence_id=evidence_id,
                start_ms=0,
                end_ms=0,
                page_number=page_num,
                slide_asset_path=str(slide_path),
                transcript_span=text,
                visual_caption=f"Slide page {page_num} covering: {text[:200]}...",
            )

            chunk = MultimodalChunk(
                identity=identity,
                provenance=provenance,
                text_for_embedding=f"[Slide {page_num}]\n{text}",
                extraction_method=EXTRACTION_METHOD,
                attribution=ATTRIBUTION,
            )
            chunks.append(chunk)

        doc.close()
        logger.info("Successfully rendered %d slides to %s", len(chunks), self.output_dir)
        return chunks

