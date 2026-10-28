"""End-to-end multimodal ingestion pipeline orchestrator.

Coordinates:
1. Video subtitle extraction & chunking
2. LEARNet TIB keyframe extraction
3. Slide PDF rendering & text extraction
4. Multimodal embedding (Dense BGE-v1.5 + Sparse BM25 + Visual CLIP)
5. Atomic Qdrant upsert with version invalidation check
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import FRAMES_DIR, RAW_DIR, SLIDES_DIR
from src.ingestion.embedder import MultimodalEmbedder
from src.ingestion.pdf_processor import PDFSlideProcessor
from src.ingestion.video_processor import (
    LEARNetTIBExtractor,
    build_video_multimodal_chunks,
    chunk_transcript_cues,
    extract_subtitles_from_video,
    parse_vtt_subtitles,
)
from src.models import MultimodalChunk
from src.storage import QdrantStorageManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates ingestion of raw lecture assets into Qdrant."""

    def __init__(self, storage: Optional[QdrantStorageManager] = None) -> None:
        self.storage = storage or QdrantStorageManager()
        self.embedder = MultimodalEmbedder()
        self.video_extractor = LEARNetTIBExtractor()
        self.pdf_processor = PDFSlideProcessor()

    def ingest_lecture(
        self,
        lecture_dir: Path,
        lecture_id: str = "lec03",
        source_prefix: str = "6.006-lec03",
    ) -> Dict[str, int]:
        """Ingest all assets in lecture_dir (video, subtitles, slides) into Qdrant."""
        logger.info("=== Starting Ingestion for %s (%s) ===", lecture_id, lecture_dir)
        lecture_path = Path(lecture_dir)
        if not lecture_path.exists():
            raise FileNotFoundError(f"Lecture directory does not exist: {lecture_path}")

        # 1. Locate assets
        mp4_files = list(lecture_path.glob("*.mp4"))
        pdf_files = list(lecture_path.glob("*.pdf"))
        vtt_files = list(lecture_path.glob("*.vtt"))
        srt_files = list(lecture_path.glob("*.srt"))

        all_chunks: List[MultimodalChunk] = []
        image_paths_for_clip: List[Optional[str]] = []

        # 2. Process Slides (PDF)
        if pdf_files:
            pdf_path = pdf_files[0]
            logger.info("Processing lecture slide notes: %s", pdf_path.name)
            slide_chunks = self.pdf_processor.process_pdf(
                str(pdf_path),
                source_id=f"{source_prefix}-notes",
                prefix=lecture_id,
            )
            for chunk in slide_chunks:
                all_chunks.append(chunk)
                image_paths_for_clip.append(chunk.provenance.slide_asset_path)
            logger.info("Generated %d slide chunks.", len(slide_chunks))
        else:
            logger.warning("No PDF notes found in %s", lecture_path)

        # 3. Process Video & Subtitles
        if mp4_files:
            video_path = mp4_files[0]
            logger.info("Processing lecture video: %s", video_path.name)

            # Subtitles: check existing vtt, or extract from mp4
            vtt_path = None
            if vtt_files:
                vtt_path = vtt_files[0]
            else:
                target_vtt = lecture_path / "subtitles.vtt"
                logger.info("Extracting embedded subtitle track to %s...", target_vtt)
                if extract_subtitles_from_video(str(video_path), str(target_vtt)):
                    vtt_path = target_vtt

            # Extract keyframes using LEARNet TIB
            logger.info("Running LEARNet Temporal Information Bottleneck keyframe extraction...")
            keyframes = self.video_extractor.extract_keyframes(
                str(video_path),
                output_dir=FRAMES_DIR,
                lecture_prefix=lecture_id,
            )

            if vtt_path and vtt_path.exists():
                cues = parse_vtt_subtitles(str(vtt_path))
                t_chunks = chunk_transcript_cues(cues)
                video_chunks = build_video_multimodal_chunks(
                    transcript_chunks=t_chunks,
                    keyframes=keyframes,
                    source_id=f"{source_prefix}-video",
                )
                for chunk in video_chunks:
                    all_chunks.append(chunk)
                    image_paths_for_clip.append(chunk.provenance.frame_asset_path)
                logger.info("Generated %d transcript-video chunks.", len(video_chunks))
            else:
                logger.warning("No subtitles available for video.")
        else:
            logger.warning("No MP4 video found in %s", lecture_path)

        if not all_chunks:
            logger.warning("No chunks generated for %s.", lecture_id)
            return {"total_chunks": 0, "slides": 0, "video_transcripts": 0}

        # 4. Generate Embeddings
        logger.info("Generating multi-vector embeddings for %d chunks...", len(all_chunks))
        texts = [chunk.text_for_embedding for chunk in all_chunks]

        # A. Dense embeddings (768d BGE)
        dense_vectors = self.embedder.embed_dense(texts)

        # B. Sparse BM25 vectors
        sparse_vectors = self.embedder.embed_sparse(texts)

        # C. Visual CLIP vectors (512d)
        visual_vectors: List[List[float]] = []
        valid_img_indices = [i for i, path in enumerate(image_paths_for_clip) if path and Path(path).exists()]
        valid_img_paths = [image_paths_for_clip[i] for i in valid_img_indices]

        if valid_img_paths:
            logger.info("Computing CLIP visual embeddings for %d visual assets...", len(valid_img_paths))
            computed_clip = self.embedder.embed_images(valid_img_paths)
            clip_map = {idx: computed_clip[j] for j, idx in enumerate(valid_img_indices)}
        else:
            clip_map = {}

        # Default fallback vector if chunk has no visual asset
        zero_visual = [0.0] * 512

        for i in range(len(all_chunks)):
            if i in clip_map:
                visual_vectors.append(clip_map[i])
            else:
                visual_vectors.append(zero_visual)

        # 5. Upsert into Qdrant
        logger.info("Upserting %d points into Qdrant collection...", len(all_chunks))
        self.storage.upsert_chunks(
            chunks=all_chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            visual_vectors=visual_vectors,
        )

        counts = {
            "total_chunks": len(all_chunks),
            "slides": len([c for c in all_chunks if c.identity.modality == "slide"]),
            "video_transcripts": len([c for c in all_chunks if c.identity.modality == "transcript"]),
        }
        logger.info("=== Ingestion complete: %s ===", counts)
        return counts

