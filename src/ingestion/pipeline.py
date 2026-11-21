"""End-to-end multimodal ingestion pipeline orchestrator.

Coordinates:
1. Multi-course & multi-lecture auto-discovery
2. Video subtitle extraction & chunking
3. LEARNet TIB keyframe extraction + pedagogical maturity pruning
4. Vision-Language Model (VLM) image captioning with rate-limit tracking
5. Slide PDF rendering & vector text extraction
6. Multi-vector embedding (Dense BGE-v1.5 + Sparse BM25 + Visual CLIP)
7. Atomic Qdrant upsert with course/lecture payload indices
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.config import (
    DEFAULT_COURSE_ID,
    FILTERED_FRAMES_DIR,
    FRAMES_DIR,
    KEYFRAME_TARGET_COUNT,
    PROJECT_ROOT,
    RAW_DIR,
    SLIDES_DIR,
)
from src.ingestion.embedder import MultimodalEmbedder
from src.ingestion.keyframe_filter import KeyframePedagogicalRefiner
from src.ingestion.pdf_processor import PDFSlideProcessor
from src.ingestion.video_processor import (
    LEARNetTIBExtractor,
    build_video_multimodal_chunks,
    chunk_transcript_cues,
    extract_subtitles_from_video,
    parse_vtt_subtitles,
)
from src.ingestion.vlm_captioner import VLMCaptioner
from src.models import MultimodalChunk
from src.storage import QdrantStorageManager

logger = logging.getLogger(__name__)


def resolve_lecture_dir(
    lecture_dir: Union[str, Path],
    course_id: Optional[str] = None,
) -> Path:
    """Resolve a lecture directory string or Path into a verified existing Path.

    Handles absolute paths, paths relative to the current working directory,
    paths relative to PROJECT_ROOT, paths relative to RAW_DIR, and shorthand names.
    """
    p = Path(lecture_dir)
    if p.exists() and p.is_dir():
        return p.resolve()

    candidates = [
        Path.cwd() / p,
        PROJECT_ROOT / p,
        RAW_DIR / p,
    ]
    if course_id:
        candidates.append(RAW_DIR / course_id / p)
        candidates.append(RAW_DIR / course_id / p.name)

    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()

    # Search by folder name within RAW_DIR as fallback
    folder_name = p.name
    if RAW_DIR.exists():
        for matched in RAW_DIR.rglob(folder_name):
            if matched.is_dir():
                return matched.resolve()

    return (PROJECT_ROOT / p).resolve()


def discover_courses_and_lectures(base_dir: Path = RAW_DIR) -> List[Tuple[str, str, Path]]:
    """Discover courses and lectures in data/raw/.

    Supports two structures:
    1. Hierarchical: data/raw/<course_id>/<lecture_id>/
    2. Flat/Legacy:   data/raw/lecture-03/ -> (DEFAULT_COURSE_ID, "lecture-03")
    """
    results: List[Tuple[str, str, Path]] = []
    if not base_dir.exists():
        return results

    for item in sorted(base_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        # Check if legacy flat lecture folder (e.g. data/raw/lecture-03)
        if item.name.lower().startswith("lecture-") or item.name.lower().startswith("lec"):
            # Check if it contains media files
            if any(item.glob("*.mp4")) or any(item.glob("*.pdf")) or any(item.glob("*.vtt")):
                results.append((DEFAULT_COURSE_ID, item.name, item))
            continue

        # Otherwise treat as course directory: data/raw/<course_id>/<lecture_id>/
        course_id = item.name
        for sub in sorted(item.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                if any(sub.glob("*.mp4")) or any(sub.glob("*.pdf")) or any(sub.glob("*.vtt")):
                    results.append((course_id, sub.name, sub))

    return results


class IngestionPipeline:
    """Orchestrates ingestion of raw lecture assets into Qdrant."""

    def __init__(
        self,
        storage: Optional[QdrantStorageManager] = None,
        embedder: Optional[MultimodalEmbedder] = None,
        vlm_captioner: Optional[VLMCaptioner] = None,
    ) -> None:
        self.storage = storage or QdrantStorageManager()
        self.embedder = embedder or MultimodalEmbedder()
        self.video_extractor = LEARNetTIBExtractor()
        self.keyframe_refiner = KeyframePedagogicalRefiner()
        self.vlm_captioner = vlm_captioner or VLMCaptioner()
        self.pdf_processor = PDFSlideProcessor()

    def ingest_lecture(
        self,
        lecture_dir: Path,
        course_id: str = DEFAULT_COURSE_ID,
        lecture_id: str = "lecture-03",
        source_prefix: Optional[str] = None,
        enable_vlm: bool = True,
    ) -> Dict[str, int]:
        """Ingest all assets in lecture_dir (video, subtitles, slides) into Qdrant."""
        logger.info(
            "=== Starting Ingestion for Course: %s | Lecture: %s (%s) ===",
            course_id,
            lecture_id,
            lecture_dir,
        )
        lecture_path = resolve_lecture_dir(lecture_dir, course_id=course_id)
        if not lecture_path.exists():
            raise FileNotFoundError(
                f"Lecture directory does not exist: {lecture_path}\n"
                f"Searched for '{lecture_dir}' relative to cwd ({Path.cwd()}), "
                f"project root ({PROJECT_ROOT}), and raw directory ({RAW_DIR})."
            )

        source_pfx = source_prefix or f"{course_id}-{lecture_id}"

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
                source_id=f"{source_pfx}-notes",
                prefix=f"{course_id}_{lecture_id}",
                course_id=course_id,
                lecture_id=lecture_id,
            )

            # Generate VLM captions for slides if enabled
            if enable_vlm:
                logger.info("Generating VLM pedagogical captions for %d slides...", len(slide_chunks))
                for chunk in slide_chunks:
                    slide_img = chunk.provenance.slide_asset_path
                    if slide_img and Path(slide_img).exists():
                        caption = self.vlm_captioner.caption_image(
                            slide_img, content_hash=chunk.identity.content_hash
                        )
                        if caption:
                            chunk.provenance.visual_caption = caption
                            chunk.text_for_embedding = f"{chunk.text_for_embedding}\n[Slide Diagram Context]: {caption}"

            for chunk in slide_chunks:
                all_chunks.append(chunk)
                image_paths_for_clip.append(chunk.provenance.slide_asset_path)
            logger.info("Generated %d slide chunks.", len(slide_chunks))
        else:
            logger.warning("No PDF notes found in %s", lecture_path)

        # 3. Process Video & Subtitles
        vtt_path = None
        if vtt_files:
            vtt_path = vtt_files[0]

        raw_keyframes: List[Dict] = []

        if mp4_files:
            video_path = mp4_files[0]
            logger.info("Processing lecture video: %s", video_path.name)

            # Subtitles: check existing vtt, or extract from mp4
            if not vtt_path:
                target_vtt = lecture_path / "subtitles.vtt"
                logger.info("Extracting embedded subtitle track to %s...", target_vtt)
                if extract_subtitles_from_video(str(video_path), str(target_vtt)):
                    vtt_path = target_vtt

            # Step 3A: Extract raw keyframes using LEARNet TIB
            logger.info("Running LEARNet Temporal Information Bottleneck keyframe extraction...")
            raw_keyframes = self.video_extractor.extract_keyframes(
                str(video_path),
                output_dir=FRAMES_DIR,
                lecture_prefix=f"{course_id}_{lecture_id}",
            )
        else:
            # Check if pre-extracted keyframes already exist in FRAMES_DIR
            existing_frames = sorted(list(FRAMES_DIR.glob(f"*{lecture_id}*.png")))
            if not existing_frames:
                existing_frames = sorted(list(FRAMES_DIR.glob("*.png")))

            if existing_frames:
                logger.info(
                    "Raw MP4 video not present, but using %d pre-extracted keyframes in %s.",
                    len(existing_frames),
                    FRAMES_DIR,
                )
                for ef in existing_frames:
                    try:
                        ts_str = ef.stem.split("_")[-1].replace("ms", "")
                        ts_val = int(ts_str)
                    except ValueError:
                        ts_val = 0
                    raw_keyframes.append(
                        {
                            "timestamp_ms": ts_val,
                            "frame_path": str(ef),
                            "pis_score": 0.5,
                            "content_hash": ef.name,
                        }
                    )
            else:
                logger.warning("No MP4 video or pre-extracted frames found in %s", lecture_path)

        if raw_keyframes or (vtt_path and vtt_path.exists()):

            # Step 3B: Pedagogical Maturity Refinement (Filter blurred, empty, and incomplete frames)
            mature_keyframes = self.keyframe_refiner.refine_keyframes(
                raw_keyframes=raw_keyframes,
                target_count=KEYFRAME_TARGET_COUNT,
                output_dir=FILTERED_FRAMES_DIR,
                save_frames=True,
            )

            # Step 3C: Generate VLM Captions for mature keyframes
            keyframe_captions: Dict[str, str] = {}
            if enable_vlm and mature_keyframes:
                logger.info(
                    "Generating VLM captions for %d mature keyframes (Groq Qwen 2.5 27B / fallbacks)...",
                    len(mature_keyframes),
                )
                for kf in mature_keyframes:
                    f_path = kf.get("frame_path")
                    f_hash = kf.get("content_hash")
                    if f_path and Path(f_path).exists():
                        cap = self.vlm_captioner.caption_image(f_path, content_hash=f_hash)
                        if cap:
                            keyframe_captions[f_path] = cap

            # Step 3D: Build multimodal chunks
            if vtt_path and vtt_path.exists():
                cues = parse_vtt_subtitles(str(vtt_path))
                t_chunks = chunk_transcript_cues(cues)
                video_chunks = build_video_multimodal_chunks(
                    transcript_chunks=t_chunks,
                    keyframes=mature_keyframes,
                    source_id=f"{source_pfx}-video",
                    course_id=course_id,
                    lecture_id=lecture_id,
                    keyframe_captions=keyframe_captions,
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
            logger.warning("No chunks generated for %s / %s.", course_id, lecture_id)
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
        valid_img_indices = [
            i for i, path in enumerate(image_paths_for_clip) if path and Path(path).exists()
        ]
        valid_img_paths = [image_paths_for_clip[i] for i in valid_img_indices]

        if valid_img_paths:
            logger.info("Computing CLIP visual embeddings for %d visual assets...", len(valid_img_paths))
            computed_clip = self.embedder.embed_images(valid_img_paths)
            clip_map = {idx: computed_clip[j] for j, idx in enumerate(valid_img_indices)}
        else:
            clip_map = {}

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
        logger.info("=== Ingestion complete for %s / %s: %s ===", course_id, lecture_id, counts)
        return counts

    def ingest_all(self, base_dir: Path = RAW_DIR, enable_vlm: bool = True) -> Dict[str, Any]:
        """Auto-discover and ingest all courses and lectures."""
        discovered = discover_courses_and_lectures(base_dir)
        logger.info("Discovered %d course/lecture packages to ingest.", len(discovered))
        results = {}
        for course_id, lecture_id, path in discovered:
            key = f"{course_id}/{lecture_id}"
            try:
                res = self.ingest_lecture(
                    lecture_dir=path,
                    course_id=course_id,
                    lecture_id=lecture_id,
                    enable_vlm=enable_vlm,
                )
                results[key] = res
            except Exception as e:
                logger.error("Failed to ingest %s: %s", key, e)
                results[key] = {"error": str(e)}
        return results

    def close(self) -> None:
        """Release storage and client resources."""
        if hasattr(self, "storage") and self.storage:
            self.storage.close()
