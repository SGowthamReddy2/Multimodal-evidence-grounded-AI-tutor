"""Stratified CLIP Pedagogical Keyframe Refiner.

Replaces heuristic Laplacian/Canny edge thresholds with a two-tier semantic pipeline:
1. Zero-Shot Semantic Evaluation (FastEmbed CLIP ViT-B/32):
   Computes cosine similarity against pedagogical prompts (math derivations, slides, diagrams)
   versus non-pedagogical noise (talking heads, empty walls, motion blur) to derive a Net Pedagogical Score (NPS).
2. Stratified Temporal Binning:
   Partitions the lecture into fixed, non-overlapping temporal windows (e.g., 90s bins)
   and selects the peak-information frame within each window. This completely eliminates
   single-linkage cluster collapse and guarantees uniform coverage across the entire lecture.
3. Consecutive Inter-Bin Deduplication (SSIM):
   Removes static slides or unchanged boards that span across multiple temporal bins.
4. Disk Persistence:
   Saves mature pedagogical frames to data/evidence/filtered_frames/ alongside manifest metadata.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np
from fastembed import ImageEmbedding, TextEmbedding

from src.config import (
    CLIP_MODEL,
    CLIP_TEXT_MODEL,
    FILTERED_FRAMES_DIR,
    FRAMES_DIR,
    KEYFRAME_BIN_SIZE_S,
    KEYFRAME_BLUR_THRESHOLD,
    KEYFRAME_CLIP_MIN_NET_SCORE,
    KEYFRAME_DEDUP_SIM_THRESHOLD,
    KEYFRAME_TARGET_COUNT,
)
from src.ingestion.video_processor import compute_ssim

logger = logging.getLogger(__name__)

# Global singletons for FastEmbed models to avoid reloading weights repeatedly
_CLIP_IMG_EMBEDDER: Optional[ImageEmbedding] = None
_CLIP_TEXT_EMBEDDER: Optional[TextEmbedding] = None
_POS_TEXT_VECTORS: Optional[List[np.ndarray]] = None
_NEG_TEXT_VECTORS: Optional[List[np.ndarray]] = None

POSITIVE_PROMPTS = [
    "a blackboard covered with handwritten mathematical equations, algorithmic derivations, and diagrams",
    "a clear presentation slide with technical text, formulas, code, and diagrams",
    "a chalkboard covered with academic lecture notes, formulas, and diagrams",
]

NEGATIVE_PROMPTS = [
    "an instructor standing and talking in front of an empty wall or blank blackboard",
    "a blurry or out of focus video transition shot",
    "an empty lecture hall, podium, or audience",
]


def get_clip_embedders() -> Tuple[ImageEmbedding, TextEmbedding, List[np.ndarray], List[np.ndarray]]:
    """Initialize or retrieve cached FastEmbed CLIP embedders and normalized prompt vectors."""
    global _CLIP_IMG_EMBEDDER, _CLIP_TEXT_EMBEDDER, _POS_TEXT_VECTORS, _NEG_TEXT_VECTORS
    if _CLIP_IMG_EMBEDDER is None:
        logger.info("Initializing FastEmbed CLIP Vision & Text embedders (%s, %s)...", CLIP_MODEL, CLIP_TEXT_MODEL)
        _CLIP_IMG_EMBEDDER = ImageEmbedding(CLIP_MODEL)
        _CLIP_TEXT_EMBEDDER = TextEmbedding(CLIP_TEXT_MODEL)

        pos_raw = list(_CLIP_TEXT_EMBEDDER.embed(POSITIVE_PROMPTS))
        _POS_TEXT_VECTORS = [v / np.linalg.norm(v) for v in pos_raw]

        neg_raw = list(_CLIP_TEXT_EMBEDDER.embed(NEGATIVE_PROMPTS))
        _NEG_TEXT_VECTORS = [v / np.linalg.norm(v) for v in neg_raw]

    return _CLIP_IMG_EMBEDDER, _CLIP_TEXT_EMBEDDER, _POS_TEXT_VECTORS, _NEG_TEXT_VECTORS


def compute_frame_sharpness(gray_img: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance."""
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def compute_board_contrast_score(gray_img: np.ndarray) -> float:
    """Compute text/chalk contrast score using percentile difference."""
    p95 = np.percentile(gray_img, 95)
    p05 = np.percentile(gray_img, 5)
    return float(p95 - p05)


class KeyframePedagogicalRefiner:
    """Stratified CLIP Pedagogical Keyframe Refiner.
    
    Combines semantic zero-shot CLIP classification with stratified temporal binning
    to reliably select high-information blackboard and slide frames without omissions.
    """

    def __init__(
        self,
        target_count: int = KEYFRAME_TARGET_COUNT,
        bin_size_s: float = KEYFRAME_BIN_SIZE_S,
        blur_threshold: float = KEYFRAME_BLUR_THRESHOLD,
        min_net_score: float = KEYFRAME_CLIP_MIN_NET_SCORE,
        dedup_threshold: float = KEYFRAME_DEDUP_SIM_THRESHOLD,
    ) -> None:
        self.target_count = target_count
        self.bin_size_s = bin_size_s
        self.blur_threshold = blur_threshold
        self.min_net_score = min_net_score
        self.dedup_threshold = dedup_threshold

    def refine_keyframes(
        self,
        raw_keyframes: List[Dict],
        target_count: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_frames: bool = True,
        bin_size_s: Optional[float] = None,
    ) -> List[Dict]:
        """Filter and rank raw keyframes using Stratified CLIP scoring.
        
        Args:
            raw_keyframes: List of dicts with 'timestamp_ms', 'frame_path', 'pis_score'.
            target_count: Maximum mature keyframes to retain (defaults to self.target_count).
            output_dir: Destination folder for filtered frames (defaults to FILTERED_FRAMES_DIR).
            save_frames: Whether to copy filtered frames and write manifest JSON.
            bin_size_s: Non-overlapping temporal window in seconds (defaults to self.bin_size_s).
            
        Returns:
            List of filtered, mature keyframe metadata dicts sorted chronologically.
        """
        max_k = target_count or self.target_count
        if not raw_keyframes:
            return []

        logger.info(
            "Refining %d raw candidate keyframes using Stratified CLIP (target: ~%d frames)...",
            len(raw_keyframes),
            max_k,
        )

        # Ensure valid paths and timestamps
        valid_inputs = []
        for kf in raw_keyframes:
            f_path = kf.get("frame_path")
            if f_path and Path(f_path).exists():
                valid_inputs.append(kf)

        if not valid_inputs:
            logger.warning("No valid raw keyframe paths found on disk.")
            return []

        valid_inputs.sort(key=lambda x: x["timestamp_ms"])
        min_ts = valid_inputs[0]["timestamp_ms"]
        max_ts = valid_inputs[-1]["timestamp_ms"]
        duration_s = max(1.0, (max_ts - min_ts) / 1000.0)

        # Dynamic bin sizing: if fixed bin_size_s produces far more bins than target_count,
        # adjust bin_size_s to provide balanced temporal partitions
        active_bin_s = bin_size_s or self.bin_size_s
        if active_bin_s * max_k < duration_s:
            active_bin_s = duration_s / float(max_k)
        active_bin_ms = int(round(active_bin_s * 1000.0))

        logger.info(
            "Lecture temporal span: %.1f sec (%.1f mins). Using stratified bin window: %.1f sec.",
            duration_s,
            duration_s / 60.0,
            active_bin_s,
        )

        # Stage 1: Batch CLIP Zero-Shot Embedding & Quality Assessment
        img_embedder, _, pos_vecs, neg_vecs = get_clip_embedders()

        batch_size = 32
        scored_candidates = []

        for i in range(0, len(valid_inputs), batch_size):
            batch = valid_inputs[i : i + batch_size]
            batch_paths = [b["frame_path"] for b in batch]

            try:
                img_vectors = list(img_embedder.embed(batch_paths))
            except Exception as e:
                logger.error("Error embedding keyframe batch with CLIP: %s", e)
                continue

            for kf, iv in zip(batch, img_vectors):
                norm_iv = iv / (np.linalg.norm(iv) + 1e-6)

                # Positive & Negative Cosine Similarities
                pos_sim = max(float(np.dot(norm_iv, pv)) for pv in pos_vecs)
                neg_sim = max(float(np.dot(norm_iv, nv)) for nv in neg_vecs)
                nps = pos_sim - neg_sim

                # Quick pixel checks for pure noise rejection
                gray_img = cv2.imread(kf["frame_path"], cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    continue

                sharpness = compute_frame_sharpness(gray_img)
                contrast = compute_board_contrast_score(gray_img)
                pis = float(kf.get("pis_score", 0.0))

                # Gating: drop only extreme motion blur or pure talking heads with no board
                if sharpness < self.blur_threshold:
                    continue
                if nps < self.min_net_score and neg_sim > 0.30:
                    continue

                # Pedagogical composite score:
                # Rewards net positive content (math/slides), penalizes talking heads,
                # slightly boosts high-contrast text and high-PIS transitions.
                composite_score = (
                    (nps * 10.0)
                    + (pos_sim * 5.0)
                    + (contrast / 100.0)
                    + (pis * 2.0)
                )

                scored_candidates.append(
                    {
                        **kf,
                        "pos_sim": round(pos_sim, 4),
                        "neg_sim": round(neg_sim, 4),
                        "net_pedagogical_score": round(nps, 4),
                        "sharpness": round(sharpness, 2),
                        "contrast": round(contrast, 2),
                        "composite_score": round(composite_score, 4),
                    }
                )

        logger.info(
            "Semantic quality gate: retained %d of %d frames (filtered talking heads & blur).",
            len(scored_candidates),
            len(valid_inputs),
        )

        if not scored_candidates:
            logger.warning("No frames passed semantic gate; falling back to raw keyframes.")
            return valid_inputs[:max_k]

        # Stage 2: Stratified Temporal Binning (Fixed Non-Overlapping Windows)
        # Groups candidates into discrete temporal bins [0, bin_s), [bin_s, 2*bin_s)...
        # Guarantees that every pedagogical section is evaluated and represented.
        bins: Dict[int, List[Dict]] = {}
        for cand in scored_candidates:
            bin_idx = (cand["timestamp_ms"] - min_ts) // active_bin_ms
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(cand)

        bin_selected: List[Dict] = []
        for b_idx in sorted(bins.keys()):
            candidates_in_bin = bins[b_idx]
            # Pick the candidate with the highest pedagogical composite score in this window
            best_candidate = max(candidates_in_bin, key=lambda x: x["composite_score"])
            best_candidate["temporal_bin"] = int(b_idx)
            bin_selected.append(best_candidate)

        logger.info(
            "Stratified binning: extracted %d peak pedagogical frames across %d active time bins.",
            len(bin_selected),
            len(bins),
        )

        # Stage 3: Consecutive Inter-Bin Deduplication (SSIM)
        # Drops redundant static slides or identical blackboard views spanning multiple bins
        deduped: List[Dict] = []
        for cand in bin_selected:
            if not deduped:
                deduped.append(cand)
            else:
                prev_cand = deduped[-1]
                img1 = cv2.resize(cv2.imread(prev_cand["frame_path"], cv2.IMREAD_GRAYSCALE), (224, 224))
                img2 = cv2.resize(cv2.imread(cand["frame_path"], cv2.IMREAD_GRAYSCALE), (224, 224))
                if img1 is not None and img2 is not None:
                    sim = compute_ssim(img1, img2)
                    if sim > self.dedup_threshold:
                        # Static content across consecutive bins; keep the higher scoring one
                        if cand["composite_score"] > prev_cand["composite_score"]:
                            deduped[-1] = cand
                        continue
                deduped.append(cand)

        logger.info(
            "Inter-bin SSIM dedup (threshold=%.2f): retained %d unique mature keyframes.",
            self.dedup_threshold,
            len(deduped),
        )

        # Stage 4: Cap to target count if necessary while maintaining temporal spread
        final_keyframes = deduped
        if len(final_keyframes) > max_k:
            step = len(final_keyframes) / float(max_k)
            indices = [int(i * step) for i in range(max_k)]
            final_keyframes = [final_keyframes[i] for i in indices]

        final_keyframes.sort(key=lambda x: x["timestamp_ms"])

        logger.info(
            "Final Stratified CLIP refinement complete: %d mature keyframes ready for captioning.",
            len(final_keyframes),
        )

        # Stage 5: Save filtered keyframes to output_dir
        if save_frames and final_keyframes:
            dest_dir = Path(output_dir) if output_dir else FILTERED_FRAMES_DIR
            dest_dir.mkdir(parents=True, exist_ok=True)

            for kf in final_keyframes:
                orig_path_str = kf.get("frame_path")
                if orig_path_str:
                    orig_p = Path(orig_path_str)
                    if orig_p.exists():
                        target_p = dest_dir / orig_p.name
                        if orig_p.resolve() != target_p.resolve():
                            shutil.copy2(orig_p, target_p)
                        kf["frame_path"] = str(target_p.resolve())

            # Save manifest file
            manifest_path = dest_dir / "filtered_keyframes_manifest.json"
            manifest_payload = {
                "total_filtered": len(final_keyframes),
                "bin_size_seconds": round(active_bin_s, 1),
                "keyframes": [
                    {
                        "timestamp_ms": kf.get("timestamp_ms"),
                        "frame_path": str(kf.get("frame_path")),
                        "net_pedagogical_score": kf.get("net_pedagogical_score", 0.0),
                        "pos_sim": kf.get("pos_sim", 0.0),
                        "neg_sim": kf.get("neg_sim", 0.0),
                        "sharpness": kf.get("sharpness", 0.0),
                        "contrast": kf.get("contrast", 0.0),
                        "composite_score": kf.get("composite_score", 0.0),
                        "temporal_bin": kf.get("temporal_bin", 0),
                        "content_hash": kf.get("content_hash", ""),
                    }
                    for kf in final_keyframes
                ],
            }
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_payload, f, indent=2)
                logger.info(
                    "Saved %d filtered keyframes and manifest to %s",
                    len(final_keyframes),
                    dest_dir,
                )
            except Exception as e:
                logger.warning("Failed to write filtered keyframes manifest: %s", e)

        return final_keyframes


def main() -> None:
    """CLI runner to filter raw keyframes from disk using Stratified CLIP."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Filter raw keyframes to extract high-information, mature blackboard and slide frames using Stratified CLIP."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=str(FRAMES_DIR),
        help=f"Directory containing raw keyframes (default: {FRAMES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(FILTERED_FRAMES_DIR),
        help=f"Directory to save filtered mature keyframes (default: {FILTERED_FRAMES_DIR})",
    )
    parser.add_argument(
        "--max-count",
        "-k",
        type=int,
        default=KEYFRAME_TARGET_COUNT,
        help=f"Maximum mature keyframes to retain (default: {KEYFRAME_TARGET_COUNT})",
    )
    parser.add_argument(
        "--bin-size",
        "-b",
        type=float,
        default=KEYFRAME_BIN_SIZE_S,
        help=f"Stratified temporal window size in seconds (default: {KEYFRAME_BIN_SIZE_S})",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        logger.error("Input directory does not exist: %s", in_dir)
        sys.exit(1)

    # Discover frame images
    extensions = ("*.png", "*.jpg", "*.jpeg")
    frame_files = []
    for ext in extensions:
        frame_files.extend(list(in_dir.glob(ext)))
    frame_files = sorted(list(set(frame_files)))

    if not frame_files:
        logger.warning("No frame images found in %s", in_dir)
        sys.exit(0)

    logger.info("Discovered %d raw keyframe images in %s", len(frame_files), in_dir)

    raw_keyframes = []
    for f in frame_files:
        ts_val = 0
        match = re.search(r"frame_(\d+)ms", f.name)
        if match:
            try:
                ts_val = int(match.group(1))
            except ValueError:
                ts_val = 0
        else:
            num_match = re.search(r"(\d+)", f.stem)
            if num_match:
                try:
                    ts_val = int(num_match.group(1))
                except ValueError:
                    ts_val = 0

        raw_keyframes.append(
            {
                "timestamp_ms": ts_val,
                "frame_path": str(f.resolve()),
                "pis_score": 0.5,
                "content_hash": f.name,
            }
        )

    refiner = KeyframePedagogicalRefiner(
        target_count=args.max_count,
        bin_size_s=args.bin_size,
    )

    filtered = refiner.refine_keyframes(
        raw_keyframes=raw_keyframes,
        target_count=args.max_count,
        output_dir=Path(args.output_dir),
        save_frames=True,
        bin_size_s=args.bin_size,
    )

    print("\n" + "=" * 60)
    print("STRATIFIED CLIP KEYFRAME FILTER SUMMARY")
    print("=" * 60)
    print(f"Raw Input Keyframes:     {len(raw_keyframes)}")
    print(f"Filtered Mature Frames:  {len(filtered)}")
    print(f"Saved Destination:       {Path(args.output_dir).resolve()}")
    print(f"Manifest File:           {Path(args.output_dir).resolve() / 'filtered_keyframes_manifest.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
