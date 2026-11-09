"""Pedagogical Keyframe Refiner & Blackboard Maturity Filter.

Filters raw keyframes extracted by LEARNet TIB (200-400 frames down to 25-35 mature
frames) before calling expensive Vision-Language Model (VLM) APIs.

Filters out:
1. Motion-blurred and out-of-focus camera transition frames (Laplacian variance).
2. Empty blackboards, plain walls, and instructor talking-head frames with no content (Canny edge density).
3. Incomplete intermediate chalkboard writing states by selecting the peak-information
   mature state in each pedagogical scene cluster.
4. Saves filtered mature keyframes to disk (data/evidence/filtered_frames/) with manifest metadata.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path if run directly
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import numpy as np

from src.config import (
    FILTERED_FRAMES_DIR,
    FRAMES_DIR,
    KEYFRAME_BLUR_THRESHOLD,
    KEYFRAME_DEDUP_SIM_THRESHOLD,
    KEYFRAME_MIN_EDGE_DENSITY,
    KEYFRAME_TARGET_COUNT,
)

logger = logging.getLogger(__name__)


def compute_frame_sharpness(gray_img: np.ndarray) -> float:
    """Compute image sharpness using Laplacian variance."""
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def compute_edge_density(gray_img: np.ndarray) -> float:
    """Compute the ratio of high-frequency edges (text, equations, diagrams, slide borders)."""
    edges = cv2.Canny(gray_img, 50, 150)
    total_pixels = gray_img.shape[0] * gray_img.shape[1]
    if total_pixels == 0:
        return 0.0
    return float(np.count_nonzero(edges) / total_pixels)


def compute_board_contrast_score(gray_img: np.ndarray) -> float:
    """Compute text/chalk contrast score using percentile variance."""
    p95 = np.percentile(gray_img, 95)
    p05 = np.percentile(gray_img, 5)
    return float(p95 - p05)


class KeyframePedagogicalRefiner:
    """Selects the most pedagogically informative, mature keyframes for VLM captioning and UI display."""

    def __init__(
        self,
        target_count: int = KEYFRAME_TARGET_COUNT,
        blur_threshold: float = KEYFRAME_BLUR_THRESHOLD,
        min_edge_density: float = KEYFRAME_MIN_EDGE_DENSITY,
        dedup_threshold: float = KEYFRAME_DEDUP_SIM_THRESHOLD,
    ) -> None:
        self.target_count = target_count
        self.blur_threshold = blur_threshold
        self.min_edge_density = min_edge_density
        self.dedup_threshold = dedup_threshold

    def refine_keyframes(
        self,
        raw_keyframes: List[Dict],
        target_count: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        save_frames: bool = True,
    ) -> List[Dict]:
        """Filter and rank raw keyframes, returning the top mature pedagogical frames.
        
        If save_frames is True, copies/saves the selected mature frames into output_dir
        (defaults to data/evidence/filtered_frames/) and writes filtered_keyframes_manifest.json.
        """
        max_k = target_count or self.target_count
        if not raw_keyframes:
            return []

        logger.info(
            "Refining %d raw keyframes (target max: %d mature frames)...",
            len(raw_keyframes),
            max_k,
        )

        candidates = []
        for kf in raw_keyframes:
            frame_path = kf.get("frame_path")
            if not frame_path or not Path(frame_path).exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Blur test
            sharpness = compute_frame_sharpness(gray)
            if sharpness < self.blur_threshold:
                continue

            # 2. Blackboard / Slide edge density test
            edge_density = compute_edge_density(gray)
            if edge_density < self.min_edge_density:
                continue

            # 3. Text contrast test
            contrast = compute_board_contrast_score(gray)
            if contrast < 40.0:  # low contrast washed-out frames
                continue

            pis = float(kf.get("pis_score", 0.0))

            # Composite maturity score: favours sharp, high-density filled boards
            maturity_score = (edge_density * 100.0) * 0.5 + (sharpness / 100.0) * 0.2 + (contrast / 10.0) * 0.3

            candidates.append(
                {
                    **kf,
                    "sharpness": sharpness,
                    "edge_density": edge_density,
                    "contrast": contrast,
                    "maturity_score": maturity_score,
                    "gray_small": cv2.resize(gray, (160, 90)),
                }
            )

        logger.info(
            "Quality filter passed %d of %d frames (discarded blurred & empty boards).",
            len(candidates),
            len(raw_keyframes),
        )

        if not candidates:
            logger.warning("No frames passed strict quality filter. Falling back to top PIS frames.")
            final_keyframes = raw_keyframes[:max_k]
        else:
            # 4. Temporal Scene Clustering & Derivation Peak Selection
            # Group frames that occur within 35 seconds of each other into the same pedagogical scene
            # and select the single frame with highest maturity (completed derivation).
            scene_clusters: List[List[Dict]] = []
            current_cluster: List[Dict] = []
            temporal_window_ms = 35000  # 35 seconds

            for cand in candidates:
                if not current_cluster:
                    current_cluster.append(cand)
                else:
                    last_ts = current_cluster[-1]["timestamp_ms"]
                    if cand["timestamp_ms"] - last_ts <= temporal_window_ms:
                        current_cluster.append(cand)
                    else:
                        scene_clusters.append(current_cluster)
                        current_cluster = [cand]

            if current_cluster:
                scene_clusters.append(current_cluster)

            # In each cluster, pick the frame with the maximum maturity (filled board state)
            peak_frames = []
            for cluster in scene_clusters:
                best_frame = max(cluster, key=lambda x: x["maturity_score"])
                peak_frames.append(best_frame)

            logger.info(
                "Clustered into %d pedagogical scenes, extracted peak filled-board frames.",
                len(peak_frames),
            )

            # 5. Cap to target count by overall maturity score
            if len(peak_frames) > max_k:
                peak_frames.sort(key=lambda x: x["maturity_score"], reverse=True)
                peak_frames = peak_frames[:max_k]

            # 6. Re-sort chronologically by timestamp
            peak_frames.sort(key=lambda x: x["timestamp_ms"])

            # Clean temporary fields
            final_keyframes = []
            for pf in peak_frames:
                pf_clean = {k: v for k, v in pf.items() if k != "gray_small"}
                final_keyframes.append(pf_clean)

        logger.info(
            "Successfully refined to %d mature, high-information keyframes.",
            len(final_keyframes),
        )

        # 7. Save filtered keyframes to output_dir
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
                "keyframes": [
                    {
                        "timestamp_ms": kf.get("timestamp_ms"),
                        "frame_path": str(kf.get("frame_path")),
                        "sharpness": round(float(kf.get("sharpness", 0.0)), 2),
                        "edge_density": round(float(kf.get("edge_density", 0.0)), 4),
                        "contrast": round(float(kf.get("contrast", 0.0)), 2),
                        "maturity_score": round(float(kf.get("maturity_score", 0.0)), 2),
                        "pis_score": round(float(kf.get("pis_score", 0.0)), 4),
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
    """CLI runner to filter raw keyframes from disk and save mature frames."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Filter raw keyframes to extract high-information, mature blackboard and slide frames."
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
        "--blur-threshold",
        type=float,
        default=KEYFRAME_BLUR_THRESHOLD,
        help=f"Laplacian variance blur threshold (default: {KEYFRAME_BLUR_THRESHOLD})",
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=KEYFRAME_MIN_EDGE_DENSITY,
        help=f"Minimum Canny edge density ratio (default: {KEYFRAME_MIN_EDGE_DENSITY})",
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
        blur_threshold=args.blur_threshold,
        min_edge_density=args.min_edge_density,
    )

    filtered = refiner.refine_keyframes(
        raw_keyframes=raw_keyframes,
        target_count=args.max_count,
        output_dir=Path(args.output_dir),
        save_frames=True,
    )

    print("\n" + "=" * 60)
    print("KEYFRAME FILTER SUMMARY")
    print("=" * 60)
    print(f"Raw Input Keyframes:     {len(raw_keyframes)}")
    print(f"Filtered Mature Frames:  {len(filtered)}")
    print(f"Saved Destination:       {Path(args.output_dir).resolve()}")
    print(f"Manifest File:           {Path(args.output_dir).resolve() / 'filtered_keyframes_manifest.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
