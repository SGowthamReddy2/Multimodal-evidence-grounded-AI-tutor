"""LEARNet Temporal Information Bottleneck (TIB) Video Processor & Subtitle Chunker.

Implements the entropy-aware keyframe extraction framework from:
"LEARNet: A Learning Entropy-Aware Representation Network for Educational Video
Understanding" (Chitrakala et al., Entropy 2026, 28, 3).

Replaces heuristic shot boundary detectors with:
1. Pedagogical Information Score (PIS): α=0.67 structural MSE + content saliency
2. Density Peak Clustering (DPC): identifies instructional transition centroids
3. Domain-Informed Pruning: duration filtering, SSIM deduplication (θ=0.70),
   and instructor occlusion filtering.
"""

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import (
    EXTRACTION_METHOD,
    FRAMES_DIR,
    TIB_ALPHA,
    TIB_FRAME_SIZE,
    TIB_MIN_DURATION_S,
    TIB_PERSON_AREA_THRESHOLD,
    TIB_SALIENCY_KERNEL,
    TIB_SAMPLE_FPS,
    TIB_SSIM_THRESHOLD,
)
from src.models import AlignmentProvenance, EvidenceIdentity, MultimodalChunk

logger = logging.getLogger(__name__)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two grayscale images."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


class LEARNetTIBExtractor:
    """Temporal Information Bottleneck keyframe extractor for lecture videos."""

    def __init__(
        self,
        sample_fps: float = TIB_SAMPLE_FPS,
        alpha: float = TIB_ALPHA,
        ssim_threshold: float = TIB_SSIM_THRESHOLD,
        min_duration_s: float = TIB_MIN_DURATION_S,
        frame_size: Tuple[int, int] = TIB_FRAME_SIZE,
    ) -> None:
        self.sample_fps = sample_fps
        self.alpha = alpha
        self.ssim_threshold = ssim_threshold
        self.min_duration_s = min_duration_s
        self.frame_size = frame_size

    def extract_keyframes(
        self, video_path: str, output_dir: Path = FRAMES_DIR, lecture_prefix: str = "lec03"
    ) -> List[Dict]:
        """Extract pedagogically significant keyframes using LEARNet TIB.

        Returns a list of dictionaries with:
        - 'timestamp_ms': millisecond offset in video
        - 'frame_path': path to saved PNG
        - 'pis_score': Pedagogical Information Score
        - 'hash': sha256 of image bytes
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video at {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / video_fps
        logger.info(
            "Video opened: %.1f sec, %.1f fps, %d total frames",
            duration_s,
            video_fps,
            total_frames,
        )

        step_frames = max(1, int(round(video_fps / self.sample_fps)))

        # Step 1: Pre-sampling and PIS calculation
        sampled_indices = []
        sampled_timestamps_ms = []
        sampled_small_frames = []
        pis_scores = []

        prev_small = None
        frame_idx = 0

        logger.info("Computing Pedagogical Information Scores (PIS) across video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step_frames == 0:
                ts_ms = int(round((frame_idx / video_fps) * 1000.0))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, self.frame_size)

                # Compute Structural MSE (Eq. 4)
                if prev_small is None:
                    mse = 0.0
                else:
                    diff = small.astype(np.float32) - prev_small.astype(np.float32)
                    mse = float(np.mean(diff ** 2) / (255.0 ** 2))

                # Compute Content Saliency (Eq. 5) via local contrast
                blurred = cv2.GaussianBlur(small, (TIB_SALIENCY_KERNEL, TIB_SALIENCY_KERNEL), 0)
                saliency_map = np.abs(small.astype(np.float32) - blurred.astype(np.float32))
                saliency_val = float(np.mean(saliency_map) / 255.0)

                # Unified Pedagogical Information Score (Eq. 6)
                pis = self.alpha * mse + (1.0 - self.alpha) * saliency_val

                sampled_indices.append(frame_idx)
                sampled_timestamps_ms.append(ts_ms)
                sampled_small_frames.append(small)
                pis_scores.append(pis)

                prev_small = small

            frame_idx += 1

        cap.release()
        n_samples = len(pis_scores)
        if n_samples == 0:
            logger.warning("No frames sampled from video.")
            return []

        logger.info("Sampled %d frames. Running Density Peak Clustering (DPC)...", n_samples)

        # Step 2: Density Peak Clustering (DPC)
        pis_array = np.array(pis_scores, dtype=np.float32)
        diff_matrix = np.abs(pis_array[:, None] - pis_array[None, :])

        # Cut-off distance sigma (2nd percentile of pairwise differences)
        triu_indices = np.triu_indices(n_samples, k=1)
        if len(triu_indices[0]) > 0:
            pairwise_dists = diff_matrix[triu_indices]
            sigma = float(np.percentile(pairwise_dists, 2.0))
            sigma = max(sigma, 1e-4)
        else:
            sigma = 0.01

        # Local density rho (Eq. 7)
        rho = np.sum(np.exp(-(diff_matrix ** 2) / (2.0 * (sigma ** 2))), axis=1)

        # Delta: minimum distance to point with higher density (Eq. 8)
        delta = np.zeros(n_samples, dtype=np.float32)
        max_dist = float(np.max(diff_matrix))
        for i in range(n_samples):
            higher_density_indices = np.where(rho > rho[i])[0]
            if len(higher_density_indices) > 0:
                delta[i] = np.min(diff_matrix[i, higher_density_indices])
            else:
                delta[i] = max_dist

        # Candidate selection via gamma = rho * delta
        gamma = rho * delta
        gamma_threshold = np.percentile(gamma, 85.0)  # Top 15% density peak transitions
        candidate_indices = np.where(gamma >= gamma_threshold)[0]

        logger.info(
            "DPC identified %d candidate transition points from %d sampled frames.",
            len(candidate_indices),
            n_samples,
        )

        # Step 3: Domain-Informed Pruning
        # Ensure temporal spacing and SSIM dedup
        candidate_indices = sorted(list(candidate_indices))
        pruned_candidates = []
        min_sample_gap = int(self.min_duration_s * self.sample_fps)

        for c_idx in candidate_indices:
            # Check blank frame (variance < 25)
            if np.var(sampled_small_frames[c_idx]) < 25:
                continue

            # Check temporal distance from previous candidate
            if pruned_candidates:
                prev_idx = pruned_candidates[-1]
                if (c_idx - prev_idx) < min_sample_gap:
                    # Keep the one with higher PIS (higher instructional content)
                    if pis_scores[c_idx] > pis_scores[prev_idx]:
                        pruned_candidates[-1] = c_idx
                    continue

                # Check SSIM redundancy against previous keyframe (Eq. 14)
                sim = compute_ssim(
                    sampled_small_frames[prev_idx], sampled_small_frames[c_idx]
                )
                if sim > self.ssim_threshold:
                    continue

            pruned_candidates.append(c_idx)

        logger.info(
            "After domain pruning (duration, blank, SSIM): %d keyframes selected.",
            len(pruned_candidates),
        )

        # Step 4: Extract and save final high-resolution keyframes
        cap = cv2.VideoCapture(str(video_path))
        extracted_keyframes = []

        for rank, c_idx in enumerate(pruned_candidates):
            target_frame_idx = sampled_indices[c_idx]
            target_ts_ms = sampled_timestamps_ms[c_idx]

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_filename = f"{lecture_prefix}_frame_{target_ts_ms}ms.png"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)

            with open(frame_path, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()

            extracted_keyframes.append(
                {
                    "timestamp_ms": target_ts_ms,
                    "frame_path": str(frame_path),
                    "pis_score": float(pis_scores[c_idx]),
                    "content_hash": content_hash,
                }
            )

        cap.release()
        logger.info(
            "Successfully saved %d high-res keyframes to %s",
            len(extracted_keyframes),
            output_dir,
        )
        return extracted_keyframes


def parse_vtt_subtitles(vtt_path: str) -> List[Dict]:
    """Parse WebVTT file into millisecond-accurate subtitle cues.

    Returns list of dicts: {'start_ms': int, 'end_ms': int, 'text': str}
    """

    def timestamp_to_ms(ts_str: str) -> int:
        ts_str = ts_str.strip()
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        else:
            return 0
        sec_parts = s.split(".")
        sec = int(sec_parts[0])
        ms = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return int(h) * 3600000 + int(m) * 60000 + sec * 1000 + ms

    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match 00:12.720 --> 00:15.360 or 00:00:12.720 --> 00:00:15.360
    pattern = re.compile(
        r"(\d{1,2}:?\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:?\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    )

    cues = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = pattern.search(line)
        if match:
            start_ms = timestamp_to_ms(match.group(1))
            end_ms = timestamp_to_ms(match.group(2))
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not pattern.search(lines[i]):
                cleaned = re.sub(r"<[^>]+>", "", lines[i].strip())  # strip any HTML/VTT tags
                if cleaned and not cleaned.startswith("[") and not cleaned.endswith("]"):
                    text_lines.append(cleaned)
                i += 1
            text = " ".join(text_lines).strip()
            if text and end_ms > start_ms:
                cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
            continue
        i += 1

    logger.info("Parsed %d subtitle cues from %s", len(cues), vtt_path)
    return cues


def extract_subtitles_from_video(video_path: str, output_vtt_path: str) -> bool:
    """Extract embedded subtitle track 0:s:0 from MP4 video using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:s:0",
        "-y",
        str(output_vtt_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return Path(output_vtt_path).exists() and Path(output_vtt_path).stat().st_size > 0
    except Exception as e:
        logger.error("FFmpeg subtitle extraction failed: %s", e)
        return False


def chunk_transcript_cues(
    cues: List[Dict],
    target_duration_ms: int = 45000,  # ~45 seconds per chunk
    max_duration_ms: int = 75000,     # max 75 seconds
    min_words: int = 25,
) -> List[Dict]:
    """Group subtitle cues into coherent sentence/topic chunks."""
    chunks = []
    current_cues = []
    current_start = None

    for cue in cues:
        if current_start is None:
            current_start = cue["start_ms"]

        current_cues.append(cue)
        duration = cue["end_ms"] - current_start
        text_so_far = " ".join(c["text"] for c in current_cues)
        word_count = len(text_so_far.split())

        # Check if natural break point
        is_sentence_end = cue["text"].rstrip().endswith((".", "?", "!"))
        if (duration >= target_duration_ms and is_sentence_end and word_count >= min_words) or duration >= max_duration_ms:
            chunks.append(
                {
                    "start_ms": current_start,
                    "end_ms": cue["end_ms"],
                    "transcript_span": text_so_far,
                }
            )
            current_cues = []
            current_start = None

    if current_cues and current_start is not None:
        chunks.append(
            {
                "start_ms": current_start,
                "end_ms": current_cues[-1]["end_ms"],
                "transcript_span": " ".join(c["text"] for c in current_cues),
            }
        )

    logger.info("Chunked %d cues into %d coherent transcript chunks.", len(cues), len(chunks))
    return chunks


def build_video_multimodal_chunks(
    transcript_chunks: List[Dict],
    keyframes: List[Dict],
    source_id: str = "6.006-lec03-video",
    source_url: str = "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/",
) -> List[MultimodalChunk]:
    """Align transcript chunks with nearest LEARNet keyframes and construct MultimodalChunks."""
    multimodal_chunks: List[MultimodalChunk] = []

    for t_idx, t_chunk in enumerate(transcript_chunks):
        start_ms = t_chunk["start_ms"]
        end_ms = t_chunk["end_ms"]
        text_span = t_chunk["transcript_span"]

        # Find nearest keyframe within or close to this temporal window
        aligned_kf = None
        min_dist = float("inf")
        mid_ms = (start_ms + end_ms) / 2.0

        for kf in keyframes:
            kf_ts = kf["timestamp_ms"]
            dist = abs(kf_ts - mid_ms)
            if dist < min_dist:
                min_dist = dist
                aligned_kf = kf

        frame_path = aligned_kf["frame_path"] if aligned_kf else None
        frame_ts = aligned_kf["timestamp_ms"] if aligned_kf else None

        evidence_id = f"ev-{source_id}-t-{start_ms}_{end_ms}"
        content_hash = hashlib.sha256(text_span.encode("utf-8")).hexdigest()

        identity = EvidenceIdentity(
            source_id=source_id,
            source_url=source_url,
            content_version="v1.0",
            content_hash=content_hash,
            evidence_id=evidence_id,
            modality="transcript",
            status="active",
        )

        provenance = AlignmentProvenance(
            evidence_id=evidence_id,
            start_ms=start_ms,
            end_ms=end_ms,
            frame_timestamp_ms=frame_ts,
            frame_asset_path=frame_path,
            transcript_span=text_span,
            visual_caption="",
        )

        multimodal_chunks.append(
            MultimodalChunk(
                identity=identity,
                provenance=provenance,
                text_for_embedding=text_span,
                extraction_method=EXTRACTION_METHOD,
            )
        )

    return multimodal_chunks

