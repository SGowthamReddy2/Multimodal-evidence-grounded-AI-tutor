"""Centralised configuration constants for the Multimodal AI Tutor.

All directory paths, model identifiers, embedding dimensions, and tunable
parameters are defined here so that every module draws from a single source
of truth.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & data directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVIDENCE_DIR = DATA_DIR / "evidence"
FRAMES_DIR = EVIDENCE_DIR / "frames"
SLIDES_DIR = EVIDENCE_DIR / "slides"
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_data")
MEMORY_DB_PATH = str(DATA_DIR / "learner_memory.db")

# ---------------------------------------------------------------------------
# Collection & versioning
# ---------------------------------------------------------------------------
COLLECTION_NAME = "course_evidence"
CONTENT_VERSION = "v1.0"
EXTRACTION_METHOD = "whisperx+learnet_tib"
ATTRIBUTION = "MIT OCW CC BY-NC-SA 4.0 - Prof. Erik Demaine"

# ---------------------------------------------------------------------------
# Embedding model identifiers & dimensions
# ---------------------------------------------------------------------------
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768

SPARSE_MODEL = "Qdrant/bm25"  # FastEmbed BM25

CLIP_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
VISUAL_DIM = 512

# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# LEARNet TIB keyframe extraction parameters
# ---------------------------------------------------------------------------
TIB_SAMPLE_FPS = 1.0            # Uniform sampling rate (frames per second)
TIB_ALPHA = 0.67                # PIS weighting: structural MSE vs. saliency
TIB_SSIM_THRESHOLD = 0.70       # Consecutive frame dedup threshold
TIB_MIN_DURATION_S = 3.0        # Minimum screen-time for a valid keyframe
TIB_PERSON_AREA_THRESHOLD = 0.35  # Max instructor area before pruning
TIB_FRAME_SIZE = (224, 224)     # Grayscale resize for scoring
TIB_SALIENCY_KERNEL = 11        # Gaussian blur kernel for saliency map

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
RRF_K = 30                      # Reciprocal Rank Fusion constant
PREFETCH_LIMIT_A = 50           # Config A (Quality)
PREFETCH_LIMIT_B = 20           # Config B (Speed)
RERANK_TOP_N = 20               # Candidates sent to cross-encoder
FINAL_TOP_K = 5                 # Final evidence count

# Retrieval weights
WEIGHT_DENSE = 1.0
WEIGHT_SPARSE = 0.8
WEIGHT_VISUAL_DIAGRAM = 0.9
WEIGHT_VISUAL_TEXT = 0.3

# ---------------------------------------------------------------------------
# Abstention thresholds
# ---------------------------------------------------------------------------
ABSTENTION_FULL_THRESHOLD = 0.65
ABSTENTION_QUALIFIED_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# LLM / Groq API
# ---------------------------------------------------------------------------
GROQ_MODEL_QUALITY = "openai/gpt-oss-120b"   # Frontier 120B reasoning model on Groq
GROQ_MODEL_SPEED = "qwen/qwen3.8-27b"        # Ultra-fast 27B instruction model on Groq (~520 tok/s)
GROQ_MAX_TOKENS_A = 1200
GROQ_MAX_TOKENS_B = 800

# ---------------------------------------------------------------------------
# Learner memory
# ---------------------------------------------------------------------------
MEMORY_CONFIDENCE_HALF_LIFE_DAYS = 30
MEMORY_MAX_RECORD_LENGTH = 180     # Max characters for a memory value
MEMORY_INJECTION_PATTERNS = [
    "[INST]", "SYSTEM:", "<|im_start|>", "<|im_end|>",
    "ignore all previous", "you are now", "forget everything",
]

# ---------------------------------------------------------------------------
# Slide rendering
# ---------------------------------------------------------------------------
SLIDE_DPI = 150

# ---------------------------------------------------------------------------
# Ensure data directories exist
# ---------------------------------------------------------------------------
for _dir in [DATA_DIR, RAW_DIR, EVIDENCE_DIR, FRAMES_DIR, SLIDES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

