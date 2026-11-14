"""Centralised configuration constants for the Multimodal AI Tutor.

All directory paths, model identifiers, embedding dimensions, and tunable
parameters are defined here so that every module draws from a single source
of truth.
"""

import os
from pathlib import Path

# Disable remote cost map fetches in LiteLLM to ensure fast offline/sandboxed startup
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# ---------------------------------------------------------------------------
# Project root & data directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVIDENCE_DIR = DATA_DIR / "evidence"
FRAMES_DIR = EVIDENCE_DIR / "frames"
FILTERED_FRAMES_DIR = EVIDENCE_DIR / "filtered_frames"
SLIDES_DIR = EVIDENCE_DIR / "slides"
QDRANT_PATH = str(PROJECT_ROOT / "qdrant_data")
MEMORY_DB_PATH = str(DATA_DIR / "learner_memory.db")

# ---------------------------------------------------------------------------
# Multi-course defaults & versioning
# ---------------------------------------------------------------------------
DEFAULT_COURSE_ID = "mit-6.006"
COLLECTION_NAME = "course_evidence"
CONTENT_VERSION = "v1.0"
EXTRACTION_METHOD = "whisperx+learnet_tib+vlm"
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
# LLM & VLM Model Configurations (Unified via LiteLLM)
# ---------------------------------------------------------------------------
# Primary models (Groq)
PRIMARY_LLM_MODEL = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")
SPEED_LLM_MODEL = "groq/qwen/qwen3.8-27b"
FALLBACK_LLM_MODELS = ["groq/openai/gpt-oss-20b", "groq/qwen/qwen3.8-27b"]

# Vision-Language Models (VLM) for Keyframe and Slide Captioning
PRIMARY_VLM_MODEL = os.getenv("VLM_MODEL", "groq/qwen/qwen3.8-27b")
FALLBACK_VLM_MODELS = [
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
]

GROQ_MODEL_QUALITY = "openai/gpt-oss-120b"
GROQ_MODEL_SPEED = "qwen/qwen3.8-27b"
GROQ_MAX_TOKENS_A = 2500  # Generous headroom for full mathematical proofs, derivations & code
GROQ_MAX_TOKENS_B = 1600  # Fast Pareto profile with complete conceptual explanation
GROQ_MAX_TOKENS_CONVERSATIONAL = 1200  # Conversational & study coaching budget

# ---------------------------------------------------------------------------
# Keyframe Refinement & Pedagogical Maturity Filter (Stratified CLIP Refiner)
# ---------------------------------------------------------------------------
# Controls the secondary pruning stage on raw LEARNet TIB keyframes
KEYFRAME_TARGET_COUNT = 35         # Maximum mature keyframes to caption per lecture
KEYFRAME_BIN_SIZE_S = 90.0         # Stratified temporal window duration in seconds
KEYFRAME_BLUR_THRESHOLD = 25.0     # Baseline Laplacian variance for pure blur/noise rejection
KEYFRAME_DEDUP_SIM_THRESHOLD = 0.88 # SSIM dedup for adjacent mature frames across bins
KEYFRAME_CLIP_MIN_NET_SCORE = -0.04 # Minimum (pos_sim - neg_sim) to reject pure talking heads

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
for _dir in [DATA_DIR, RAW_DIR, EVIDENCE_DIR, FRAMES_DIR, FILTERED_FRAMES_DIR, SLIDES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

