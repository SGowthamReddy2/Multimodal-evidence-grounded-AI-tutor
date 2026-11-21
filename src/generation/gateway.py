"""Unified Model Gateway using LiteLLM for LLMs and Vision-Language Models (VLMs).

Provides:
1. Unified interface for LLM text reasoning and VLM multimodal image captioning.
2. Dynamic rate-limit handling with automatic fallback to secondary models.
3. Accurate rate-limit reset time tracking with automatic failback to the primary
   model as soon as its reset window clears.
"""

import base64
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set local cost map BEFORE importing litellm to avoid outbound HTTP retries
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from dotenv import load_dotenv

from src.config import (
    FALLBACK_LLM_MODELS,
    FALLBACK_VLM_MODELS,
    PRIMARY_LLM_MODEL,
    PRIMARY_VLM_MODEL,
    PROJECT_ROOT,
    SPEED_LLM_MODEL,
)

# Load environment variables (.env)
load_dotenv(PROJECT_ROOT / ".env")

# Ensure LiteLLM uses local backup pricing map in sandboxed/offline environments
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
litellm.telemetry = False

logger = logging.getLogger(__name__)


class RateLimitSkipException(Exception):
    """Raised when rate limit sleeping duration is >= 60 seconds to skip the query."""
    pass


def parse_rate_limit_reset(exception_or_headers: Any, default_seconds: float = 10.0) -> float:
    """Extract reset duration in seconds from 429 response headers or error messages."""
    headers = getattr(exception_or_headers, "headers", None) or {}
    if hasattr(exception_or_headers, "response") and hasattr(exception_or_headers.response, "headers"):
        headers = exception_or_headers.response.headers

    # Check standard and provider-specific rate limit headers
    candidate_keys = [
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ]
    for key in candidate_keys:
        val = headers.get(key)
        if val:
            try:
                # Strip units like 's', 'ms'
                clean_val = str(val).strip().lower()
                if clean_val.endswith("ms"):
                    return max(1.0, float(clean_val[:-2]) / 1000.0)
                if clean_val.endswith("s"):
                    clean_val = clean_val[:-1]
                return max(1.0, float(clean_val))
            except (ValueError, TypeError):
                pass

    # Parse error text if header wasn't available (e.g. 'try again in 12.5s', 'retry after 5s')
    err_str = str(exception_or_headers)
    patterns = [
        r"try again in (\d+(?:\.\d+)?)\s*s",
        r"retry after (\d+(?:\.\d+)?)\s*s",
        r"try again in (\d+(?:\.\d+)?)\s*seconds?",
        r"rate limit.*wait(?:ing)?\s*(\d+(?:\.\d+)?)\s*s",
    ]
    for pat in patterns:
        match = re.search(pat, err_str, re.IGNORECASE)
        if match:
            try:
                return max(1.0, float(match.group(1)))
            except ValueError:
                pass

    m_match = re.search(r"(\d+)\s*m(?:inutes?)?\s*(\d+(?:\.\d+)?)?\s*s?", err_str, re.IGNORECASE)
    if m_match:
        try:
            mins = float(m_match.group(1))
            secs = float(m_match.group(2) or 0.0)
            return max(1.0, mins * 60.0 + secs)
        except ValueError:
            pass

    return default_seconds


class LiteLLMGateway:
    """Unified provider gateway with rate-limit tracking and automatic fallback."""

    _instance: Optional["LiteLLMGateway"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        primary_llm: Optional[str] = None,
        fallback_llms: Optional[List[str]] = None,
        primary_vlm: Optional[str] = None,
        fallback_vlms: Optional[List[str]] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self.primary_llm = primary_llm or PRIMARY_LLM_MODEL
        self.fallback_llms = fallback_llms or FALLBACK_LLM_MODELS
        self.primary_vlm = primary_vlm or PRIMARY_VLM_MODEL
        self.fallback_vlms = fallback_vlms or FALLBACK_VLM_MODELS

        # Cooldown trackers: maps task_type ("llm" or "vlm") to timestamp when cooldown ends
        self._cooldown_until: Dict[str, float] = {}
        self._current_active_model: Dict[str, str] = {
            "llm": self.primary_llm,
            "vlm": self.primary_vlm,
        }
        self._initialized = True
        logger.info(
            "LiteLLMGateway initialized (Primary LLM: %s, Primary VLM: %s)",
            self.primary_llm,
            self.primary_vlm,
        )

    def _get_active_model(self, task_type: str, requested_model: Optional[str] = None) -> str:
        """Return the active model for a task, automatically checking cooldown resets."""
        if requested_model:
            return requested_model

        cooldown_end = self._cooldown_until.get(task_type, 0.0)
        now = time.time()

        primary = self.primary_llm if task_type == "llm" else self.primary_vlm

        if cooldown_end > 0.0 and now >= cooldown_end:
            logger.info(
                "Rate limit cooldown for %s expired. Restoring primary model: %s",
                task_type.upper(),
                primary,
            )
            self._cooldown_until[task_type] = 0.0
            self._current_active_model[task_type] = primary
            return primary

        return self._current_active_model.get(task_type, primary)

    def _record_rate_limit(self, task_type: str, failed_model: str, exc: Exception) -> str:
        """Record rate limit hit, schedule cooldown, and pick the next fallback model."""
        reset_secs = parse_rate_limit_reset(exc, default_seconds=60.0)
        self._cooldown_until[task_type] = time.time() + reset_secs

        fallback_pool = self.fallback_llms if task_type == "llm" else self.fallback_vlms
        next_model = None
        for cand in fallback_pool:
            if cand != failed_model:
                next_model = cand
                break

        if not next_model:
            next_model = fallback_pool[0] if fallback_pool else failed_model

        self._current_active_model[task_type] = next_model
        logger.warning(
            "Model '%s' hit rate limit! Resetting in %.1fs (at %s). Falling back to '%s'.",
            failed_model,
            reset_secs,
            time.strftime("%H:%M:%S", time.localtime(self._cooldown_until[task_type])),
            next_model,
        )
        return next_model

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        """Generate text completion with rate-limit tracking, fallback, and continuation for complete answers."""
        task_type = "llm"
        active_model = self._get_active_model(task_type, model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call_with_continuation(mdl: str) -> str:
            resp = litellm.completion(
                model=mdl,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)

            # If completion was cut off due to max_tokens, automatically fetch continuation
            if finish_reason == "length" and content.strip():
                logger.warning(
                    "Model '%s' output reached max_tokens limit (%d). Fetching seamless continuation...",
                    mdl,
                    max_tokens,
                )
                continuation_messages = list(messages) + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Please continue and conclude your explanation completely from where you left off.",
                    },
                ]
                try:
                    cont_resp = litellm.completion(
                        model=mdl,
                        messages=continuation_messages,
                        max_tokens=max(1000, max_tokens // 2),
                        temperature=temperature,
                    )
                    cont_text = (cont_resp.choices[0].message.content or "").strip()
                    if cont_text:
                        content = content.rstrip() + "\n\n" + cont_text
                except Exception as cont_err:
                    logger.warning("Continuation request failed (%s); returning initial content.", cont_err)

            return content

        max_retries = 3
        current_model = active_model

        for attempt in range(max_retries + 1):
            try:
                return _call_with_continuation(current_model)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = any(term in err_msg for term in [
                    "rate_limit", "429", "rate limit", "ratelimit", "tokens per minute", "requests per minute", "tpm", "rpm"
                ])
                if is_rate_limit:
                    reset_secs = parse_rate_limit_reset(e, default_seconds=12.0)
                    wait_time = max(2.0, reset_secs + 1.0)
                    if wait_time >= 60.0:
                        logger.warning(
                            "Model '%s' hit rate limit with wait time %.1fs >= 60s. Skipping query as per policy.",
                            current_model,
                            wait_time,
                        )
                        raise RateLimitSkipException(f"Rate limit wait time {wait_time:.1f}s >= 60s")

                    if attempt < max_retries:
                        logger.warning(
                            "Model '%s' hit rate limit (attempt %d/%d). Employing wait methodology: sleeping %.1fs for token bucket reset...",
                            current_model,
                            attempt + 1,
                            max_retries,
                            wait_time,
                        )
                        time.sleep(wait_time)

                        # On later attempts, also rotate to next fallback model if available
                        if attempt >= 1:
                            fallback_model = self._record_rate_limit(task_type, current_model, e)
                            logger.info("Switching to fallback model for retry: %s", fallback_model)
                            current_model = fallback_model
                        continue
                    else:
                        logger.error("Exceeded max retries (%d) on rate limits for model '%s': %s", max_retries, current_model, e)
                        raise
                else:
                    raise

    def caption_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> str:
        """Generate multimodal VLM image description with rate-limit tracking and fallback."""
        task_type = "vlm"
        active_model = self._get_active_model(task_type, model)

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at {p}")

        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        # Determine mime type
        ext = p.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ]

        try:
            resp = litellm.completion(
                model=active_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit" in err_msg or "429" in err_msg or "rate limit" in err_msg:
                reset_secs = parse_rate_limit_reset(e, default_seconds=20.0)
                # If reset window is short (<= 35s), sleep and retry primary VLM directly
                if reset_secs <= 35.0:
                    logger.warning(
                        "Primary VLM '%s' hit rate limit. Waiting %.1fs for token bucket reset...",
                        active_model,
                        reset_secs + 1.0,
                    )
                    time.sleep(reset_secs + 1.0)
                    try:
                        resp = litellm.completion(
                            model=active_model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        return (resp.choices[0].message.content or "").strip()
                    except Exception as retry_err:
                        logger.warning("Retry after wait failed: %s", retry_err)

                fallback_model = self._record_rate_limit(task_type, active_model, e)
                logger.info("Retrying image captioning with fallback VLM: %s", fallback_model)
                try:
                    resp = litellm.completion(
                        model=fallback_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as fb_err:
                    logger.error("Fallback VLM '%s' also failed: %s", fallback_model, fb_err)
                    return ""
            raise
