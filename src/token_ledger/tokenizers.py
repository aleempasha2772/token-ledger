# llm_tracker/tokenizers.py
"""
Token counting with priority fallback chain:

1. Provider-returned usage (ground truth — handled in adapters, not here)
2. tiktoken       — for OpenAI and compatible models
3. LiteLLM        — token_counter supports many providers
4. HuggingFace    — AutoTokenizer for local/HF models
5. Char heuristic — last resort, is_estimated=True

This module is called ONLY when the provider did not return
usage counts in the response.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model → tokenizer family mapping ─────────────────────────────────────────
_TIKTOKEN_MODELS = {
    # OpenAI family uses cl100k_base or o200k_base
    "gpt-4o", "gpt-4o-mini", "o1", "o3-mini",
    "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
    "text-embedding-3-small", "text-embedding-3-large",
    # xAI Grok uses a similar vocabulary
    "grok-2-latest",
}

_TIKTOKEN_ENCODING_MAP = {
    "gpt-4o":         "o200k_base",
    "gpt-4o-mini":    "o200k_base",
    "o1":             "o200k_base",
    "o3-mini":        "o200k_base",
    "gpt-4":          "cl100k_base",
    "gpt-4-turbo":    "cl100k_base",
    "gpt-3.5-turbo":  "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

_DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


def count_tokens(text: str, model: str) -> tuple[int, bool]:
    """
    Count tokens in text for the given model.

    Returns:
        (token_count, is_estimated)
        is_estimated=False means we used the accurate tokenizer.
        is_estimated=True  means we used a fallback heuristic.
    """
    if not text:
        return 0, False

    model_lower = model.lower()

    # ── Strategy 1: tiktoken ──────────────────────────────────
    count = _try_tiktoken(text, model_lower)
    if count is not None:
        return count, False

    # ── Strategy 2: LiteLLM token_counter ────────────────────
    count = _try_litellm(text, model)
    if count is not None:
        return count, False

    # ── Strategy 3: HuggingFace AutoTokenizer ─────────────────
    count = _try_huggingface(text, model)
    if count is not None:
        return count, True   # HF is approximate for non-HF models

    # ── Strategy 4: Character heuristic ───────────────────────
    count = _heuristic(text)
    return count, True


def count_tokens_list(texts: list[str], model: str) -> tuple[int, bool]:
    """Count tokens across a list of texts (for embedding batches)."""
    total = 0
    any_estimated = False
    for text in texts:
        n, estimated = count_tokens(text, model)
        total += n
        if estimated:
            any_estimated = True
    return total, any_estimated


# ── Strategy implementations ─────────────────────────────────────────────────

def _try_tiktoken(text: str, model: str) -> Optional[int]:
    try:
        import tiktoken

        # Find the right encoding
        encoding_name = None
        for key in _TIKTOKEN_ENCODING_MAP:
            if model.startswith(key):
                encoding_name = _TIKTOKEN_ENCODING_MAP[key]
                break

        if encoding_name is None:
            # Try tiktoken's own model lookup
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                # Model not in tiktoken's registry
                # If it looks like an OpenAI model, use default
                if any(model.startswith(p) for p in ("gpt-", "o1", "o3", "text-")):
                    enc = tiktoken.get_encoding(_DEFAULT_TIKTOKEN_ENCODING)
                else:
                    return None
        else:
            enc = tiktoken.get_encoding(encoding_name)

        return len(enc.encode(text))

    except ImportError:
        return None
    except Exception as e:
        logger.debug("tiktoken failed for model %s: %s", model, e)
        return None


def _try_litellm(text: str, model: str) -> Optional[int]:
    try:
        import litellm
        return litellm.token_counter(model=model, text=text)
    except ImportError:
        return None
    except Exception as e:
        logger.debug("LiteLLM token_counter failed for model %s: %s", model, e)
        return None


def _try_huggingface(text: str, model: str) -> Optional[int]:
    """
    Attempt HuggingFace AutoTokenizer.
    Only works if the model name matches a HuggingFace repo (e.g. local models).
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokens = tokenizer.encode(text)
        return len(tokens)
    except ImportError:
        return None
    except Exception:
        return None


def _heuristic(text: str) -> int:
    """
    Last-resort estimation.
    ~4 characters per token for English.
    ~2 characters per token for CJK scripts.
    """
    cjk_count = sum(
        1 for ch in text
        if '\u4e00' <= ch <= '\u9fff'
        or '\u3400' <= ch <= '\u4dbf'
        or '\uac00' <= ch <= '\ud7af'
    )
    cjk_ratio = cjk_count / max(len(text), 1)
    chars_per_token = 2.0 if cjk_ratio > 0.3 else 4.0
    return max(1, int(len(text) / chars_per_token))