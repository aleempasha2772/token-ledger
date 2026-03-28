# llm_tracker/providers/__init__.py
"""
Each adapter answers one question:
  Given this raw API response object, what are the token counts?

Returns TokenUsage or None.
None means: caller should fall back to tokenizer estimation.
"""

from ..trace import TokenUsage
from typing import Optional


def extract_usage(response, provider_hint: str = "") -> Optional[TokenUsage]:
    """
    Try each adapter in priority order.
    provider_hint speeds up selection but is not required.
    """
    adapters = _get_adapters(provider_hint)
    for adapter in adapters:
        result = adapter(response)
        if result is not None:
            return result
    return None


def _get_adapters(hint: str):
    """Order adapters by provider hint for faster resolution."""
    all_adapters = [
        _openai_adapter,
        _anthropic_adapter,
        _gemini_adapter,
        _litellm_adapter,
        _ollama_adapter,
        _generic_dict_adapter,
    ]

    hint = hint.lower()
    if "openai" in hint or "gpt" in hint or "o1" in hint:
        ordered = [_openai_adapter] + [a for a in all_adapters if a != _openai_adapter]
    elif "anthropic" in hint or "claude" in hint:
        ordered = [_anthropic_adapter] + [a for a in all_adapters if a != _anthropic_adapter]
    elif "gemini" in hint or "google" in hint:
        ordered = [_gemini_adapter] + [a for a in all_adapters if a != _gemini_adapter]
    else:
        ordered = all_adapters

    return ordered


# ── Individual adapters ───────────────────────────────────────────────────────

def _openai_adapter(response) -> Optional[TokenUsage]:
    """
    OpenAI response: response.usage.prompt_tokens / completion_tokens
    Also handles cached tokens and reasoning tokens (o1 family).
    """
    try:
        usage = response.usage
        if usage is None:
            return None

        cached    = 0
        reasoning = 0

        # Cached tokens (available when prompt caching is active)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        # Reasoning tokens (o1, o3 family)
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        return TokenUsage(
            input_tokens     = (usage.prompt_tokens or 0) - cached,
            output_tokens    = (usage.completion_tokens or 0) - reasoning,
            cached_tokens    = cached,
            reasoning_tokens = reasoning,
        )
    except AttributeError:
        return None


def _anthropic_adapter(response) -> Optional[TokenUsage]:
    """
    Anthropic response: response.usage.input_tokens / output_tokens
    Also handles cache_creation_input_tokens and cache_read_input_tokens.
    """
    try:
        usage = response.usage
        if usage is None:
            return None

        cache_read  = getattr(usage, "cache_read_input_tokens",    0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return TokenUsage(
            input_tokens  = (usage.input_tokens  or 0),
            output_tokens = (usage.output_tokens or 0),
            cached_tokens = cache_read,
            # cache_write tokens are billed at 1.25× input price
            # we store them as overhead in the trace separately if needed
        )
    except AttributeError:
        return None


def _gemini_adapter(response) -> Optional[TokenUsage]:
    """
    Google Gemini response: response.usage_metadata
    """
    try:
        meta = response.usage_metadata
        if meta is None:
            return None
        return TokenUsage(
            input_tokens  = getattr(meta, "prompt_token_count",     0) or 0,
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0,
        )
    except AttributeError:
        return None


def _litellm_adapter(response) -> Optional[TokenUsage]:
    """
    LiteLLM normalizes all responses to OpenAI format.
    This catches any provider routed through LiteLLM.
    """
    try:
        # LiteLLM responses look like OpenAI responses
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        pt = getattr(usage, "prompt_tokens",     None)
        ct = getattr(usage, "completion_tokens", None)
        if pt is None and ct is None:
            return None
        return TokenUsage(
            input_tokens  = pt or 0,
            output_tokens = ct or 0,
        )
    except Exception:
        return None


def _ollama_adapter(response) -> Optional[TokenUsage]:
    """
    Ollama dict response format:
    {"prompt_eval_count": N, "eval_count": M}
    """
    try:
        if not isinstance(response, dict):
            return None
        if "eval_count" not in response and "prompt_eval_count" not in response:
            return None
        return TokenUsage(
            input_tokens  = response.get("prompt_eval_count", 0) or 0,
            output_tokens = response.get("eval_count",        0) or 0,
        )
    except Exception:
        return None


def _generic_dict_adapter(response) -> Optional[TokenUsage]:
    """
    Catch-all for dict responses with a 'usage' key.
    Handles vLLM and other OpenAI-compatible servers.
    """
    try:
        if not isinstance(response, dict):
            return None
        usage = response.get("usage", {})
        if not usage:
            return None
        pt = usage.get("prompt_tokens") or usage.get("input_tokens")
        ct = usage.get("completion_tokens") or usage.get("output_tokens")
        if pt is None and ct is None:
            return None
        return TokenUsage(
            input_tokens  = pt or 0,
            output_tokens = ct or 0,
        )
    except Exception:
        return None