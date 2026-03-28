# llm_tracker/pricing.py
"""
Pricing source of truth: LiteLLM's model_cost dictionary.
We try to import it at runtime. If LiteLLM is not installed
or the model is not found, we fall back to a local snapshot.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Local fallback snapshot (most common models) ──────────────────────────────
# Prices in USD per single token (not per million).
# Updated as of early 2026. Multiply by 1,000,000 for $/1M display.
_LOCAL_FALLBACK: dict[str, dict] = {
    # OpenAI
    "gpt-4o":                    {"input": 0.0000025,  "output": 0.000010,   "cache_read": 0.00000125},
    "gpt-4o-mini":               {"input": 0.00000015, "output": 0.0000006,  "cache_read": 0.000000075},
    "o1":                        {"input": 0.000015,   "output": 0.000060,   "cache_read": 0.0000075},
    "o3-mini":                   {"input": 0.0000011,  "output": 0.0000044,  "cache_read": 0.00000055},
    "gpt-3.5-turbo":             {"input": 0.0000005,  "output": 0.0000015},
    "text-embedding-3-small":    {"input": 0.00000002, "output": 0.0},
    "text-embedding-3-large":    {"input": 0.00000013, "output": 0.0},
    # Anthropic
    "claude-3-5-sonnet-20241022":{"input": 0.000003,   "output": 0.000015,   "cache_read": 0.0000003},
    "claude-3-5-haiku-20241022": {"input": 0.0000008,  "output": 0.000004,   "cache_read": 0.00000008},
    "claude-opus-4-6":           {"input": 0.000015,   "output": 0.000075,   "cache_read": 0.0000015},
    # Google
    "gemini-2.0-flash":          {"input": 0.00000010, "output": 0.00000040},
    "gemini-1.5-pro":            {"input": 0.00000125, "output": 0.000005},
    "gemini-1.5-flash":          {"input": 0.000000075,"output": 0.0000003},
    # Mistral
    "mistral-large-latest":      {"input": 0.000002,   "output": 0.000006},
    "mistral-small-latest":      {"input": 0.0000002,  "output": 0.0000006},
    # DeepSeek
    "deepseek-chat":             {"input": 0.00000027, "output": 0.0000011},
    "deepseek-reasoner":         {"input": 0.00000055, "output": 0.00000219},
    # xAI
    "grok-2-latest":             {"input": 0.000002,   "output": 0.000010},
}


@dataclass(frozen=True)
class ModelPricing:
    """
    Immutable pricing record for one model.
    All values are USD per single token.
    """
    input_per_token:     float
    output_per_token:    float
    cache_read_per_token: float = 0.0
    embedding_per_token:  float = 0.0  # same as input for embedding models

    @property
    def input_per_million(self) -> float:
        return self.input_per_token * 1_000_000

    @property
    def output_per_million(self) -> float:
        return self.output_per_token * 1_000_000


class PricingEngine:
    """
    Resolves model name → ModelPricing.

    Resolution order:
    1. LiteLLM model_cost dict (most up-to-date, 1600+ models)
    2. Local fallback snapshot (no network, common models only)
    3. Returns None (caller must handle gracefully)
    """

    def __init__(self):
        self._litellm_prices: Optional[dict] = None
        self._try_load_litellm()

    def _try_load_litellm(self) -> None:
        try:
            import litellm
            self._litellm_prices = litellm.model_cost
            logger.debug("LiteLLM pricing loaded: %d models", len(self._litellm_prices))
        except ImportError:
            logger.debug("LiteLLM not installed — using local fallback pricing only")
        except Exception as e:
            logger.warning("LiteLLM pricing load failed: %s", e)

    def get(self, model: str) -> Optional[ModelPricing]:
        """
        Returns ModelPricing for a model name, or None if unknown.
        Never raises.
        """
        # ── Strategy 1: LiteLLM ───────────────────────────────
        if self._litellm_prices:
            entry = self._litellm_prices.get(model)
            if entry:
                return ModelPricing(
                    input_per_token      = entry.get("input_cost_per_token", 0.0),
                    output_per_token     = entry.get("output_cost_per_token", 0.0),
                    cache_read_per_token = entry.get("cache_read_input_token_cost", 0.0),
                )

        # ── Strategy 2: Local fallback ────────────────────────
        # Normalize: strip date suffixes, lowercase
        normalized = self._normalize_model_name(model)
        entry = _LOCAL_FALLBACK.get(normalized)
        if entry:
            return ModelPricing(
                input_per_token      = entry.get("input", 0.0),
                output_per_token     = entry.get("output", 0.0),
                cache_read_per_token = entry.get("cache_read", 0.0),
            )

        # ── Strategy 3: Unknown ───────────────────────────────
        logger.warning("[llm_tracker] Unknown model '%s' — cost will be 0.0", model)
        return None

    def _normalize_model_name(self, model: str) -> str:
        """
        Maps messy model strings to canonical keys.
        'claude-3-5-sonnet-20241022' → 'claude-3-5-sonnet-20241022' (exact)
        'gpt-4o-2024-08-06'         → 'gpt-4o'
        """
        model = model.strip().lower()

        # Try exact match first
        if model in _LOCAL_FALLBACK:
            return model

        # Strip trailing date patterns like -20241022, -2024-08-06
        import re
        stripped = re.sub(r"-\d{4}-?\d{0,2}-?\d{0,2}$", "", model)
        if stripped in _LOCAL_FALLBACK:
            return stripped

        return model


# ── Module-level singleton ────────────────────────────────────────────────────
_engine = PricingEngine()


def get_pricing(model: str) -> Optional[ModelPricing]:
    """Public interface — get pricing for a model name."""
    return _engine.get(model)