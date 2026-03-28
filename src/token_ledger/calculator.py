# llm_tracker/calculator.py
"""
Pure function: given TokenUsage + ModelPricing, compute CostBreakdown.
No side effects. No external calls. Fully testable in isolation.

Production formula:
  Total = (input - cached) × P_in
        + cached            × P_cache
        + (output + reasoning) × P_out
        + embedding         × P_emb
        + retry_cost        (passed in from caller)
"""

from .trace import TokenUsage, CostBreakdown
from .pricing import ModelPricing
from typing import Optional


def calculate(
    usage:        TokenUsage,
    pricing:      Optional[ModelPricing],
    retry_cost:   float = 0.0,
    call_type:    str   = "completion",
) -> CostBreakdown:
    """
    Compute the full cost breakdown for one LLM call.

    Args:
        usage:      Token counts extracted from response (or estimated)
        pricing:    Prices for this model (None = unknown model, cost = 0)
        retry_cost: Sum of costs from previous failed attempts
        call_type:  "completion" or "embedding"

    Returns:
        CostBreakdown with all dimensions filled
    """
    if pricing is None:
        # Unknown model — return zero costs but preserve retry cost
        return CostBreakdown(retry_cost=retry_cost)

    if call_type == "embedding":
        # Embedding calls: only input tokens, billed at input rate
        emb_cost = usage.embedding_tokens * pricing.input_per_token
        return CostBreakdown(
            embedding_cost = emb_cost,
            retry_cost     = retry_cost,
        )

    # ── Completion call ────────────────────────────────────────────────────
    # Input tokens split into: regular input vs cache-read input
    regular_input = max(0, usage.input_tokens - usage.cached_tokens)
    input_cost    = regular_input     * pricing.input_per_token
    cached_cost   = usage.cached_tokens * pricing.cache_read_per_token

    # Output includes reasoning tokens (billed at output rate for o1/o3)
    total_output  = usage.output_tokens + usage.reasoning_tokens
    output_cost   = total_output        * pricing.output_per_token

    # Reasoning tokens get their own line in the breakdown
    if usage.reasoning_tokens > 0:
        reasoning_cost = usage.reasoning_tokens * pricing.output_per_token
        output_cost    = usage.output_tokens    * pricing.output_per_token
    else:
        reasoning_cost = 0.0

    return CostBreakdown(
        input_cost     = input_cost,
        output_cost    = output_cost,
        cached_cost    = cached_cost,
        reasoning_cost = reasoning_cost,
        retry_cost     = retry_cost,
    )