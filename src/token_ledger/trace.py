# llm_tracker/trace.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class TokenUsage:
    """
    Holds all token count dimensions for a single LLM call.
    is_estimated = True means we used a fallback tokenizer,
    not the provider's ground-truth count.
    """
    input_tokens:      int   = 0
    output_tokens:     int   = 0
    cached_tokens:     int   = 0   # prompt cache read hits
    reasoning_tokens:  int   = 0   # hidden thinking tokens (o1, etc.)
    embedding_tokens:  int   = 0   # only for embedding calls
    is_estimated:      bool  = False

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.embedding_tokens
            # cached_tokens are a subset of input, not additive
        )


@dataclass
class CostBreakdown:
    """
    Full cost breakdown matching the production formula:
    Total = input + output + cached + reasoning + embedding + retry
    """
    input_cost:     float = 0.0
    output_cost:    float = 0.0
    cached_cost:    float = 0.0
    reasoning_cost: float = 0.0
    embedding_cost: float = 0.0
    retry_cost:     float = 0.0  # sum of all failed attempt costs

    @property
    def total_cost(self) -> float:
        return (
            self.input_cost
            + self.output_cost
            + self.cached_cost
            + self.reasoning_cost
            + self.embedding_cost
            + self.retry_cost
        )


@dataclass
class Trace:
    """
    One Trace = one LLM call (or embedding call).
    Traces link to each other via parent_id to form
    the agent call tree.
    """
    # ── Identity ──────────────────────────────────────────────
    id:         str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:   str = ""          # root of the call tree
    parent_id:  Optional[str] = None
    depth:      int = 0
    name:       str = ""          # human label e.g. "planner", "summarizer"

    # ── Call metadata ─────────────────────────────────────────
    model:      str = ""
    call_type:  str = "completion"   # "completion" | "embedding"
    timestamp:  str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = 0.0

    # ── Token usage ───────────────────────────────────────────
    usage:      TokenUsage    = field(default_factory=TokenUsage)

    # ── Cost ──────────────────────────────────────────────────
    cost:       CostBreakdown = field(default_factory=CostBreakdown)

    # ── Reliability ───────────────────────────────────────────
    status:          str = "success"   # "success" | "failed" | "partial"
    retry_count:     int = 0
    failed_attempts: int = 0
    error_type:      str = ""

    def to_dict(self) -> dict:
        """Serialize for file output and logging."""
        return {
            "id":               self.id,
            "trace_id":         self.trace_id,
            "parent_id":        self.parent_id,
            "depth":            self.depth,
            "name":             self.name,
            "model":            self.model,
            "call_type":        self.call_type,
            "timestamp":        self.timestamp,
            "latency_ms":       round(self.latency_ms, 2),
            "input_tokens":     self.usage.input_tokens,
            "output_tokens":    self.usage.output_tokens,
            "cached_tokens":    self.usage.cached_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "embedding_tokens": self.usage.embedding_tokens,
            "total_tokens":     self.usage.total_tokens,
            "is_estimated":     self.usage.is_estimated,
            "input_cost":       round(self.cost.input_cost,     8),
            "output_cost":      round(self.cost.output_cost,    8),
            "cached_cost":      round(self.cost.cached_cost,    8),
            "reasoning_cost":   round(self.cost.reasoning_cost, 8),
            "embedding_cost":   round(self.cost.embedding_cost, 8),
            "retry_cost":       round(self.cost.retry_cost,     8),
            "total_cost":       round(self.cost.total_cost,     8),
            "status":           self.status,
            "retry_count":      self.retry_count,
            "failed_attempts":  self.failed_attempts,
            "error_type":       self.error_type,
        }