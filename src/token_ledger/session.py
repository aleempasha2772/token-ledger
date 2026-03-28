# llm_tracker/session.py
"""
The Session Manager is the single source of truth for all traces
during the application's lifetime.

Responsibilities:
  - Store every Trace (thread-safely)
  - Maintain the call tree (parent → children)
  - Aggregate totals
  - Enforce budget cap
  - Register atexit + signal handlers for clean exit
"""

import atexit
import json
import logging
import os
import signal
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .trace import Trace
from .exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Thread-safe, singleton-per-application session tracker.
    """

    def __init__(
        self,
        project_name: str        = "llm_project",
        budget:       Optional[float] = None,
        output_dir:   str        = ".",
        print_each:   bool       = True,
        save_on_exit: bool       = True,
    ):
        self.project_name = project_name
        self.budget       = budget
        self.output_dir   = output_dir
        self.print_each   = print_each
        self.save_on_exit = save_on_exit

        self._lock   = threading.Lock()
        self._traces: list[Trace]              = []
        self._tree:   dict[str, list[str]]     = defaultdict(list)  # parent_id → [child_ids]

        # Aggregates (updated atomically under lock)
        self._total_cost:           float             = 0.0
        self._total_input_tokens:   int               = 0
        self._total_output_tokens:  int               = 0
        self._total_embedding_cost: float             = 0.0
        self._total_retry_cost:     float             = 0.0
        self._cost_by_model:        dict[str, float]  = defaultdict(float)

        if save_on_exit:
            atexit.register(self._on_exit)
            # Handle Ctrl-C and kill signals
            try:
                signal.signal(signal.SIGINT,  self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except (OSError, ValueError):
                # Signal handling may not work in threads / notebooks
                pass

    # ── Public interface ──────────────────────────────────────────────────────

    def record(self, trace: Trace) -> None:
        """
        Add a completed Trace to the session.
        Thread-safe. Raises BudgetExceededError if budget is hit.
        """
        with self._lock:
            self._traces.append(trace)

            # Update tree
            if trace.parent_id:
                self._tree[trace.parent_id].append(trace.id)

            # Update aggregates
            self._total_cost           += trace.cost.total_cost
            self._total_input_tokens   += trace.usage.input_tokens
            self._total_output_tokens  += trace.usage.output_tokens
            self._total_embedding_cost += trace.cost.embedding_cost
            self._total_retry_cost     += trace.cost.retry_cost
            self._cost_by_model[trace.model] += trace.cost.total_cost

        # Print per-call log (outside lock to minimize contention)
        if self.print_each:
            self._print_call(trace)

        # Budget check (outside lock — reads are safe here)
        if self.budget is not None and self._total_cost >= self.budget:
            raise BudgetExceededError(
                budget=self.budget,
                total_cost=self._total_cost,
            )

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def cost_by_model(self) -> dict:
        with self._lock:
            return dict(self._cost_by_model)

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "project_name":               self.project_name,
                "total_calls":           len(self._traces),
                "total_input_tokens":    self._total_input_tokens,
                "total_output_tokens":   self._total_output_tokens,
                "total_tokens":          self._total_input_tokens + self._total_output_tokens,
                "total_cost":            round(self._total_cost, 8),
                "embedding_cost":        round(self._total_embedding_cost, 8),
                "retry_cost":            round(self._total_retry_cost, 8),
                "cost_by_model":         dict(self._cost_by_model),
                "budget":                self.budget,
                "budget_remaining":      (
                    round(self.budget - self._total_cost, 8)
                    if self.budget else None
                ),
            }

    def get_traces(self) -> list[Trace]:
        with self._lock:
            return list(self._traces)

    # ── Internal / exit handling ──────────────────────────────────────────────

    def _print_call(self, trace: Trace) -> None:
        prefix = "  " * trace.depth
        call_icon = "E" if trace.call_type == "embedding" else "C"
        estimated = " ~" if trace.usage.is_estimated else ""
        print(
            f"{prefix}[{call_icon}] {trace.name or trace.model} | "
            f"in={trace.usage.input_tokens} "
            f"out={trace.usage.output_tokens} | "
            f"${trace.cost.total_cost:.6f}{estimated}"
        )

    def _on_exit(self) -> None:
        from .exporters import print_summary, write_file
        summary = self.get_summary()
        traces  = self.get_traces()
        print_summary(summary)
        if self.save_on_exit:
            write_file(
                project_name = self.project_name,
                summary      = summary,
                traces       = traces,
                output_dir   = self.output_dir,
            )

    def _signal_handler(self, signum, frame) -> None:
        self._on_exit()
        # Re-raise default handler
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# ── Module-level singleton (the default session) ──────────────────────────────
_default_session: Optional[SessionManager] = None
_session_lock = threading.Lock()


def get_session() -> SessionManager:
    """Return the default session, creating it if needed."""
    global _default_session
    if _default_session is None:
        with _session_lock:
            if _default_session is None:
                _default_session = SessionManager()
    return _default_session


def configure(
    project_name: str        = "llm_project",
    budget:       Optional[float] = None,
    output_dir:   str        = ".",
    print_each:   bool       = True,
    save_on_exit: bool       = True,
) -> SessionManager:
    """
    Configure the global session before any decorators run.
    Call this once at application startup.
    """
    global _default_session
    with _session_lock:
        _default_session = SessionManager(
            project_name = project_name,
            budget       = budget,
            output_dir   = output_dir,
            print_each   = print_each,
            save_on_exit = save_on_exit,
        )
    return _default_session