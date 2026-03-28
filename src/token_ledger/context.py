# llm_tracker/context.py
"""
Uses Python's ContextVar — works correctly in both:
  - asyncio (each coroutine gets its own context)
  - threading (each thread gets its own context)

This is how we track parent → child relationships
in agent systems without any global mutable state.
"""

from contextvars import ContextVar
from typing import Optional
import uuid

# The currently active trace ID.
# When a decorated function calls another decorated function,
# the inner one sees the outer one's ID as its parent.
_current_trace_id:  ContextVar[Optional[str]] = ContextVar(
    "_current_trace_id", default=None
)
_current_root_id:   ContextVar[Optional[str]] = ContextVar(
    "_current_root_id", default=None
)
_current_depth:     ContextVar[int] = ContextVar(
    "_current_depth", default=0
)


class TraceContext:
    """
    Context manager that sets the current trace on entry
    and restores the previous state on exit.

    Usage:
        with TraceContext(trace_id="abc", root_id="root", depth=1):
            # inner calls see "abc" as their parent
            ...
    """

    def __init__(self, trace_id: str, root_id: str, depth: int):
        self.trace_id = trace_id
        self.root_id  = root_id
        self.depth    = depth
        self._tokens  = []   # stores reset tokens for ContextVar.reset()

    def __enter__(self):
        self._tokens.append(_current_trace_id.set(self.trace_id))
        self._tokens.append(_current_root_id.set(self.root_id))
        self._tokens.append(_current_depth.set(self.depth))
        return self

    def __exit__(self, *_):
        # Restore previous values in reverse order
        for token in reversed(self._tokens):
            token.var.reset(token)
        self._tokens.clear()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args):
        return self.__exit__(*args)


def get_parent_id() -> Optional[str]:
    """Returns the currently active trace ID (becomes parent of new calls)."""
    return _current_trace_id.get()


def get_root_id() -> Optional[str]:
    """Returns the root trace ID for the entire request tree."""
    return _current_root_id.get()


def get_depth() -> int:
    """Returns current nesting depth (0 = root call)."""
    return _current_depth.get()


def new_trace_ids() -> tuple[str, str, str, int]:
    """
    Create IDs for a new trace.
    Returns: (new_id, root_id, parent_id, depth)
    """
    parent_id = get_parent_id()
    root_id   = get_root_id()
    depth     = get_depth()

    new_id = str(uuid.uuid4())

    # If there's no existing root, this call IS the root
    if root_id is None:
        root_id = new_id

    return new_id, root_id, parent_id, depth