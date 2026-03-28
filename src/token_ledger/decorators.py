# llm_tracker/decorators.py
"""
The decorator layer — the only thing users touch.

@track_llm_cost(model="gpt-4o")
def generate(prompt):
    return openai_client.chat.completions.create(...)

@track_embedding_cost(model="text-embedding-3-small")
def embed(texts):
    return openai_client.embeddings.create(...)
"""

import asyncio
import functools
import inspect
import logging
import time
from typing import Callable, Optional

from .context   import TraceContext, new_trace_ids
from .trace     import Trace, TokenUsage, CostBreakdown
from .providers import extract_usage
from .tokenizers import count_tokens
from .pricing    import get_pricing
from .calculator import calculate
from .session    import get_session

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# @track_llm_cost
# ══════════════════════════════════════════════════════════════════════════════

def track_llm_cost(
    model:       str,
    name:        str           = "",
    model_param: str           = "model",    # kwarg name to read model from
    on_complete: Optional[Callable] = None,  # callback(trace)
    session     = None,                      # use custom session
):
    """
    Decorator for LLM completion calls.

    @track_llm_cost(model="gpt-4o")
    def call_llm(messages):
        return client.chat.completions.create(model="gpt-4o", messages=messages)
    """
    def decorator(fn: Callable) -> Callable:
        resolved_session = session or get_session()
        display_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return await _run_tracked(
                    fn, args, kwargs,
                    model=_resolve_model(kwargs, model_param, model),
                    name=display_name,
                    call_type="completion",
                    session=resolved_session,
                    on_complete=on_complete,
                    is_async=True,
                )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                return _run_sync(
                    fn, args, kwargs,
                    model=_resolve_model(kwargs, model_param, model),
                    name=display_name,
                    call_type="completion",
                    session=resolved_session,
                    on_complete=on_complete,
                )
            return sync_wrapper

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# @track_embedding_cost
# ══════════════════════════════════════════════════════════════════════════════

def track_embedding_cost(
    model:       str,
    name:        str           = "",
    input_param: str           = "input",    # kwarg/arg name for the text input
    on_complete: Optional[Callable] = None,
    session      = None,
):
    """
    Decorator for embedding calls.

    @track_embedding_cost(model="text-embedding-3-small")
    def embed(texts):
        return client.embeddings.create(model="text-embedding-3-small", input=texts)
    """
    def decorator(fn: Callable) -> Callable:
        resolved_session = session or get_session()
        display_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                input_text = _extract_input(args, kwargs, input_param, fn)
                return await _run_tracked(
                    fn, args, kwargs,
                    model=model,
                    name=display_name,
                    call_type="embedding",
                    session=resolved_session,
                    on_complete=on_complete,
                    is_async=True,
                    input_text=input_text,
                )
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                input_text = _extract_input(args, kwargs, input_param, fn)
                return _run_sync(
                    fn, args, kwargs,
                    model=model,
                    name=display_name,
                    call_type="embedding",
                    session=resolved_session,
                    on_complete=on_complete,
                    input_text=input_text,
                )
            return sync_wrapper

    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# Core execution logic
# ══════════════════════════════════════════════════════════════════════════════

def _run_sync(fn, args, kwargs, *, model, name, call_type, session, on_complete, input_text=None):
    trace_id, root_id, parent_id, depth = new_trace_ids()
    t_start  = time.perf_counter()
    status   = "success"
    error_type = ""
    response = None

    with TraceContext(trace_id=trace_id, root_id=root_id, depth=depth + 1):
        try:
            response = fn(*args, **kwargs)
            return response
        except Exception as exc:
            status     = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - t_start) * 1000
            trace = _build_trace(
                trace_id=trace_id, root_id=root_id, parent_id=parent_id,
                depth=depth, name=name, model=model, call_type=call_type,
                response=response, input_text=input_text,
                latency_ms=latency_ms, status=status, error_type=error_type,
            )
            _finalize(trace, session, on_complete)


async def _run_tracked(fn, args, kwargs, *, model, name, call_type, session,
                        on_complete, is_async, input_text=None):
    trace_id, root_id, parent_id, depth = new_trace_ids()
    t_start    = time.perf_counter()
    status     = "success"
    error_type = ""
    response   = None

    async with TraceContext(trace_id=trace_id, root_id=root_id, depth=depth + 1):
        try:
            response = await fn(*args, **kwargs)
            return response
        except Exception as exc:
            status     = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            latency_ms = (time.perf_counter() - t_start) * 1000
            trace = _build_trace(
                trace_id=trace_id, root_id=root_id, parent_id=parent_id,
                depth=depth, name=name, model=model, call_type=call_type,
                response=response, input_text=input_text,
                latency_ms=latency_ms, status=status, error_type=error_type,
            )
            _finalize(trace, session, on_complete)


def _build_trace(
    *, trace_id, root_id, parent_id, depth,
    name, model, call_type, response, input_text,
    latency_ms, status, error_type,
) -> Trace:
    """
    Build a complete Trace from a finished call.
    Tries provider extraction first, falls back to tokenizer.
    """
    # ── Step 1: extract token usage ───────────────────────────
    usage = None
    if response is not None:
        usage = extract_usage(response, provider_hint=model)

    if usage is None:
        # Tokenizer fallback
        if call_type == "embedding" and input_text is not None:
            text = input_text if isinstance(input_text, str) else " ".join(input_text)
            n, estimated = count_tokens(text, model)
            usage = TokenUsage(embedding_tokens=n, is_estimated=estimated)
        else:
            # No response and no input text — zero cost, mark estimated
            usage = TokenUsage(is_estimated=True)

    # ── Step 2: get pricing ───────────────────────────────────
    pricing = get_pricing(model)

    # ── Step 3: calculate cost ────────────────────────────────
    cost = calculate(usage=usage, pricing=pricing, call_type=call_type)

    return Trace(
        id         = trace_id,
        trace_id   = root_id,
        parent_id  = parent_id,
        depth      = depth,
        name       = name,
        model      = model,
        call_type  = call_type,
        latency_ms = latency_ms,
        usage      = usage,
        cost       = cost,
        status     = status,
        error_type = error_type,
    )


def _finalize(trace: Trace, session, on_complete) -> None:
    """Record to session and fire callback. Never raises."""
    try:
        session.record(trace)
    except Exception as exc:
        # BudgetExceededError should propagate; others should not
        from .exceptions import BudgetExceededError
        if isinstance(exc, BudgetExceededError):
            raise
        logger.warning("[llm_tracker] Session record failed: %s", exc)

    if on_complete:
        try:
            on_complete(trace)
        except Exception as e:
            logger.warning("[llm_tracker] on_complete callback failed: %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_model(kwargs: dict, param_name: str, default: str) -> str:
    return str(kwargs.get(param_name, default))


def _extract_input(args, kwargs, input_param: str, fn: Callable):
    """Extract the input text/list from function arguments."""
    # Try kwargs first
    if input_param in kwargs:
        return kwargs[input_param]
    # Try positional args using function signature
    try:
        sig    = inspect.signature(fn)
        params = list(sig.parameters.keys())
        if input_param in params:
            idx = params.index(input_param)
            if idx < len(args):
                return args[idx]
    except Exception:
        pass
    return None