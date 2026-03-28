# llm_tracker/__init__.py
"""
llm_tracker — Token usage and cost tracking for LLM applications.

Quick start:
    from llm_tracker import track_llm_cost, track_embedding_cost, configure

    configure(project_name="my_app", budget=5.0)

    @track_llm_cost(model="gpt-4o")
    def generate(messages):
        return openai_client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
"""

from .decorators  import track_llm_cost, track_embedding_cost
from .session     import configure, get_session
from .exceptions  import BudgetExceededError
from .pricing     import get_pricing
from .trace       import Trace, TokenUsage, CostBreakdown

__version__ = "0.1.0"
__all__ = [
    "track_llm_cost",
    "track_embedding_cost",
    "configure",
    "get_session",
    "BudgetExceededError",
    "get_pricing",
    "Trace",
    "TokenUsage",
    "CostBreakdown",
]