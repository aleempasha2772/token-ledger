# llm_tracker/exceptions.py

class BudgetExceededError(Exception):
    """Raised when total cost crosses the configured budget."""
    def __init__(self, budget: float, total_cost: float):
        self.budget     = budget
        self.total_cost = total_cost
        super().__init__(
            f"Budget ${budget:.4f} exceeded — total spent: ${total_cost:.6f}"
        )


class ModelNotIdentifiableError(Exception):
    """Raised for local models that cannot be identified."""
    def __init__(self, model: str):
        super().__init__(
            f"Model '{model}' is not identifiable — cost cannot be computed"
        )