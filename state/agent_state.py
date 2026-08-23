# state/agent_state.py
from dataclasses import dataclass, field

from state.customer import Customer

"""
|State                        | Lifetime      | Location       |
|-----------------------------|---------------|----------------|
|messages                     | entire run    | AgentState     |
|iteration                    | entire run    | AgentState     |
|retrieved_count              | entire run    | AgentState     |
|seen_failed_tool_calls       | entire run    | AgentState     |
|seen_tool_calls_in_iteration | one iteration | local variable |
|approved_calls               | one iteration | local variable |
|results                      | one iteration | local variable |
|executed_calls               | one iteration | local variable |
"""


@dataclass
class AgentState:
    """
    Run-scoped state for a single agent execution.

    Contains:
    - LLM-visible conversation state
    - execution safety state
    - retrieval budget state
    - semantic working state

    This state is intentionally discarded when run_agent()
    finishes. It is not persistent agent memory.
    """

    messages: list[dict] = field(default_factory=list)

    iteration: int = 0
    retrieved_count: int = 0

    seen_failed_tool_calls: set[str] = field(default_factory=set)

    selected_customer: Customer | None = (
        None  # exactly one successfully retrieved customer
    )

    def __post_init__(self):
        if self.iteration < 0:
            raise ValueError("iteration cannot be negative")

        if self.retrieved_count < 0:
            raise ValueError("retrieved_count cannot be negative")

    def select_customer(self, data: dict) -> None:
        if data["id"] <= 0:
            raise ValueError("customer id must be positive")

        self.selected_customer = Customer(
            data["id"],
            data["name"],
            data["email"],
            data["plan"],
        )

    def add_retrieved_results(self, count: int) -> None:
        if count < 0:
            raise ValueError("retrieved result count cannot be negative")

        self.retrieved_count += count

    def record_failed_tool_call(self, signature: str) -> None:
        self.seen_failed_tool_calls.add(signature)

    def increment_iteration(self) -> None:
        self.iteration += 1
