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

    Conversation state:
    - messages

    Execution state:
    - iteration
    - seen_failed_tool_calls
    - retrieved_count

    Semantic working state:
    - retrieved_customer

    This state is intentionally discarded when run_agent()
    finishes. It is not persistent agent memory.
    """

    messages: list[dict] = field(default_factory=list)

    iteration: int = 0
    retrieved_count: int = 0

    seen_failed_tool_calls: set[str] = field(default_factory=set)

    """
                   Plan
                    │
            intended workflow
                    │
                    ↓
                PlanExecutor
                    │
                    ↓
                AgentState
            ┌───────┴────────┐
            │                │
    completed steps      semantic facts
            │                │
            └───────┬────────┘
                    ↓
                Replan
                    ↓
                New Plan
    """

    # semantic state
    retrieved_customer: Customer | None = (
        None  # exactly one successfully retrieved customer
    )

    def __post_init__(self):
        if self.iteration < 0:
            raise ValueError("iteration cannot be negative")

        if self.retrieved_count < 0:
            raise ValueError("retrieved_count cannot be negative")

    """
    Every mutation of AgentState happens through an AgentState method.
    """

    def select_customer(self, data: dict) -> None:
        self.retrieved_customer = Customer(
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

    def add_messages(self, messages: list[dict]) -> None:
        """
        Continue the existing conversation.
        """
        self.messages.extend(messages)

    def initialize_messages(
        self,
        system_prompt: str,
        query: str,
    ) -> None:
        """
        Start the state for a new agent run.
        """
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

    def has_selected_customer(self) -> bool:
        return self.retrieved_customer is not None
