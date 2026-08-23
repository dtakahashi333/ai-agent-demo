# agent_state.py
from dataclasses import dataclass, field

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
    messages: list[dict] = field(default_factory=list)

    iteration: int = 0
    retrieved_count: int = 0

    seen_failed_tool_calls: set[str] = field(default_factory=set)

    selected_customer: dict | None = None  # exactly one successfully retrieved customer
