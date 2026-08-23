# agent_state.py
from dataclasses import dataclass, field

@dataclass
class AgentState:
    messages: list[dict] = field(default_factory=list)

    iteration: int = 0
    retrieved_count: int = 0

    seen_failed_tool_calls: set = field(default_factory=set)