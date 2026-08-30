# config/agent_config.py
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Why AgentConfig?
    AgentConfig
        │
        ├── What can this agent do?
        ├── Which model should it use?
        ├── Other agent-level configuration
        │
        ▼
    run_agent()
        │
        ├── Planner
        ├── PlannerLLM
        ├── PlanExecutor
        └── ReActExecutor
    """

    capabilities: list[str]
