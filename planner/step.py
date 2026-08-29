# planner/step.py
from dataclasses import dataclass, field

"""
             Your application
                  │
                  │ tells LLM:
                  │
                  │ "Return a plan with
                  │  steps containing
                  │  id, description,
                  │  dependencies"
                  ↓
              ┌───────┐
User request →│  LLM  │
              └───┬───┘
                  │
                  │ structured output
                  ↓
          ┌──────────────┐
          │     Plan     │
          │              │
          │ PlanStep     │
          │ PlanStep     │
          │ PlanStep     │
          └──────────────┘
                  │
                  ↓
          PlanValidator
"""


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    # state: str  # not started, in progress, completed, blocked
