# planner/plan.py
from dataclasses import dataclass

from planner.step import PlanStep

"""
        PLAN
        ↓
        EXECUTE
        ↓
    new information
        ↓
    plan still valid?
        /       \
    yes        no
    ↓          ↓
    continue    REPLAN
"""

"""
Plan
    = data describing intended work

PlanValidator
    = determines whether that plan is structurally valid

Planner
    = generates the plan

Executor
    = executes the plan

"""

"""
                    ┌──────────────┐
User request ──────>│   Planner    │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │     Plan     │
                    └──────┬───────┘
                           │
                           v
                    ┌──────────────┐
                    │PlanValidator │
                    └──────┬───────┘
                           │
                      valid plan
                           │
                           v
                    ┌──────────────┐
                    │   Executor   │
                    └──────┬───────┘
                           │
                           v
                      AgentState
                           │
                           v
                         Tools
"""


@dataclass(frozen=True)
class Plan:
    steps: list[PlanStep]
