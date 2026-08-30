# planner/planned_step.py
from dataclasses import Field

from openai import BaseModel

"""
{
  "steps": [
    {
      "id": "A",
      "description": "Find customer",
      "dependencies": []
    },
    {
      "id": "B",
      "description": "Retrieve customer orders",
      "dependencies": ["A"]
    },
    {
      "id": "C",
      "description": "Retrieve customer plan",
      "dependencies": ["A"]
    },
    {
      "id": "D",
      "description": "Create customer summary",
      "dependencies": ["B", "C"]
    }
  ]
}
"""


class PlannedStep(BaseModel):
    id: str
    description: str
    dependencies: list[str]


class PlanningResponse(BaseModel):
    steps: list[PlannedStep]
