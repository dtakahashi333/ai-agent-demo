# tests/utils/llm_responses.py
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletionMessage

from planner.planning_response import PlannedStep, PlanningResponse


def make_planner_client_response(
    steps: list[PlannedStep],
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=PlanningResponse(
                        steps=steps,
                    ),
                )
            )
        ]
    )


def make_react_client_response(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )
