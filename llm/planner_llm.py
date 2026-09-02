# llm/planner_llm.py
from typing import Any

from openai.types.chat import ChatCompletion

from planner.planning_response import PlanningResponse


class PlannerLLM:
    def __init__(
        self,
        client: Any,
        model: str,
    ):
        self.client = client
        self.model = model

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> ChatCompletion:
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=PlanningResponse,
        )

        print(response)

        return response
