# llm/planner_llm.py
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

from planner.planning_response import PlanningResponse

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


# def planner_llm(
#     messages: list[dict[str, str]],
# ):
#     return client.responses.parse(
#         model=os.getenv("LLM_MODEL"),
#         input=messages,
#         text_format=PlanningResponse,
#     )


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
