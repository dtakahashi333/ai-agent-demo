# llm/react_llm.py
from typing import Any

from openai.types.chat import ChatCompletion


class ReActLLM:
    def __init__(
        self,
        client: Any,
        model: str,
        tools: list[dict[str, Any]],
    ):
        self.client = client
        self.model = model
        self.tools = tools

    def __call__(
        self,
        messages: list[Any],
        tool_choice=None,
    ) -> ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice=tool_choice,
            # extra_body={"enable_thinking": False},
        )
