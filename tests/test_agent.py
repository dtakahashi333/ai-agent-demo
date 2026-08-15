# tests/test_agent.py

import json
from types import SimpleNamespace
from unittest import TestCase
from agent import execute_tool_call


class TestAgent(TestCase):
    def test_execute_tool_call1(self):
        tool_calls = [
            SimpleNamespace(
                id="call_d59653a0ff764ad3a39573",
                function=SimpleNamespace(
                    arguments=json.dumps({"city": "Dallas"}),
                    name="get_weather",
                ),
                type="function",
                index=0,
            )
        ]

        result = execute_tool_call(tool_calls[0])

        print(result)
