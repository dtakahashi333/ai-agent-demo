# llm/tests/test_planner_llm.py
from unittest import TestCase

from llm.planner_llm import PlannerLLM
from planner.planning_response import PlanningResponse


class FakeResponses:
    def __init__(self):
        self.model = None
        self.input = None
        self.text_format = None

    def parse(self, *, model, input, text_format):
        self.model = model
        self.input = input
        self.text_format = text_format

        return "fake response"


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()


class TestPlannerLLM(TestCase):

    def test_calls_openai_with_structured_output(self):
        client = FakeOpenAIClient()

        planner_llm = PlannerLLM(
            client=client,
            model="test-model",
        )

        messages = [
            {
                "role": "user",
                "content": "Create a customer summary",
            },
        ]

        response = planner_llm(messages)

        self.assertEqual(
            "test-model",
            client.responses.model,
        )

        self.assertEqual(
            messages,
            client.responses.input,
        )

        self.assertIs(
            PlanningResponse,
            client.responses.text_format,
        )

        self.assertEqual(
            "fake response",
            response,
        )
