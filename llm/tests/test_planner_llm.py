# llm/tests/test_planner_llm.py
from unittest import TestCase
from unittest.mock import Mock

from llm.planner_llm import PlannerLLM
from planner.planning_response import PlanningResponse

mock_planner_client = Mock()
mock_planner_client.chat.completions.parse.return_value = "fake response"


class TestPlannerLLM(TestCase):
    def test_calls_openai_with_structured_output(self):
        planner_llm = PlannerLLM(
            client=mock_planner_client,
            model="test-model",
        )

        messages = [
            {
                "role": "user",
                "content": "Create a customer summary",
            },
        ]

        response = planner_llm(messages)

        call = mock_planner_client.chat.completions.parse.call_args

        self.assertEqual(
            call.kwargs["model"],
            "test-model",
        )

        self.assertEqual(
            call.kwargs["messages"],
            messages,
        )

        self.assertIs(
            call.kwargs["response_format"],
            PlanningResponse,
        )

        self.assertEqual(
            "fake response",
            response,
        )
