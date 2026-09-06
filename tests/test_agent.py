# tests/test_agent.py
from unittest import TestCase
from unittest.mock import Mock
from agent import run_agent
from tests.utils.client_responses import (
    make_planner_client_response,
    make_react_client_response,
)
from llm.planner_llm import PlannerLLM
from llm.react_llm import ReActLLM
from planner.planning_response import PlannedStep
from tool_registry import build_llm_tools, tool_registry
from config.settings import config

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)


class TestAgent(TestCase):
    def test_runs_successfully(self):
        mock_planner_client = Mock()
        mock_planner_client.chat.completions.parse.return_value = (
            make_planner_client_response(
                steps=[
                    PlannedStep(
                        id="step1",
                        description="Find customer",
                        dependencies=[],
                    ),
                ]
            )
        )

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = (
            make_react_client_response(
                content="Customer found",
                tool_calls=[],
            )
        )

        result = run_agent(
            query="Find customer",
            planner_llm=PlannerLLM(
                client=mock_planner_client,
                model="test-model",
            ),
            react_llm=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
        )

        self.assertEqual(result, "Customer found")
