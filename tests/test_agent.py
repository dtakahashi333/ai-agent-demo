# tests/test_agent.py
from unittest import TestCase
from unittest.mock import Mock
from agent import run_agent
from executor.react_executor import ReActExecutionResult, ReActExecutor
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
    def test_composition_of_existing_components(self):
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

        messages = mock_react_client.chat.completions.create.call_args.kwargs[
            "messages"
        ]

        self.assertEqual(messages[-2]["content"], "Find customer")

    def test_two_dependent_planned_steps(self):
        steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
            PlannedStep(
                id="step2",
                description="Get customer orders",
                dependencies=["step1"],
            ),
        ]

        mock_planner_client = Mock()
        mock_planner_client.chat.completions.parse.return_value = (
            make_planner_client_response(
                steps=steps,
            )
        )

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = (
            make_react_client_response(
                content="Done",
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

        self.assertEqual(result, "Done")

        calls = mock_react_client.chat.completions.create.call_args_list

        executed_objectives = [call.kwargs["messages"][-2]["content"] for call in calls]

        self.assertEqual(
            executed_objectives,
            [
                "Find customer",
                "Get customer orders",
            ],
        )

    def test_replans_after_plan_execution_failure(self):
        plan1 = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2 = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_client = Mock()
        mock_planner_client.chat.completions.parse.side_effect = [
            make_planner_client_response(steps=plan1),
            make_planner_client_response(steps=plan2),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=False,
            response="Failed",
        )

        run_agent(
            query="Find customer",
            planner_llm=PlannerLLM(
                client=mock_planner_client,
                model="test-model",
            ),
            react_executor=mock_react_executor,
        )

        self.assertEqual(
            mock_planner_client.chat.completions.parse.call_count,
            2,
        )
