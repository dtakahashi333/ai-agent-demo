# tests/test_agent.py
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from agent import run_agent
from executor.plan_executor import PlanExecutionStatus, PlanExecutor
from executor.react_executor import ReActExecutionResult, ReActExecutor
from llm.planner_llm import PlannerLLM
from llm.react_llm import ReActLLM
from planner.plan import Plan
from planner.planning_response import PlannedStep
from state.agent_state import AgentState
from tool_registry import build_llm_tools, tool_registry
from config.settings import config

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)


class TestAgent(TestCase):
    def test_composition_of_existing_components(self):
        mock_planner_client = Mock()
        mock_planner_client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=SimpleNamespace(
                            steps=[
                                PlannedStep(
                                    id="step1",
                                    description="Find customer",
                                    dependencies=[],
                                ),
                            ]
                        )
                    )
                )
            ]
        )

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Customer found",
                        tool_calls=[],
                    )
                )
            ]
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
        mock_planner_client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        parsed=SimpleNamespace(
                            steps=steps,
                        )
                    )
                )
            ]
        )

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Done",
                        tool_calls=[],
                    )
                )
            ]
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
