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
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_client = Mock()
        mock_planner_client.chat.completions.parse.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
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

    def test_replans_after_step_failure(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=True, response="Done"),
        ]

        result = run_agent(
            query="Find customer",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertEqual(result, "Done")

        self.assertEqual(mock_planner_llm.call_count, 2)

        second_messages = mock_planner_llm.call_args_list[1].kwargs["messages"]

        self.assertEqual(
            second_messages[-1]["content"],
            "Execution Result\nstep1 -> failed",
        )
        self.assertEqual(
            second_messages[-2]["content"],
            "Previous Plan\nstep1: Find customer",
        )

    def test_executes_replanned_steps(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=True, response="Done"),
        ]

        result = run_agent(
            query="Find customer",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertEqual(result, "Done")

        self.assertEqual(mock_react_executor.execute.call_count, 2)

        second_execution = mock_react_executor.execute.call_args_list[1]

        self.assertEqual(
            second_execution.kwargs["objective"], "Try finding customer another way"
        )

    def test_reuses_agent_state_when_replanning(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=True, response="Done"),
        ]

        result = run_agent(
            query="Find customer",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertEqual(result, "Done")

        self.assertEqual(mock_react_executor.execute.call_count, 2)

        first_execution = mock_react_executor.execute.call_args_list[0]
        second_execution = mock_react_executor.execute.call_args_list[1]

        self.assertIs(
            first_execution.kwargs["state"],
            second_execution.kwargs["state"],
        )

    def test_stops_after_one_replan(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=False, response=""),
        ]

        run_agent(
            query="Find customer",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertEqual(mock_planner_llm.call_count, 2)
        self.assertEqual(mock_react_executor.execute.call_count, 2)

    def test_returns_response_when_replanning_is_exhausted(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Try finding customer another way",
                dependencies=[],
            ),
        ]
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=False, response=""),
        ]

        result = run_agent(
            query="Find customer",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertIsNone(result)

    def test_preserves_agent_state_when_replanning(self):
        plan1_steps = [
            PlannedStep(
                id="step1",
                description="Find customer",
                dependencies=[],
            ),
        ]
        plan2_steps = [
            PlannedStep(
                id="step1",
                description="Create customer summary",
                dependencies=[],
            ),
        ]

        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(steps=plan1_steps),
            make_planner_client_response(steps=plan2_steps),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)

        def execute(objective, state):
            if mock_react_executor.execute.call_count == 1:
                state.retrieved_count = 3

                return ReActExecutionResult(
                    success=False,
                    response="",
                )

            self.assertEqual(state.retrieved_count, 3)

            return ReActExecutionResult(
                success=True,
                response="Done",
            )

        mock_react_executor.execute.side_effect = execute

        result = run_agent(
            query="Find customer and create summary",
            planner_llm=mock_planner_llm,
            react_executor=mock_react_executor,
        )

        self.assertEqual(result, "Done")
