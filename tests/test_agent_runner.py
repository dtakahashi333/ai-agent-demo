# tests/agent_runner.py
from unittest import TestCase
from unittest.mock import Mock

from agent_runner import AgentRunner
from executor.react_executor import ReActExecutionResult, ReActExecutor
from planner.planner import Planner
from planner.planning_response import PlannedStep
from tests.utils.client_responses import (
    make_planner_client_response,
)


class TestAgentRunner(TestCase):
    def test_returns_response_when_plan_completes(self):
        mock_planner_llm = Mock()
        mock_planner_llm.return_value = make_planner_client_response(
            steps=[
                PlannedStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                )
            ],
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=True,
            response="Alice found",
        )

        runner = AgentRunner(
            planner=Planner(llm_call=mock_planner_llm),
            react_executor=mock_react_executor,
            capabilities=[
                "Find customer by email",
                "Get customer orders",
                "Get customer subscription plan",
                "Create a customer summary",
            ],
            max_replans=1,
        )

        result = runner.run(objective="Find customer")

        self.assertEqual(result, "Alice found")

        self.assertEqual(
            mock_planner_llm.call_args.kwargs["messages"][1]["content"]
            .split("Available capabilities:")[1]
            .strip()
            .splitlines(),
            [
                "- Find customer by email",
                "- Get customer orders",
                "- Get customer subscription plan",
                "- Create a customer summary",
            ],
        )

    def test_replans_after_execution_failure(self):
        mock_planner_llm = Mock()
        mock_planner_llm.side_effect = [
            make_planner_client_response(
                steps=[
                    PlannedStep(
                        id="step1",
                        description="Find customer",
                        dependencies=[],
                    ),
                ],
            ),
            make_planner_client_response(
                steps=[
                    PlannedStep(
                        id="step1",
                        description="Try another way",
                        dependencies=[],
                    ),
                ],
            ),
        ]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(
                success=False,
                response="Customer service unavailable",
            ),
            ReActExecutionResult(
                success=True,
                response="Alice found",
            ),
        ]

        runner = AgentRunner(
            planner=Planner(llm_call=mock_planner_llm),
            react_executor=mock_react_executor,
            capabilities=["Find customer"],
            max_replans=1,
        )

        result = runner.run(objective="Find customer")

        self.assertEqual(result, "Alice found")
        self.assertEqual(mock_planner_llm.call_count, 2)
        self.assertEqual(mock_react_executor.execute.call_count, 2)

    def test_sends_replanning_context(self):
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
            ReActExecutionResult(
                success=False,
                response="Failed",
            ),
            ReActExecutionResult(
                success=True,
                response="Alice found",
            ),
        ]

        runner = AgentRunner(
            planner=Planner(llm_call=mock_planner_llm),
            react_executor=mock_react_executor,
            capabilities=["Find customer"],
            max_replans=1,
        )

        result = runner.run(objective="Find customer")

        self.assertEqual(result, "Alice found")

        second_messages = mock_planner_llm.call_args_list[1].kwargs["messages"]

        self.assertEqual(
            second_messages[-2]["content"],
            "Previous Plan\nstep1: Find customer",
        )

        self.assertEqual(
            second_messages[-1]["content"],
            "Execution Result\nstep1 -> failed: Failed",
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
                    response="Failed",
                )

            self.assertEqual(state.retrieved_count, 3)

            return ReActExecutionResult(
                success=True,
                response="Done",
            )

        mock_react_executor.execute.side_effect = execute

        runner = AgentRunner(
            planner=Planner(llm_call=mock_planner_llm),
            react_executor=mock_react_executor,
            capabilities=["Find customer", "Create customer summary"],
            max_replans=1,
        )

        result = runner.run(objective="Find customer and create summary")

        self.assertEqual(result, "Done")
