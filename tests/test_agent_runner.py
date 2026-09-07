# tests/agent_runner.py
from unittest import TestCase
from unittest.mock import Mock

from agent_runner import AgentRunner
from executor.plan_executor import PlanExecutionStatus
from executor.react_executor import ReActExecutionResult, ReActExecutor
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planner import Planner
from planner.planning_response import PlannedStep
from tests.utils.client_responses import (
    make_planner_client_response,
)


class TestAgentRunner(TestCase):
    def test_returns_response_when_plan_completes(self):
        mock_planner = Mock(spec=Planner)
        mock_planner.plan.return_value = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                )
            ]
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=True,
            response="Alice found",
        )

        runner = AgentRunner(
            planner=mock_planner,
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

    def test_passes_capabilities_to_planner(self):
        plan = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )

        mock_planner = Mock(spec=Planner)
        mock_planner.plan.return_value = plan

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=True,
            response="Alice found",
        )

        capabilities = ["Find customer", "Get customer orders"]

        runner = AgentRunner(
            planner=mock_planner,
            react_executor=mock_react_executor,
            capabilities=capabilities,
        )

        runner.run(objective="Find customer")

        mock_planner.plan.assert_called_once_with(
            objective="Find customer",
            capabilities=capabilities,
            previous_plan=None,
            execution_result=None,
        )

    def test_replans_after_execution_failure(self):
        plan1 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ],
        )
        plan2 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Try another way",
                    dependencies=[],
                ),
            ],
        )

        mock_planner = Mock(spec=Planner)
        mock_planner.plan.side_effect = [plan1, plan2]

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
            planner=mock_planner,
            react_executor=mock_react_executor,
            capabilities=["Find customer"],
            max_replans=1,
        )

        runner.run(objective="Find customer")

        self.assertEqual(mock_planner.plan.call_count, 2)

    def test_passes_replanning_context_to_planner(self):
        plan1 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )

        plan2 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Try finding customer another way",
                    dependencies=[],
                ),
            ]
        )

        mock_planner = Mock(spec=Planner)
        mock_planner.plan.side_effect = [plan1, plan2]

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
            planner=mock_planner,
            react_executor=mock_react_executor,
            capabilities=["Find customer"],
            max_replans=1,
        )

        result = runner.run(objective="Find customer")

        self.assertEqual(result, "Alice found")

        second_call = mock_planner.plan.call_args_list[1]

        self.assertIs(second_call.kwargs["previous_plan"], plan1)

        execution_result = second_call.kwargs["execution_result"]

        self.assertEqual(
            execution_result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )
        self.assertEqual(
            execution_result.failed_steps,
            {"step1": "Failed"},
        )

    def test_reuses_agent_state_when_replanning(self):
        mock_planner = Mock(spec=Planner)
        mock_planner.plan.side_effect = [
            Plan(
                steps=[
                    PlanStep(
                        id="step1",
                        description="Find customer",
                        dependencies=[],
                    ),
                ]
            ),
            Plan(
                steps=[
                    PlanStep(
                        id="step1",
                        description="Create customer summary",
                        dependencies=[],
                    ),
                ]
            ),
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
            planner=mock_planner,
            react_executor=mock_react_executor,
            capabilities=["Find customer", "Create customer summary"],
            max_replans=1,
        )

        result = runner.run(objective="Find customer and create summary")

        self.assertEqual(result, "Done")

    def test_passes_previous_plan_and_execution_result_when_replanning(self):
        plan1 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )
        plan2 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Try another way",
                    dependencies=[],
                ),
            ]
        )

        mock_planner = Mock(spec=Planner)
        mock_planner.plan.side_effect = [plan1, plan2]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(
                success=False,
                response="Customer service unavailable",
            ),
            ReActExecutionResult(
                success=True,
                response="Customer found",
            ),
        ]

        runner = AgentRunner(
            planner=mock_planner,
            react_executor=mock_react_executor,
            capabilities=["Find customer"],
            max_replans=1,
        )

        runner.run(objective="Find customer")

        first_call = mock_planner.plan.call_args_list[0]
        second_call = mock_planner.plan.call_args_list[1]

        self.assertIsNone(first_call.kwargs["previous_plan"])
        self.assertIsNone(first_call.kwargs["execution_result"])

        self.assertIs(second_call.kwargs["previous_plan"], plan1)

        execution_result = second_call.kwargs["execution_result"]

        self.assertEqual(
            execution_result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )
        self.assertEqual(
            execution_result.failed_steps,
            {"step1": "Customer service unavailable"},
        )

    def test_raises_when_replanning_is_exhausted(self):
        plan1 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )
        plan2 = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Try finding customer another way",
                    dependencies=[],
                ),
            ]
        )

        mock_planner = Mock(spac=Planner)
        mock_planner.plan.side_effect = [plan1, plan2]

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=False, response=""),
            ReActExecutionResult(success=False, response=""),
        ]

        with self.assertRaisesRegex(RuntimeError, "Failed steps"):
            runner = AgentRunner(
                planner=mock_planner,
                react_executor=mock_react_executor,
                capabilities=["Find customer"],
                max_replans=1,
            )
            runner.run(objective="Find customer")
