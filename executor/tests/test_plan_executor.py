# executor/tests/test_executor.py
from unittest import TestCase
from unittest.mock import Mock

from executor.plan_executor import PlanExecutionStatus, PlanExecutor, StepStatus
from executor.react_executor import ReActExecutionResult, ReActExecutor
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planning_response import PlannedStep
from state.agent_state import AgentState


class TestGetStepStatus(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Count orders", dependencies=["B"]),
        ]
        self.plan = Plan(steps=self.steps)
        self.mock_react_executor = Mock(spec=ReActExecutor)

    def test_completed_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(step=self.steps[0])

        self.assertEqual(StepStatus.COMPLETED, status)

    def test_failed_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.failed_steps[self.steps[0].id] = ""

        status = executor.get_step_status(step=self.steps[0])

        self.assertEqual(StepStatus.FAILED, status)

    def test_in_progress_step(self):
        executor = PlanExecutor(
            self.plan,
            self.mock_react_executor,
        )
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(step=self.steps[0])

        self.assertEqual(StepStatus.IN_PROGRESS, status)

    def test_blocked_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.failed_steps[self.steps[0].id] = ""

        status = executor.get_step_status(step=self.steps[1])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_transitive_blocked_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.failed_steps[self.steps[0].id] = ""

        status = executor.get_step_status(step=self.steps[2])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_ready_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(step=self.steps[1])

        self.assertEqual(StepStatus.READY, status)

    def test_waiting_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(step=self.steps[1])

        self.assertEqual(StepStatus.WAITING, status)

    def test_step_with_no_dependencies_is_ready(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )

        status = executor.get_step_status(step=self.steps[0])

        self.assertEqual(StepStatus.READY, status)

    def test_blocked_dependency_propagation(self):
        steps = [
            PlanStep(
                id="A",
                description="Find customer",
                dependencies=[],
            ),
            PlanStep(
                id="B",
                description="Locate customer order records",
                dependencies=[],
            ),
            PlanStep(
                id="C",
                description="Retrieve customer profile",
                dependencies=["A"],
            ),
            PlanStep(
                id="D",
                description="Retrieve customer orders",
                dependencies=["B"],
            ),
            PlanStep(
                id="E",
                description="Create customer summary",
                dependencies=["C", "D"],
            ),
        ]

        plan = Plan(steps=steps)

        executor = PlanExecutor(
            plan=plan,
            react_executor=self.mock_react_executor,
        )

        executor.completed_steps.add(steps[0].id)
        executor.failed_steps[steps[1].id] = ""

        status = executor.get_step_status(step=steps[4])

        self.assertEqual(StepStatus.BLOCKED, status)


class TestGetNextReadyStep(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Get customer plan", dependencies=["A"]),
        ]
        self.plan = Plan(steps=self.steps)
        self.mock_react_executor = Mock(spec=ReActExecutor)

    def test_get_next_ready_step_returns_first_ready_step(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        step = executor.get_next_ready_step()

        self.assertEqual(self.steps[1], step)

    def test_get_next_ready_step_returns_none_when_no_step_is_ready(self):
        executor = PlanExecutor(
            plan=self.plan,
            react_executor=self.mock_react_executor,
        )
        executor.failed_steps[self.steps[0].id] = ""

        step = executor.get_next_ready_step()

        self.assertIsNone(step)


class TestExecute(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
        ]
        self.plan = Plan(steps=self.steps)

    def test_execute_runs_ready_step(self):
        mock_react_executor = Mock(spec=ReActExecutor)

        executor = PlanExecutor(
            plan=self.plan,
            react_executor=mock_react_executor,
        )

        state = AgentState()

        executor.execute(state=state)

        self.assertEqual(
            [
                call.kwargs["objective"]
                for call in mock_react_executor.execute.call_args_list
            ],
            ["Find customer"],
        )

    def test_dependent_step_is_blocked_after_failure(self):
        state = AgentState()

        plan = Plan(
            steps=[
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
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=False,
            response="Failed",
        )

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=mock_react_executor,
        )

        result = plan_executor.execute(state=state)

        self.assertEqual(
            result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )

        self.assertEqual(result.failed_steps, {"step1": "Failed"})

    def test_returns_final_response(self):
        state = AgentState()

        plan = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=True,
            response="Customer found",
        )

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=mock_react_executor,
        )

        result = plan_executor.execute(state=state)

        self.assertEqual(result.response, "Customer found")

    def test_returns_completed_and_failed_steps(self):
        state = AgentState()

        plan = Plan(
            steps=[
                PlanStep(
                    id="A",
                    description="Find Alice",
                    dependencies=[],
                ),
                PlanStep(
                    id="B",
                    description="Get Alice's orders",
                    dependencies=["A"],
                ),
            ]
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=True, response="Alice found"),
            ReActExecutionResult(success=False, response=""),
        ]

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=mock_react_executor,
        )

        result = plan_executor.execute(state=state)

        self.assertEqual(
            result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )

        self.assertEqual(result.completed_steps, {"A"})

        self.assertEqual(result.failed_steps, {"B": ""})

    def test_does_not_return_intermediate_response_when_replanning(self):
        state = AgentState()

        plan = Plan(
            steps=[
                PlanStep(
                    id="A",
                    description="Find customer",
                    dependencies=[],
                ),
                PlanStep(
                    id="B",
                    description="Get customer orders",
                    dependencies=["A"],
                ),
            ]
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.side_effect = [
            ReActExecutionResult(success=True, response="Customer found"),
            ReActExecutionResult(success=False, response=""),
        ]

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=mock_react_executor,
        )

        result = plan_executor.execute(state=state)

        self.assertEqual(
            result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )

        self.assertIsNone(result.response)

        self.assertEqual(result.completed_steps, {"A"})

        self.assertEqual(result.failed_steps, {"B": ""})

    def test_preserves_failure_reason_for_replanning(self):
        plan = Plan(
            steps=[
                PlanStep(
                    id="step1",
                    description="Find customer",
                    dependencies=[],
                ),
            ]
        )

        mock_react_executor = Mock(spec=ReActExecutor)
        mock_react_executor.execute.return_value = ReActExecutionResult(
            success=False,
            response="Customer service unavailable",
        )

        executor = PlanExecutor(
            plan=plan,
            react_executor=mock_react_executor,
        )

        result = executor.execute(AgentState())

        self.assertEqual(
            result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )

        self.assertIsNone(result.response)

        self.assertEqual(
            result.failed_steps,
            {
                "step1": "Customer service unavailable",
            },
        )
