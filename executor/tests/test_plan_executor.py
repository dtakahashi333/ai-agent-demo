# executor/tests/test_executor.py
from unittest import TestCase

from executor.plan_executor import PlanExecutionStatus, PlanExecutor, StepStatus
from executor.react_executor import ReActExecutionResult
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planning_response import PlannedStep
from state.agent_state import AgentState


class FakeReActExecutor:
    def __init__(
        self,
        success: bool = True,
        response: str = "Done",
    ):
        self.success = success
        self.response = response
        self.objectives = []

    def execute(
        self,
        objective: str,
        state: AgentState,
    ) -> ReActExecutionResult:
        self.objectives.append(objective)

        return ReActExecutionResult(
            success=self.success,
            response=self.response,
        )


class TestGetStepStatus(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Count orders", dependencies=["B"]),
        ]
        self.plan = Plan(self.steps)
        self.react_executor = FakeReActExecutor()

    def test_completed_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.COMPLETED, status)

    def test_failed_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.FAILED, status)

    def test_in_progress_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.IN_PROGRESS, status)

    def test_blocked_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_transitive_blocked_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[2])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_ready_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.READY, status)

    def test_waiting_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.WAITING, status)

    def test_step_with_no_dependencies_is_ready(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.READY, status)

    def test_blocked_dependency_propagation(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(
                id="B", description="Locate customer order records", dependencies=[]
            ),
            PlanStep(
                id="C", description="Retrieve customer profile", dependencies=["A"]
            ),
            PlanStep(
                id="D", description="Retrieve customer orders", dependencies=["B"]
            ),
            PlanStep(
                id="E", description="Create customer summary", dependencies=["C", "D"]
            ),
        ]
        plan = Plan(steps)

        executor = PlanExecutor(
            plan,
            self.react_executor,
        )
        executor.completed_steps.add(steps[0].id)
        executor.failed_steps.add(steps[1].id)

        status = executor.get_step_status(steps[4])

        self.assertEqual(StepStatus.BLOCKED, status)


class TestGetNextReadyStep(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Get customer plan", dependencies=["A"]),
        ]
        self.plan = Plan(self.steps)
        self.react_executor = FakeReActExecutor()

    def test_get_next_ready_step_returns_first_ready_step(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.completed_steps.add(self.steps[0].id)

        step = executor.get_next_ready_step()

        self.assertEqual(self.steps[1], step)

    def test_get_next_ready_step_returns_none_when_no_step_is_ready(self):
        executor = PlanExecutor(
            self.plan,
            self.react_executor,
        )
        executor.failed_steps.add(self.steps[0].id)

        step = executor.get_next_ready_step()

        self.assertIsNone(step)


class TestExecute(TestCase):
    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
        ]
        self.plan = Plan(self.steps)

    def test_execute_runs_ready_step(self):
        react_executor = FakeReActExecutor()

        executor = PlanExecutor(
            self.plan,
            react_executor,
        )

        state = AgentState()

        result = executor.execute(state)

        self.assertEqual(
            ["Find customer"],
            react_executor.objectives,
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

        react_executor = FakeReActExecutor(
            success=False,
            response="Failed",
        )

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=react_executor,
        )

        result = plan_executor.execute(state)

        self.assertEqual(
            result.status,
            PlanExecutionStatus.NEEDS_REPLAN,
        )

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

        react_executor = FakeReActExecutor(
            success=True,
            response="Customer found",
        )

        plan_executor = PlanExecutor(
            plan=plan,
            react_executor=react_executor,
        )

        result = plan_executor.execute(
            state=state,
        )

        self.assertEqual(
            result.response,
            "Customer found",
        )
