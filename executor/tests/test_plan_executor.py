# executor/tests/test_executor.py

from unittest import TestCase

from executor.plan_executor import PlanExecutor, StepStatus
from planner.plan import Plan
from planner.step import PlanStep


class TestGetStepStatus(TestCase):

    def setUp(self):
        super().setUp()
        self.steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Count orders", dependencies=["B"]),
        ]
        self.plan = Plan(self.steps)

    def test_completed_step(self):
        executor = PlanExecutor(self.plan)
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.COMPLETED, status)

    def test_failed_step(self):
        executor = PlanExecutor(self.plan)
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.FAILED, status)

    def test_in_progress_step(self):
        executor = PlanExecutor(self.plan)
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(self.steps[0])

        self.assertEqual(StepStatus.IN_PROGRESS, status)

    def test_blocked_step(self):
        executor = PlanExecutor(self.plan)
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_transitive_blocked_step(self):
        executor = PlanExecutor(self.plan)
        executor.failed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[2])

        self.assertEqual(StepStatus.BLOCKED, status)

    def test_ready_step(self):
        executor = PlanExecutor(self.plan)
        executor.completed_steps.add(self.steps[0].id)

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.READY, status)

    def test_waiting_step(self):
        executor = PlanExecutor(self.plan)
        executor.in_progress_step = self.steps[0].id

        status = executor.get_step_status(self.steps[1])

        self.assertEqual(StepStatus.WAITING, status)

    def test_step_with_no_dependencies_is_ready(self):
        executor = PlanExecutor(self.plan)

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

        executor = PlanExecutor(plan)
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

    def test_get_next_ready_step_returns_first_ready_step(self):
        executor = PlanExecutor(self.plan)
        executor.completed_steps.add(self.steps[0].id)

        step = executor.get_next_ready_step()

        self.assertEqual(self.steps[1], step)

    def test_get_next_ready_step_returns_none_when_no_step_is_ready(self):
        executor = PlanExecutor(self.plan)
        executor.failed_steps.add(self.steps[0].id)

        step = executor.get_next_ready_step()

        self.assertIsNone(step)
