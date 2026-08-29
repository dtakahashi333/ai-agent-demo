# planner/tests/test_validator.py

from unittest import TestCase

from planner.plan import Plan
from planner.step import PlanStep
from planner.validator import PlanValidator


class TestPlanValidator(TestCase):

    def setUp(self):
        super().setUp()
        self.validator = PlanValidator()

    def test_empty_plan(self):
        plan = Plan(steps=[])

        errors = self.validator.validate(plan)

        self.assertEqual(1, len(errors))
        self.assertEqual("EMPTY_PLAN", errors[0].code)

    def test_valid_plan(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual([], errors)

    def test_duplicate_step_id(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="A", description="Get orders", dependencies=[]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual(1, len(errors))
        self.assertEqual("DUPLICATE_STEP_ID", errors[0].code)

    def test_unknown_dependency(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(id="B", description="Get orders", dependencies=["X"]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual(1, len(errors))
        self.assertEqual("UNKNOWN_DEPENDENCY", errors[0].code)

    def test_multiple_unknown_dependencies(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=["X"]),
            PlanStep(id="B", description="Get orders", dependencies=["Y"]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        codes = [error.code for error in errors]

        self.assertEqual(2, len(errors))
        self.assertEqual(2, codes.count("UNKNOWN_DEPENDENCY"))

    def test_direct_circular_dependency(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=["B"]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual(1, len(errors))
        self.assertEqual("CIRCULAR_DEPENDENCY", errors[0].code)

    def test_multi_step_circular_dependency(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=["C"]),
            PlanStep(id="B", description="Get orders", dependencies=["A"]),
            PlanStep(id="C", description="Count orders", dependencies=["B"]),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual(1, len(errors))
        self.assertEqual("CIRCULAR_DEPENDENCY", errors[0].code)

    def test_valid_branching_dependency_graph(self):
        steps = [
            PlanStep(id="A", description="Find customer", dependencies=[]),
            PlanStep(
                id="B", description="Retrieve customer orders", dependencies=["A"]
            ),
            PlanStep(id="C", description="Retrieve customer plan", dependencies=["A"]),
            PlanStep(
                id="D",
                description="Create customer order summary",
                dependencies=["B", "C"],
            ),
        ]

        plan = Plan(steps)

        errors = self.validator.validate(plan)

        self.assertEqual([], errors)
