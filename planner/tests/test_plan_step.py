# tests/test_plan_step.py

from unittest import TestCase

from planner.plan_step import PlanStep


class TestPlanStep(TestCase):

    def test_step_can_be_created_without_dependencies(self):
        step = PlanStep(
            id="find_customer",
            description="Find Alice Smith",
        )

        self.assertEqual(step.id, "find_customer")
        self.assertEqual(step.description, "Find Alice Smith")
        self.assertEqual(step.dependencies, [])

    def test_step_can_have_dependencies(self):
        step = PlanStep(
            id="get_orders",
            description="Retrieve customer orders",
            dependencies=["find_customer"],
        )

        self.assertEqual(step.dependencies, ["find_customer"])
