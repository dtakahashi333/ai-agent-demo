# tests/test_plan.py

from unittest import TestCase

from planner.plan import Plan
from planner.step import PlanStep


class TestPlan(TestCase):
    
    def test_plan_can_contain_steps(self):
        plan = Plan(
            steps=[
                PlanStep(
                    id="find_customer",
                    description="Find Alice Smith",
                ),
                PlanStep(
                    id="get_orders",
                    description="Retrieve customer orders",
                    dependencies=["find_customer"],
                ),
            ]
        )

        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].id, "find_customer")
        self.assertEqual(plan.steps[1].id, "get_orders")
