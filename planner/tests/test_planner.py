# planner/tests/test_planner.py
from unittest import TestCase
from unittest.mock import Mock

from executor.plan_executor import PlanExecutionResult, PlanExecutionStatus
from llm.planner_llm import PlannerLLM
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planner import Planner
from planner.planning_response import PlannedStep
from tests.utils.client_responses import make_planner_client_response


class TestPlanner(TestCase):
    def setUp(self):
        super().setUp()
        self.fake_planner_llm = Mock(spec=PlannerLLM)
        self.fake_planner_llm.return_value = make_planner_client_response(
            steps=[
                PlannedStep(
                    id="A",
                    description="Find customer",
                    dependencies=[],
                ),
                PlannedStep(
                    id="B",
                    description="Get orders",
                    dependencies=["A"],
                ),
            ]
        )

        self.invalid_plan_planner_llm = Mock(spec=PlannerLLM)
        self.invalid_plan_planner_llm.return_value = make_planner_client_response(
            steps=[
                PlannedStep(
                    id="A",
                    description="Find customer",
                    dependencies=[],
                ),
                PlannedStep(
                    id="B",
                    description="Get orders",
                    dependencies=["X"],
                ),
            ]
        )

        self.failing_planner_llm = Mock(spec=PlannerLLM)
        self.failing_planner_llm.side_effect = RuntimeError("LLM unavailable")

    def test_creates_plan(self):
        planner = Planner(llm_call=self.fake_planner_llm)

        plan = planner.plan(
            "Create a customer summary",
            capabilities=[
                "Find customer",
                "Get customer orders",
                "Get customer plan",
            ],
        )

        self.assertEqual(2, len(plan.steps))

        self.assertEqual("A", plan.steps[0].id)
        self.assertEqual("Find customer", plan.steps[0].description)
        self.assertEqual([], plan.steps[0].dependencies)

        self.assertEqual("B", plan.steps[1].id)
        self.assertEqual("Get orders", plan.steps[1].description)
        self.assertEqual(["A"], plan.steps[1].dependencies)

    def test_sends_objective_and_capabilities_to_llm(self):
        planner = Planner(llm_call=self.fake_planner_llm)

        planner.plan(
            "Create a customer summary",
            capabilities=[
                "Find customer",
                "Get customer orders",
                "Get customer plan",
            ],
        )

        messages = self.fake_planner_llm.call_args.kwargs["messages"]

        self.assertEqual(
            "Create a customer summary",
            messages[1]["content"]
            .split("Available capabilities:")[0]
            .replace("Objective:\n", "")
            .strip(),
        )

        self.assertIn("- Find customer", messages[1]["content"])

        self.assertIn("- Get customer orders", messages[1]["content"])

        self.assertIn("- Get customer plan", messages[1]["content"])

    def test_rejects_invalid_plan(self):
        planner = Planner(llm_call=self.invalid_plan_planner_llm)

        with self.assertRaises(ValueError) as context:
            planner.plan(
                "Create a customer summary",
                capabilities=[
                    "Find customer",
                    "Get customer orders",
                    "Get customer plan",
                ],
            )

        self.assertIn("UNKNOWN_DEPENDENCY", str(context.exception))

    def test_propagates_llm_error(self):
        planner = Planner(llm_call=self.failing_planner_llm)

        with self.assertRaises(RuntimeError):
            planner.plan(
                "Create a customer summary",
                capabilities=[
                    "Find customer",
                    "Get customer orders",
                    "Get customer plan",
                ],
            )

    def test_rejects_empty_capabilities(self):
        planner = Planner(llm_call=self.fake_planner_llm)

        with self.assertRaises(ValueError):
            planner.plan(
                "Create a customer summary",
                capabilities=[],
            )

        self.fake_planner_llm.assert_not_called()

    def test_sends_previous_execution_context_to_llm(self):
        planner_llm = Mock(spec=PlannerLLM)
        planner_llm.return_value = make_planner_client_response(
            steps=[
                PlannedStep(
                    id="A",
                    description="Find Alice",
                    dependencies=[],
                ),
            ]
        )
        planner = Planner(llm_call=planner_llm)

        previous_plan = Plan(
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

        execution_result = PlanExecutionResult(
            status=PlanExecutionStatus.NEEDS_REPLAN,
            completed_steps={"A"},
            failed_steps={"B": "Orders service unavailable"},
        )

        planner.plan(
            objective="Find Alice and get her orders",
            capabilities=[
                "Find customer",
                "Get customer orders",
            ],
            previous_plan=previous_plan,
            execution_result=execution_result,
        )

        messages = planner_llm.call_args.kwargs["messages"]

        self.assertEqual(
            messages[-2]["content"],
            "Previous Plan\n" "A: Find Alice\n" "B: Get Alice's orders",
        )

        self.assertEqual(
            messages[-1]["content"],
            "Execution Result\n"
            "A -> completed\n"
            "B -> failed: Orders service unavailable",
        )
