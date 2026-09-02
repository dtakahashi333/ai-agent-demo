# planner/tests/test_planner.py
from unittest import TestCase
from unittest.mock import Mock

from executor.plan_executor import PlanExecutionResult, PlanExecutionStatus
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.plan_validator import PlanValidator
from planner.planner import Planner
from planner.planning_response import PlannedStep
from tests.utils.client_responses import make_planner_client_response


class FakePlannerLLM:
    def __init__(self):
        self.messages = None

    def __call__(self, messages):
        """
        response.choices[0].message.parsed
        """
        self.messages = messages
        return make_planner_client_response(
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


class InvalidPlanPlannerLLM:
    def __init__(self):
        self.messages = None

    def __call__(self, messages):
        self.messages = messages
        return make_planner_client_response(
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


class FailingPlannerLLM:
    def __call__(self, messages):
        raise RuntimeError("LLM unavailable")


class TestPlanner(TestCase):

    def setUp(self):
        super().setUp()
        self.validator = PlanValidator()

    def test_creates_plan(self):
        llm_call = FakePlannerLLM()

        planner = Planner(llm_call=llm_call)

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
        llm_call = FakePlannerLLM()

        planner = Planner(llm_call=llm_call)

        planner.plan(
            "Create a customer summary",
            capabilities=[
                "Find customer",
                "Get customer orders",
                "Get customer plan",
            ],
        )

        self.assertEqual(
            "Create a customer summary",
            llm_call.messages[1]["content"]
            .split("Available capabilities:")[0]
            .replace("Objective:\n", "")
            .strip(),
        )

        self.assertIn(
            "- Find customer",
            llm_call.messages[1]["content"],
        )

        self.assertIn(
            "- Get customer orders",
            llm_call.messages[1]["content"],
        )

        self.assertIn(
            "- Get customer plan",
            llm_call.messages[1]["content"],
        )

    def test_rejects_invalid_plan(self):
        llm_call = InvalidPlanPlannerLLM()

        planner = Planner(llm_call=llm_call)

        with self.assertRaises(ValueError) as context:
            planner.plan(
                "Create a customer summary",
                capabilities=[
                    "Find customer",
                    "Get customer orders",
                    "Get customer plan",
                ],
            )

        self.assertIn(
            "UNKNOWN_DEPENDENCY",
            str(context.exception),
        )

    def test_propagates_llm_error(self):
        llm_call = FailingPlannerLLM()
        planner = Planner(llm_call=llm_call)

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
        llm_call = FakePlannerLLM()
        planner = Planner(llm_call=llm_call)

        with self.assertRaises(ValueError):
            planner.plan(
                "Create a customer summary",
                capabilities=[],
            )

        # Planner rejects an impossible planning request before making an LLM call.
        self.assertIsNone(llm_call.messages)

    def test_sends_capabilities_to_llm(self):
        llm_call = FakePlannerLLM()
        planner = Planner(llm_call=llm_call)

        planner.plan(
            "Create a customer summary",
            capabilities=[
                "Find customer",
                "Get customer orders",
            ],
        )

        user_message = llm_call.messages[1]["content"]

        self.assertIn("Create a customer summary", user_message)
        self.assertIn("Find customer", user_message)
        self.assertIn("Get customer orders", user_message)

    def test_sends_previous_execution_context_to_llm(self):
        mock_planner_llm = Mock()
        mock_planner_llm.return_value = make_planner_client_response(
            steps=[
                PlannedStep(
                    id="A",
                    description="Find Alice",
                    dependencies=[],
                ),
            ]
        )
        planner = Planner(llm_call=mock_planner_llm)

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
            failed_steps={"B"},
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

        messages = mock_planner_llm.call_args.args[0]

        self.assertEqual(
            messages[-2]["content"],
            "Previous Plan\n" "A: Find Alice\n" "B: Get Alice's orders",
        )

        self.assertEqual(
            messages[-1]["content"],
            "Execution Result\n" "A -> completed\n" "B -> failed",
        )
