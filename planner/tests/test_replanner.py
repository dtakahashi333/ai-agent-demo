# planner/tests/test_replanner.py
from unittest import TestCase
from unittest.mock import Mock

from executor.plan_executor import PlanExecutionResult, PlanExecutionStatus
from llm.planner_llm import PlannerLLM
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planning_response import PlannedStep
from planner.replanner import Replanner
from state.agent_state import AgentState
from tests.utils.client_responses import make_planner_client_response


class TestReplanner(TestCase):
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
        replanner = Replanner(llm_call=planner_llm)

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

        replanner.replan(
            objective="Find Alice and get her orders",
            capabilities=[
                "Find customer",
                "Get customer orders",
            ],
            previous_plan=previous_plan,
            execution_result=execution_result,
            state=AgentState(),
        )

        messages = planner_llm.call_args.kwargs["messages"]

        self.assertIn(
            "Previous Plan\n" "A: Find Alice\nB: Get Alice's orders",
            messages[1]["content"],
        )

        self.assertIn(
            (
                "Execution Result\n"
                "A -> completed\n"
                "B -> failed: Orders service unavailable"
            ),
            messages[1]["content"],
        )
