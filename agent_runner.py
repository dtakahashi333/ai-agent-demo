# agent_runner.py
import logging

from executor.plan_executor import PlanExecutionStatus, PlanExecutor
from executor.react_executor import ReActExecutor
from planner.planner import Planner
from planner.replanner import Replanner
from state.agent_state import AgentState

logger = logging.getLogger(__name__)


class AgentRunner:
    def __init__(
        self,
        planner: Planner,
        replanner: Replanner,
        react_executor: ReActExecutor,
        capabilities: list[str],
        max_replans: int = 1,
    ):
        self.planner = planner
        self.replanner = replanner
        self.react_executor = react_executor
        self.capabilities = capabilities
        self.max_replans = max_replans

    def run(self, objective: str) -> str:
        logger.info(
            "Planning agent execution",
            extra={
                "objective": objective,
            },
        )

        state = AgentState()

        plan = self.planner.plan(
            objective=objective,
            capabilities=self.capabilities,
        )

        logger.info(
            "Initial plan created",
            extra={
                "step_count": len(plan.steps),
            },
        )

        for attempt in range(self.max_replans + 1):
            logger.info(
                "Plan execution started",
                extra={
                    "attempt": attempt,
                },
            )

            plan_executor = PlanExecutor(
                plan=plan,
                react_executor=self.react_executor,
            )

            execution_result = plan_executor.execute(state=state)

            if execution_result.status == PlanExecutionStatus.COMPLETED:
                return execution_result.response

            logger.warning(
                "Plan execution requires replanning",
                extra={
                    "attempt": attempt,
                    "completed_steps": list(execution_result.completed_steps),
                    "failed_steps": execution_result.failed_steps,
                },
            )

            if attempt == self.max_replans:
                break

            plan = self.replanner.replan(
                objective=objective,
                capabilities=self.capabilities,
                previous_plan=plan,
                execution_result=execution_result,
                state=state,
            )

            logger.info(
                "Replanned plan created",
                extra={
                    "attempt": attempt,
                    "step_count": len(plan.steps),
                },
            )

        logger.error(
            "Agent execution failed after maximum replanning attempts",
            extra={
                "max_replans": self.max_replans,
                "failed_steps": execution_result.failed_steps,
            },
        )
        raise RuntimeError(
            f"Agent execution failed after {self.max_replans} replanning attempt(s). "
            f"Failed steps: {execution_result.failed_steps}"
        )
