# agent_runner.py
from executor.plan_executor import PlanExecutionStatus, PlanExecutor
from executor.react_executor import ReActExecutor
from planner.planner import Planner
from state.agent_state import AgentState


class AgentRunner:
    def __init__(
        self,
        planner: Planner,
        react_executor: ReActExecutor,
        capabilities: list[str],
        max_replans: int = 1,
    ):
        self.planner = planner
        self.react_executor = react_executor
        self.capabilities = capabilities
        self.max_replans = max_replans

    def run(self, objective: str) -> str:
        state = AgentState()

        # Count replanning attempts, not initial planning.
        replan_count = 0

        previous_plan = None
        execution_result = None

        while True:
            plan = self.planner.plan(
                objective=objective,
                capabilities=self.capabilities,
                previous_plan=previous_plan,
                execution_result=execution_result,
            )

            plan_executor = PlanExecutor(
                plan=plan,
                react_executor=self.react_executor,
            )

            result = plan_executor.execute(state=state)

            if result.status == PlanExecutionStatus.COMPLETED:
                return result.response

            replan_count += 1

            if replan_count > self.max_replans:
                raise RuntimeError(
                    f"Agent execution failed after {replan_count} replanning attempt(s). "
                    f"Failed steps: {result.failed_steps}"
                )

            previous_plan = plan
            execution_result = result
