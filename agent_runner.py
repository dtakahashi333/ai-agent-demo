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

        previous_plan = None
        execution_result = None

        for _ in range(self.max_replans + 1):
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

            execution_result = plan_executor.execute(state=state)

            if execution_result.status == PlanExecutionStatus.COMPLETED:
                return execution_result.response

            previous_plan = plan

        raise RuntimeError(
            f"Agent execution failed after {self.max_replans} replanning attempt(s). "
            f"Failed steps: {execution_result.failed_steps}"
        )
