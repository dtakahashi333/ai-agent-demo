# executor/plan_executor.py
from dataclasses import dataclass, field
from enum import Enum

from executor.react_executor import ReActExecutor
from planner.plan import Plan
from planner.plan_step import PlanStep
from state.agent_state import AgentState

"""
Planner
   │
   │ creates
   ↓
Plan
   │
   │ interpreted by
   ↓
PlanExecutor
   │
   │ gives objective
   ↓
ReActExecutor
   │
   │ performs
   ↓
Tools
   │
   │ produce observations
   ↓
AgentState
"""

"""
PlanExecutor
    knows PlanStep
    knows plan progress
    knows completed steps

ReActExecutor
    knows objectives
    knows tools
    knows tool execution
    knows AgentState

Tools
    know their own operations
"""

"""
Plan
 ↓
PlanExecutor
 ↓
execute PlanStep
 ↓
ReActExecutor
 ↓
observations
 ↓
PlanExecutor evaluates outcome
 ↓
 ┌───────────────┬──────────────┬──────────────┐
 │               │              │              │
COMPLETED      FAILED       unexpected      ...
 │               │           situation
 ↓               ↓              ↓
next step    retry/recover    REPLAN
                │
                ↓
             possibly
             REPLAN
"""

"""
FAILED
  ↓
Can this be retried?
  ├── yes → retry
  └── no
       ↓
Does the failure invalidate the plan?
  ├── no → fail execution
  └── yes → replan
"""

"""
There are actually three different decisions:

Should I retry this concrete operation?
→ deterministic retry policy.

Has this PlanStep ultimately failed?
→ PlanExecutor.

Does the failure mean the overall plan should change?
→ potentially Planner/replanning.
"""


class StepStatus(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanExecutionStatus(Enum):
    COMPLETED = "completed"
    NEEDS_REPLAN = "needs_replan"


@dataclass
class PlanExecutionResult:
    status: PlanExecutionStatus
    response: str | None = None
    completed_steps: set[str] = field(default_factory=set)
    failed_steps: dict[str, str] = field(default_factory=dict)


class PlanExecutor:
    """
    | State       | Meaning                                                                           |
    |-------------|-----------------------------------------------------------------------------------|
    | WAITING     | Dependencies are not yet satisfied                                                |
    | READY       | All dependencies are completed and the step can run                               |
    | IN_PROGRESS | ReActExecutor is currently working on it                                          |
    | COMPLETED   | The objective was successfully accomplished                                       |
    | FAILED      | Execution of the objective failed                                                 |
    | BLOCKED     | The step cannot execute because a required dependency failed or became impossible |

    Rules:
    No dependencies
        ↓
      READY

    All dependencies COMPLETED
        ↓
      READY

    Selected for execution
        ↓
      IN_PROGRESS

    Objective successfully accomplished
        ↓
      COMPLETED

    Objective cannot be accomplished
        ↓
      FAILED

    Required dependency FAILED/BLOCKED
        ↓
     BLOCKED

    Selection:
    scan plan.steps in order
        ↓
    select first READY step

    Derive:
    COMPLETED
        ↓
    step_id ∈ completed_steps

    FAILED
        ↓
    step_id ∈ failed_steps

    IN_PROGRESS
        ↓
    step_id == active_step

    BLOCKED
        ↓
    a dependency is FAILED/BLOCKED

    READY
        ↓
    all dependencies are COMPLETED

    WAITING
        ↓
    not READY and not BLOCKED
    """

    react_executor: ReActExecutor

    steps: list[PlanStep]
    steps_by_id: dict[str, PlanStep]

    completed_steps: set[str]
    failed_steps: dict[str, str]
    in_progress_step: str | None

    def __init__(
        self,
        plan: Plan,
        react_executor: ReActExecutor,
    ):
        self.react_executor = react_executor

        self.steps = plan.steps
        self.steps_by_id = {step.id: step for step in plan.steps}

        self.completed_steps = set()
        self.failed_steps = {}
        self.in_progress_step = None

        self.response = None

    def get_step_status(self, step: PlanStep) -> StepStatus:
        """
        1. Is it completed? (step ∈ completed_steps)
            ↓ yes → COMPLETED

        2. Is it failed? (step ∈ failed_steps)
            ↓ yes → FAILED

        3. Is it in progress? (step == in_progress_step)
            ↓ yes → IN_PROGRESS

        4. Does any dependency fail or become blocked?
            ↓ yes → BLOCKED

        5. Are all dependencies completed?
            ↓ yes → READY

        6. Otherwise
            ↓
            WAITING
        """

        if step.id in self.completed_steps:
            return StepStatus.COMPLETED

        if step.id in self.failed_steps:
            return StepStatus.FAILED

        if step.id == self.in_progress_step:
            return StepStatus.IN_PROGRESS

        dependency_statuses = [
            self.get_step_status(self.steps_by_id[dependency_id])
            for dependency_id in step.dependencies
        ]

        if any(
            status in (StepStatus.FAILED, StepStatus.BLOCKED)
            for status in dependency_statuses
        ):
            return StepStatus.BLOCKED

        if all(status == StepStatus.COMPLETED for status in dependency_statuses):
            return StepStatus.READY

        return StepStatus.WAITING

    def get_next_ready_step(self) -> PlanStep | None:
        for step in self.steps:
            if self.get_step_status(step) == StepStatus.READY:
                return step

        return None

    def execute(
        self,
        state: AgentState,
    ) -> PlanExecutionResult:
        """
        get next READY step
                ↓
        set in_progress_step
                ↓
        ReActExecutor.execute()
                ↓
        clear in_progress_step
                ↓
             success?
           ┌────┴────┐
          yes        no
           ↓         ↓
        completed  failed
           │         │
           └────┬────┘
                ↓
        next iteration
        """
        while True:
            step = self.get_next_ready_step()

            if step is None:
                if self.completed_steps == set(self.steps_by_id):
                    return PlanExecutionResult(
                        status=PlanExecutionStatus.COMPLETED,
                        response=self.response,
                        completed_steps=set(self.completed_steps),
                    )
                return PlanExecutionResult(
                    status=PlanExecutionStatus.NEEDS_REPLAN,
                    completed_steps=set(self.completed_steps),
                    failed_steps=dict(self.failed_steps),
                )

            self.in_progress_step = step.id

            result = self.react_executor.execute(
                objective=step.description,
                state=state,
            )

            self.in_progress_step = None

            if result.success:
                self.completed_steps.add(step.id)
                self.response = result.response
            else:
                self.failed_steps[step.id] = result.response
