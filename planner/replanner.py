# planner/replanner.py

from typing import Any

from executor.plan_executor import PlanExecutionResult
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.plan_validator import PlanValidator
from prompts.replanner_prompt import REPLANNER_SYSTEM_PROMPT
from state.agent_state import AgentState

"""
                   objective
                      │
                      ▼
                  ┌────────┐
                  │ Planner│
                  └───┬────┘
                      │
                     Plan
                      │
                      ▼
               ┌─────────────┐
               │PlanExecutor │
               └──────┬──────┘
                      │
                 failure
                      │
          ┌───────────┴───────────┐
          │                       │
   execution_result          AgentState
          │                 get_context()
          └───────────┬───────────┘
                      ▼
                 ┌─────────┐
                 │Replanner│
                 └────┬────┘
                      │
                 revised Plan
"""


class Replanner:
    def __init__(self, llm_call: Any):
        self.llm_call = llm_call
        self.validator = PlanValidator()
        self.system_prompt = REPLANNER_SYSTEM_PROMPT

    def replan(
        self,
        objective: str,
        capabilities: list[str],
        previous_plan: Plan,
        execution_result: PlanExecutionResult,
        state: AgentState,
    ) -> Plan:
        # If the Planner has no available capabilities, it should not call the LLM.
        if not capabilities:
            raise ValueError("Planner requires at least one capability.")

        capabilities_text = "\n".join(f"- {capability}" for capability in capabilities)
        previous_plan_context = self._format_previous_plan(plan=previous_plan)
        execution_result_context = self._format_execution_result(
            plan=previous_plan,
            execution_result=execution_result,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Objective:\n{objective}\n\n"
                    f"Available capabilities:\n{capabilities_text}\n\n"
                    f"{previous_plan_context}\n\n"
                    f"{execution_result_context}\n\n"
                    f"Current Agent State:\n{state.get_context()}"
                ),
            },
        ]

        response = self.llm_call(messages=messages)

        planning_response = response.choices[0].message.parsed

        steps = [
            PlanStep(
                id=step.id,
                description=step.description,
                dependencies=step.dependencies,
            )
            for step in planning_response.steps
        ]

        plan = Plan(steps=steps)

        # Validate before returning
        errors = self.validator.validate(plan=plan)

        if errors:
            # raise ValueError(f"Invalid plan: {errors}")
            raise ValueError(errors)

        return plan

    def _format_previous_plan(self, plan: Plan) -> str:
        return "Previous Plan\n" + "\n".join(
            [f"{step.id}: {step.description}" for step in plan.steps]
        )

    def _format_execution_result(
        self,
        plan: Plan,
        execution_result: PlanExecutionResult,
    ) -> str:
        execution_lines = [
            f"{step.id} -> "
            + (
                "completed"
                if step.id in execution_result.completed_steps
                else (
                    f"failed: {execution_result.failed_steps[step.id]}"
                    if step.id in execution_result.failed_steps
                    else "blocked"
                )
            )
            for step in plan.steps
        ]

        return "Execution Result\n" + "\n".join(execution_lines)
