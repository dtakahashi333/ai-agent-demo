# planner/planner.py
from typing import Any

from executor.plan_executor import PlanExecutionResult
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.plan_validator import PlanValidator
from prompts.planner_prompt import PLANNER_SYSTEM_PROMPT

"""
For the initial plan:

User request
+
available tools/capabilities
+
planning instructions

During replanning:

Original user request
+
current AgentState
+
previous plan
+
failure information
+
new observations
"""

"""
User request
      ↓
Planner
      ↓
Plan
      ↓
PlanExecutor
      ↓
ReActExecutor
      ↓
AgentState
      ↓
failure / observation / new fact
      ↓
build PlanningContext
      ↓
Planner
      ↓
new Plan
"""

"""
Planner benefits:

* Explicit workflow structure.
* Observability — you can inspect the intended workflow.
* Predictability — dependencies are explicit rather than implicit in conversation history.
* Progress tracking — you can say exactly which high-level goals are complete.
* Recovery — you can replan from partial progress.
* Longer workflows — the agent doesn't have to rediscover the overall structure after every observation.
* Human/debugging visibility — developers can inspect what the agent intended to accomplish.
"""


class Planner:
    def __init__(
        self,
        llm_call: Any,
    ):
        self.llm_call = llm_call
        self.validator = PlanValidator()
        self.system_prompt = PLANNER_SYSTEM_PROMPT

    def plan(
        self,
        objective: str,
        capabilities: list[str],
        previous_plan: Plan | None = None,
        execution_result: PlanExecutionResult | None = None,
    ) -> Plan:
        """
                    Planner
                      ↑
            ┌─────────┴─────────┐
            │                   │
        objective         capabilities
            │                   │
            └─────────┬─────────┘
                      ↓
                     LLM
                      ↓
                     Plan

        | State                  | Planner probably needs it? |
        |------------------------|----------------------------|
        | retrieved_customer     | Maybe yes                  |
        | retrieved_count        | Maybe                      |
        | seen_failed_tool_calls | Probably not directly      |
        | iteration              | No                         |
        | messages               | No                         |
        """

        # If the Planner has no available capabilities, it should not call the LLM.
        if not capabilities:
            raise ValueError("Planner requires at least one capability.")

        messages = self._build_messages(
            objective,
            capabilities,
        )

        if previous_plan and execution_result:
            messages.append(
                {
                    "role": "user",
                    "content": "Previous Plan\n"
                    + "\n".join(
                        [
                            f"{step.id}: {step.description}"
                            for step in previous_plan.steps
                        ]
                    ),
                }
            )

            execution_lines = [
                f"{step.id} -> "
                + (
                    "completed"
                    if step.id in execution_result.completed_steps
                    else (
                        "failed"
                        if step.id in execution_result.failed_steps
                        else "blocked"
                    )
                )
                for step in previous_plan.steps
            ]

            messages.append(
                {
                    "role": "user",
                    "content": "Execution Result\n" + "\n".join(execution_lines),
                }
            )

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

    def _build_messages(
        self,
        objective: str,
        capabilities: list[str],
    ) -> list[dict[str, str]]:
        """
        What should the prompt specify?
        1. Break the objective into meaningful steps.
        2. Give every step a unique ID.
        3. Describe one concrete objective per step.
        4. Use dependencies to express ordering requirements.
        5. Don't reference nonexistent step IDs.
        6. Don't create circular dependencies.
        7. Make the final steps collectively accomplish the user's objective.
        """
        # capabilities_text = "\n".join(f"- {capability}" for capability in capabilities)

        # return [
        #     {
        #         "role": "system",
        #         "content": self.system_prompt,
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"Objective:\n{objective}\n\n"
        #             f"Available capabilities:\n{capabilities_text}"
        #         ),
        #     },
        # ]
        capabilities_text = "\n".join(f"- {capability}" for capability in capabilities)

        return [
            {
                "role": "system",
                "content": (
                    "You are a planning agent.\n\n"
                    "Create an executable plan to accomplish the user's objective.\n"
                    "Each step must represent one concrete objective that can be "
                    "executed by an agent.\n"
                    "Use only the available capabilities.\n"
                    "Represent dependencies between steps when one step requires "
                    "the result of another step.\n"
                    "Do not create unnecessary steps.\n"
                    "The plan must be logically ordered and must not contain "
                    "circular dependencies."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Objective:\n{objective}\n\n"
                    f"Available capabilities:\n{capabilities_text}"
                ),
            },
        ]


"""
You are a planning agent.

Break the user's objective into a sequence of executable steps.

Each step must:
- have a unique ID
- have a clear description
- list IDs of steps it depends on
- only depend on steps that appear in the plan
- form a valid dependency graph

User objective:
Create a summary of customer John.
"""
