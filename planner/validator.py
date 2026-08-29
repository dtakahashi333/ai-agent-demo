# planner/validator.py
from dataclasses import dataclass, field

from planner.plan import Plan
from planner.step import PlanStep


@dataclass
class PlanValidationError(Exception):
    """
    Potential errors could be:

    * DUPLICATE_STEP_ID
    * UNKNOWN_DEPENDENCY
    * CIRCULAR_DEPENDENCY
    * EMPTY_PLAN
    * EMPTY_STEP_ID
    * EMPTY_DESCRIPTION

    For example:
    PlanValidationError(
        code="UNKNOWN_DEPENDENCY",
        message="Step 'get_orders' depends on unknown step 'find_customer'.",
    )

    Then the validator could conceptually return:

    errors = validator.validate(plan)

    if errors:
        ...

    rather than throwing immediately.

    the Planner could receive something like:

    The generated plan is invalid.

    Errors:

    1. UNKNOWN_DEPENDENCY
    Step "get_orders" depends on "find_customer",
    but "find_customer" does not exist.

    2. CIRCULAR_DEPENDENCY
    Steps form a cycle:
    A -> B -> C -> A
    """

    code: str
    message: str


"""
PlanValidator
    |
    +-- duplicate step IDs
    +-- unknown dependency IDs
    +-- circular dependencies
    +-- empty plan
"""

"""
Rule 1 — Plan cannot be empty
[]
should be invalid.
There's nothing to execute.

Rule 2 — Step IDs must be unique
This is invalid:
A: Find Alice
A: Get Alice's orders
Because dependencies can no longer unambiguously refer to a step.

Rule 3 — Dependencies must exist
This is invalid:
A: Find Alice
B: Get orders
   depends_on = ["X"]
because X doesn't exist.

Rule 4 — No circular dependencies
This is invalid:
A → B
B → C
C → A
because no step can become executable.
"""

"""
User request
     ↓
 Planner LLM
     ↓
   Plan
     ↓
PlanValidator
     ↓
 ┌───────────────┐
 │ valid?        │
 └───────┬───────┘
       yes│       no
          │        │
          ↓        ↓
       Executor   errors
                     │
                     ↓
                Planner LLM
                     │
                     ↓
                corrected Plan
"""


class PlanValidator:
    """
    Take a generated Plan and determine whether its structure is safe and coherent enough to execute.

    Structural validation
    PlanValidator can safely check things such as:

    - Is the plan empty?
    - Are step IDs unique?
    - Do dependencies reference existing steps?
    - Is there a circular dependency?

    These are properties of the plan graph itself.

    Feasibility validation
    Later, we can consider questions such as:

    - Can the available tools accomplish the goals?
    - Does the plan fit within retrieval limits?
    - Is it compatible with execution constraints?
    - Is the requested workflow actually possible?

    Those are more contextual.

    valid plan
        → return normally

    invalid plan
        → raise an exception
    """

    def validate(self, plan: Plan) -> list[PlanValidationError]:
        errors = []

        if not plan.steps:
            return [
                PlanValidationError(
                    "EMPTY_PLAN",
                    "The plan contains no steps.",
                )
            ]

        errors.extend(self._find_duplicate_step_ids(plan))
        unknown_dependency_errors = self._find_unknown_dependencies(plan)
        errors.extend(unknown_dependency_errors)

        if not unknown_dependency_errors:
            if self._has_cycle(plan):
                errors.append(
                    PlanValidationError(
                        "CIRCULAR_DEPENDENCY",
                        "The plan contains a circular dependency.",
                    )
                )

        return errors

    def _find_duplicate_step_ids(
        self,
        plan: Plan,
    ) -> list[PlanValidationError]:
        seen_ids = set()
        reported_duplicates = set()
        errors = []

        for step in plan.steps:
            if step.id in seen_ids:
                if step.id not in reported_duplicates:
                    errors.append(
                        PlanValidationError(
                            "DUPLICATE_STEP_ID",
                            f"Duplicate plan step ID: '{step.id}'.",
                        )
                    )
                    reported_duplicates.add(step.id)
            else:
                seen_ids.add(step.id)

        return errors

    def _find_unknown_dependencies(
        self,
        plan: Plan,
    ) -> list[PlanValidationError]:
        known_step_ids = {step.id for step in plan.steps}
        errors = []

        for step in plan.steps:
            for dependency_id in step.dependencies:
                if dependency_id not in known_step_ids:
                    errors.append(
                        PlanValidationError(
                            "UNKNOWN_DEPENDENCY",
                            (
                                f"Step '{step.id}' depends on unknown "
                                f"step '{dependency_id}'."
                            ),
                        )
                    )

        return errors

    def _has_cycle(self, plan: Plan) -> bool:
        steps_by_id = {step.id: step for step in plan.steps}

        visiting = set()
        visited = set()

        def visit(step_id: str) -> bool:
            if step_id in visited:
                return False

            if step_id in visiting:
                return True

            visiting.add(step_id)

            for dependency_id in steps_by_id[step_id].dependencies:
                if visit(dependency_id):
                    return True

            visiting.remove(step_id)
            visited.add(step_id)

            return False

        for step in plan.steps:
            if visit(step.id):
                return True

        return False
