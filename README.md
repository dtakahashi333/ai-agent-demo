# AI Agent Demo

A production-oriented AI agent built from first principles in Python.

This project demonstrates how to design a reliable tool-using agent without hiding the core architecture behind an agent framework. It combines ReAct execution, semantic agent state, tool safety, retrieval limits, dependency-aware planning, retries, failure recovery, and bounded replanning.

The goal is not to build the most complicated agent possible. The goal is to demonstrate **clear agent architecture, reliability, failure recovery, and engineering judgment**.

---

## What This Project Demonstrates

- ReAct-style tool-using agents
- Structured LLM outputs
- Tool schemas and argument validation
- Safe tool dispatching
- Multiple tool calls and iterations
- Tool-call safety and duplicate-call prevention
- Retryable and non-retryable tool failures
- Retrieval budgets and pagination
- Semantic agent state
- Dependency-aware plan execution
- Initial planning
- Replanning after execution failure
- Preservation of state across replanning
- Parallel execution of independent tool calls
- Bounded retries and replanning
- Application-level logging and debugging
- Unit and integration testing

The implementation intentionally uses **simple Python abstractions** rather than introducing an agent framework.

---

# Architecture

The system is divided into explicit responsibility boundaries:

```
                         User Objective
                                │
                                ▼
                         ┌─────────────┐
                         │ AgentRunner │
                         └──────┬──────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
               ┌─────────┐            ┌───────────┐
               │ Planner │            │ Replanner │
               └────┬────┘            └─────▲─────┘
                    │                       │
                    ▼                       │
                  ┌──────┐                  │
                  │ Plan │                  │
                  └──┬───┘                  │
                     │                      │
                     ▼                      │
              ┌──────────────┐              │
              │ PlanExecutor │──────────────┘
              └──────┬───────┘       failure
                     │
                     ▼
             ┌────────────────┐
             │ ReActExecutor  │
             └───────┬────────┘
                     │
              ┌──────┴──────┐
              ▼             ▼
            Tools       AgentState
```

## Component Responsibilities

### `AgentState`

Owns semantic working state used across agent executions.

Examples include:

- retrieved customer
- retrieved result counts
- failed tool-call signatures
- iteration state
- other semantic information needed by the agent

`AgentState` does **not** construct LLM prompts.

Its responsibility is to represent and expose semantic working state.

---

### `ReActExecutor`

Owns the LLM/tool interaction loop.

Responsibilities include:

- constructing the LLM conversation
- injecting the system prompt
- injecting the objective
- injecting semantic state context
- calling the LLM
- validating tool calls
- dispatching tools
- executing independent approved calls in parallel
- handling tool errors
- retrying retryable failures
- tracking failed tool calls
- enforcing retrieval limits
- updating semantic state
- producing the final response

This is the runtime responsible for **how the agent acts**.

---

### `PlanExecutor`

Owns execution of a validated plan.

Responsibilities include:

- dependency management
- determining which steps are ready
- executing ready steps
- tracking completed steps
- tracking failed steps
- propagating blocked dependencies
- returning an execution result
- determining when execution requires replanning

`PlanExecutor` does not know how prompts are constructed or how individual tools are executed.

---

### `Planner`

Creates the initial execution plan from:

- user objective
- available capabilities

The planner validates the resulting plan before returning it.

---

### `Replanner`

Creates a revised plan when the current plan can no longer complete successfully.

The replanner receives:

- original objective
- available capabilities
- previous plan
- execution result
- failure reasons
- current semantic agent state

This allows the replanner to make decisions based on what has already happened rather than starting from scratch.

---

### `AgentRunner`

Provides top-level orchestration.

Its responsibility is intentionally small:

```
create state
    ↓
create initial plan
    ↓
execute plan
    ↓
completed?
 ┌──┴──┐
yes    no
 │      │
 ▼      ▼
return  replan
          │
          ▼
      execute again
```

Replanning is bounded by `max_replans`.

---

# Replanning

Replanning is one of the central capabilities demonstrated by this project.

A failed tool call does not necessarily mean the entire objective has failed.

The system distinguishes between:

```
Tool-level recovery
        ↓
Retry
```

and:

```
Plan-level recovery
        ↓
Replan
```

The overall flow is:

```
User Objective
      │
      ▼
    Planner
      │
      ▼
    Plan
      │
      ▼
PlanExecutor
      │
      ├── Step succeeds
      │
      └── Step fails
             │
             ▼
       NEEDS_REPLAN
             │
             ▼
         Replanner
             │
             ▼
       Revised Plan
             │
             ▼
       PlanExecutor
             │
             ▼
       Final Response
```

## Example Recovery Scenario

Suppose the user asks:

> Find Alice by email, retrieve her orders, and create a summary.

The initial plan might be:

```
find_customer
      ↓
get_orders
      ↓
create_summary
```

The first step succeeds:

```
find_customer → completed
```

The customer information is stored in `AgentState`:

```
name: Alice
email: alice@example.com
customer_id: 42
plan: pro
```

The next step attempts to retrieve the orders.

Suppose the database temporarily fails and the configured retry policy is exhausted.

`PlanExecutor` returns:

```
status = NEEDS_REPLAN
```

with information such as:

```
completed_steps:
    find_customer

failed_steps:
    get_orders → database failure
```

The `Replanner` receives the original objective, previous plan, execution result, and current semantic state.

It can then produce a revised plan such as:

```
get_orders_retry
      ↓
create_summary
```

The previously completed customer lookup does not need to be repeated.

Execution continues using the existing `AgentState`.

This demonstrates a key property of the architecture:

> **The agent recovers from a failed plan without discarding successful work.**

---

# Retry vs Replan

Retries and replanning solve different problems.

A retry is appropriate when the same operation is still valid but a transient failure may disappear.

For example:

```
get_customer_orders
        ↓
database_error
        ↓
retry
```

If retrying does not recover the operation, the failure may become a planning problem:

```
retry exhausted
        ↓
PlanExecutor
        ↓
NEEDS_REPLAN
        ↓
Replanner
```

This distinction prevents the system from blindly retrying the same failed strategy.

---

# Tool Architecture

Tools are registered explicitly:

```
tool_registry = {
    "calculator": {
        "function": calculator,
        "description": "...",
        "parameters": {...},
        "retryable": True,
    },
}
```

Tool definitions include:

- function implementation
- description
- JSON schema
- retry policy

The executor validates arguments before executing a tool.

Invalid calls are rejected before reaching the underlying function.

---

# Tool Safety

The agent tracks tool-call history to prevent unsafe repetition.

The runtime distinguishes between cases such as:

- duplicate calls in the same iteration
- previously failed calls
- invalid arguments
- unknown tools
- not-found results
- database failures
- retrieval-limit violations

This prevents an LLM from repeatedly issuing an identical failing call indefinitely.

---

# Retrieval Limits and Pagination

Search tools are subject to explicit retrieval limits.

For example:

```
search_customers
       │
       ▼
first page
       │
       ├── has_more = true
       │
       ▼
next_cursor
       │
       ▼
next page
```

The agent is instructed to continue pagination when the objective requires all matching results.

The runtime also enforces a maximum retrieval budget.

This protects the agent from unnecessarily retrieving huge result sets and helps protect the LLM context window.

---

# Semantic Agent State

The project intentionally separates semantic state from conversation history.

For example:

```
AgentState
├── retrieved_customer
├── retrieved_count
├── failed_tool_calls
└── other semantic state
```

The state can be converted into context for the agent:

```
state.get_context()
```

`ReActExecutor` is responsible for putting that context into the LLM conversation.

This keeps the responsibilities separate:

```
AgentState
    → represents semantic state

ReActExecutor
    → decides how that state is presented to the LLM
```

Planning and orchestration components do not construct ReAct prompts.

---

# Plan Execution

Plans contain explicit dependencies.

Example:

```
find_customer
      ↓
get_customer_orders
      ↓
create_summary
```

A step becomes ready only when its dependencies have completed.

The executor can distinguish:

```
WAITING
READY
IN_PROGRESS
COMPLETED
FAILED
BLOCKED
```

This allows dependency failures to propagate without requiring the LLM to manually manage workflow state.

---

# Parallel Tool Execution

Independent tool calls can execute concurrently.

For example:

```
        LLM
       /   \
      ↓     ↓
  get_weather  search_documents
      ↓     ↓
       \   /
        ↓
       LLM
```

Dependent operations remain sequential.

The goal is to parallelize work where there is no dependency while keeping dependency-sensitive operations ordered.

---

# Error Handling

Tool failures are represented at the execution boundary.

Examples include:

```
invalid_arguments
unknown_tool
not_found
database_error
tool_execution_error
```

Retryable database failures can be retried according to configuration.

Unexpected tool exceptions are contained by the executor rather than crashing the entire agent loop.

At the plan level, unrecoverable execution failures become:

```
NEEDS_REPLAN
```

At the top level, the agent has a bounded number of replanning attempts.

---

# Observability

The application uses Python's standard `logging` module.

Logs can be written to both the console and a rotating log file:

```
logs/
└── agent.log
```

The most important orchestration events include:

```
Initial plan created
Plan execution started
Plan execution requires replanning
Replanned plan created
Agent execution completed
```

Logging is configured at the application boundary rather than inside individual agent components.

Runtime logs are excluded from version control.

---

# Configuration

Runtime behavior is controlled through configuration rather than hard-coded execution policies.

Examples include:

```
LLM_MODEL
max_iterations
max_retries
retry_delay
max_retrieved_results
max_replans
execution_timeout
```

This keeps operational limits separate from the core agent architecture.

---

# Testing Philosophy

Tests are used to protect meaningful behavioral and architectural contracts.

The project intentionally avoids maximizing test count or coverage for its own sake.

Testing focuses on:

- semantic state behavior
- tool validation
- tool execution
- retry behavior
- loop safety
- retrieval limits
- dependency execution
- failure propagation
- replanning behavior
- end-to-end recovery

Integration behavior is preferred when a feature spans multiple components.

The goal is:

```
meaningful contract
      ↓
test
      ↓
integration
      ↓
move forward
```

rather than adding tests simply to increase coverage.

---

# Design Decisions

## Explicit architecture instead of an agent framework

The project intentionally implements the core runtime directly in Python.

This makes the important mechanics visible:

- tool dispatch
- validation
- state
- planning
- dependency execution
- retries
- replanning
- failure recovery

A framework such as LangGraph can be evaluated later, but introducing one now would hide many of the concepts this project is intended to demonstrate.

## Semantic state is separate from prompts

`AgentState` owns semantic information.

`ReActExecutor` owns prompt construction.

This prevents prompt-specific concerns from leaking into the state model.

## Tool return formats are not unnecessarily normalized

Different tools may have different native return formats.

For example, the calculator can return:

```
15
```

while database tools may return structured envelopes.

Normalization happens only where the orchestration boundary actually requires it.

## Retries are bounded

A transient failure should not result in an infinite retry loop.

Both retries and replanning are explicitly bounded.

## Replanning preserves successful work

A revised plan should use information from previous execution rather than restarting blindly.

This is why completed steps, failure information, and semantic state cross the replan boundary.

---

# Project Structure

The exact structure may evolve, but the major responsibilities are organized around the agent runtime:

```
project/
├── agent/
│   ├── agent_runner.py
│   ├── agent_state.py
│   ├── planner.py
│   ├── replanner.py
│   ├── plan_executor.py
│   └── react_executor.py
│
├── tools/
│   ├── calculator.py
│   ├── customers.py
│   ├── orders.py
│   └── ...
│
├── prompts/
│   └── agent_prompt.py
│
├── tests/
│   └── ...
│
├── logs/
│   └── agent.log
│
└── README.md
```

---

# Running the Agent

Set the required environment variables, including the model configuration:

```
export LLM_MODEL=<model-name>
```

Then run the application through the project's entry point.

A typical objective is:

```
Find Alice by email, retrieve her orders, and create a summary of her orders.
```

The application will:

1. Create an initial plan.
2. Execute the plan.
3. Use ReAct/tool execution for individual steps.
4. Retry retryable failures.
5. Trigger replanning when execution cannot continue.
6. Preserve semantic state.
7. Execute the revised plan.
8. Return the final response.

---

# Running Tests

Run the complete test suite with:

```
pytest
```

For development/debugging, logs can be displayed during pytest execution with:

```
pytest -o log_cli=true --log-cli-level=INFO
```

---

# Roadmap

The core agent runtime is already implemented.

Future areas include:

- side-effecting tools
- idempotency
- transaction safety
- authorization
- prompt-injection defenses
- richer observability
- evaluation datasets
- production API architecture
- persistent agent state
- MCP
- comparison with agent frameworks such as LangGraph
- multi-agent systems

These are intentionally introduced only after the underlying agent runtime is understood and working.

---

# Why This Project Exists

Modern agent frameworks make it easy to assemble an agent, but they can also hide the mechanisms that determine whether an agent is reliable.

This project takes the opposite approach.

It builds those mechanisms explicitly and demonstrates how they interact:

```
LLM
 ↓
Tool calling
 ↓
Validation
 ↓
Execution
 ↓
Semantic state
 ↓
Planning
 ↓
Dependency management
 ↓
Failure handling
 ↓
Replanning
 ↓
Recovery
```

The result is a compact agent runtime that emphasizes **engineering judgment over framework complexity**.

This is intentionally written as a **portfolio README**, not as a complete implementation manual. It gives a reviewer the architecture, the interesting recovery story, and the design rationale without drowning them in implementation details.

I would **not add more sections right now**. Once your side-effect/tool-safety work begins, we can update the README with that material rather than trying to predict the final architecture today.