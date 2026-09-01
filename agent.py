# agent.py
import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

from config.agent_config import AgentConfig
from executor.plan_executor import (
    PlanExecutionResult,
    PlanExecutor,
)
from executor.react_executor import ReActExecutor
from llm.planner_llm import PlannerLLM
from llm.react_llm import ReActLLM
from planner.planner import Planner
from state.agent_state import AgentState
from config.settings import config
from tool_registry import build_llm_tools, tool_registry

load_dotenv()

# Tool definitions sent to the LLM
tools = build_llm_tools(tool_registry, config)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

"""
                    PlannerLLM
                       │
                       ▼
query ────────────► Planner
                       │
                       ▼
                      Plan
                       │
                       ▼
                ┌──────────────┐
                │ PlanExecutor │
                └──────┬───────┘
                       │
                       ▼
                ReActExecutor
                       │
                       ▼
                   ReActLLM


AgentState ───────────► PlanExecutor.execute()


run_agent() should:

create or receive the configuration
create the LLM adapters when they aren't injected
construct Planner
construct ReActExecutor
ask Planner for a Plan
construct PlanExecutor with that plan and executor
create the initial AgentState
execute the plan
eventually translate the execution result into the application's final response
"""


def run_agent(
    query: str,
    planner_llm: PlannerLLM = None,
    react_llm: ReActLLM = None,
) -> str:
    """
    run_agent
        │
        ├── creates/configures OpenAI client
        │
        ├── creates PlannerLLM
        │
        ├── creates Planner
        │
        └── creates PlanExecutor

    High-level flow:
    User query
        ↓
    run_agent()
        ↓
    Planner.plan()
        ↓
    Plan
        ↓
    PlanExecutor.execute()
        ↓
    ReActExecutor.execute()
        ↓
    AgentState
        ↓
    LLM / tools
    """
    agent_config = AgentConfig(
        capabilities=[
            "Find customer by email",
            "Get customer orders",
            "Get customer subscription plan",
            "Create a customer summary",
        ]
    )
    state = AgentState()

    model = os.getenv("LLM_MODEL")

    if planner_llm is None:
        planner_llm = PlannerLLM(
            client=client,
            model=model,
        )

    if react_llm is None:
        react_llm = ReActLLM(
            client=client,
            tools=tools,
        )

    planner = Planner(
        llm_call=planner_llm,
    )

    plan = planner.plan(
        objective=query,
        capabilities=agent_config.capabilities,
    )

    react_executor = ReActExecutor(
        llm_call=react_llm,
        config=config,
    )

    plan_executor = PlanExecutor(
        plan=plan,
        react_executor=react_executor,
    )

    result = plan_executor.execute(state)

    return result.response
