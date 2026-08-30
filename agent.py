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


def call_llm(messages: list[any], tool_choice=None) -> dict:
    return client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        # extra_body={"enable_thinking": False},
    )


def mock_call_llm(messages):
    # First LLM response: deliberately make an invalid tool call.
    if len(messages) == 2:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="get_customer",
                                    arguments=json.dumps({"customer_id": "abc"}),
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    # Second LLM response: deliberately repeat the exact same call.
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_2",
                            function=SimpleNamespace(
                                name="get_customer",
                                arguments=json.dumps({"customer_id": "abc"}),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def run_agent(
    query: str,
    planner_llm=None,
    react_llm=None,
) -> PlanExecutionResult:
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
    agent_config = AgentConfig()
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

    return plan_executor.execute(state)
