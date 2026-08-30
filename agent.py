# agent.py
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

from executor.react_executor import ReActExecutor
from state.agent_state import AgentState
from config import config
from tool_registry import build_llm_tools, tool_registry

from prompts.agent_prompt import SYSTEM_PROMPT

load_dotenv()

# Tool definitions sent to the LLM
tools = build_llm_tools(tool_registry, config)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)


RETRYABLE_ERRORS = {
    "timeout",
    "temporary_database_error",
}

"""
| Result            | Meaning                                                     |
| ------------------|-------------------------------------------------------------|
| success           | Useful information was obtained                             |
| invalid_arguments | The requested action was invalid                            |
| not_found         | The action was valid, but the requested entity wasn't found |
| database_error    | Infrastructure failed                                       |
| timeout           | Infrastructure may have failed temporarily                  |
"""

"""
| Situation                                         | Action  |
|---------------------------------------------------|---------|
| First invalid_arguments call                      | Execute |
| Identical invalid_arguments call later            | Block   | 
| Identical call in same response after first fails | Block   |
| Successful call repeated                          | Allow   |
| not_found repeated                                | Allow   |
| database_error repeated                           | Allow   |
| Different arguments                               | Allow   |
| Different tool                                    | Allow   |
"""

"""
| Situation                              | Action          |
|----------------------------------------|-----------------|
| Same invalid call in later iteration   | Block           |
| Same call twice in one response        | Block duplicate |
| Same not_found call in later iteration | Allow           |
| New arguments                          | Allow           |
| New tool                               | Allow           |
"""

"""
| Execution state   | Execution policy   |
|-------------------|--------------------|
| iteration_count   | max_iterations     |
| tool_calls_used   | max_tool_calls     |
| retries_performed | max_retries        |
| elapsed_time.     | max_execution_time |
"""

"""
Tool execution
│
├── State
│   └── attempt
│
└── Policy
    ├── max_retries
    ├── retry_delay
    └── retryable
"""

"""
Agent execution
│
├── State
│   ├── counter
│   ├── messages
│   └── seen_failed_tool_calls
│
└── Policy
    └── max_iterations
"""

"""
Conversation state
└── messages

Execution state
├── iteration counter
└── seen_failed_tool_calls

This distinction becomes important if you ever build:

* durable agents
* background agents
* pause/resume
* crash recovery
* long-running workflows
* agent checkpoints
"""


def run_agent(
    query: str,
    llm_call=None,
) -> str:
    """
    If the LLM needs to know it → put it in messages
    Examples:
    * user request
    * previous tool calls
    * tool results
    * pagination results
    * previous assistant responses

    If only the Python agent needs it → keep it as execution state
    Examples:
    * iteration counter
    * duplicate-call tracking
    * retry counters
    * internal policy bookkeeping
    """

    state = AgentState()

    if llm_call is None:
        llm_call = call_llm

    agent_policy = "Agent retrieval policy:\n"
    agent_policy += (
        f"- Maximum total customers that may be retrieved: "
        f"{config['max_retrieved_results']}\n"
    )

    agent_policy += (
        "When a tool returns has_more=true and the user's request "
        "requires all matching results, continue retrieving pages "
        "using next_cursor. Do not claim that all results have been "
        "retrieved until has_more=false.\n"
    )

    agent_policy += (
        "When multiple requested tool calls are independent, request them "
        "together in the same tool-call response so they can be executed "
        "in parallel. Do not wait for one independent call to finish before "
        "requesting another.\n"
    )

    system_prompt = SYSTEM_PROMPT + "\n\n" + agent_policy

    state.initialize_messages(system_prompt, query)

    executor = ReActExecutor(llm_call=call_llm)

    return executor.execute(state)


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
