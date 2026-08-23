# agent.py
from concurrent.futures import ThreadPoolExecutor
import os
import json
import time
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

from models.agent_state import AgentState
from config import config
from models.customer import Customer
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

# same invalid call was already attempted
repeated_tool_call_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "repeated_tool_call",
        "message": (
            "The same tool call was already attempted and "
            "failed. Do not repeat it; change the approach."
        ),
    },
}
# same call appeared twice in the same response
duplicate_tool_call_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "duplicate_tool_call",
        "message": (
            "This exact tool call was already requested "
            "in the current tool-call batch."
        ),
    },
}
# found more customers than LLM can safely retrieve
too_many_results_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "too_many_results",
        "message": (
            "The requested result set exceeds the maximum "
            "retrieval limit of 100 customers."
        ),
    },
}
retrieval_limit_exceeded_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "retrieval_limit_exceeded",
        "message": (
            "This tool call was not executed because the "
            "agent's maximum retrieval limit was reached."
        ),
    },
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

    state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Ask the LLM what to do next
    message = call_agent_llm(state.messages, llm_call)

    print("\n".join(str(tool_call) for tool_call in message.tool_calls))

    while message.tool_calls and state.iteration < config["max_iterations"]:

        state.iteration += 1

        # Record what the assistant requested
        state.messages.append(message.model_dump())

        results = process_tool_call_batch(message, state)

        state.messages += results

        if (
            estimate_message_tokens(state.messages)
            > config["max_estimated_context_tokens"]
        ):
            return (
                "Agent stopped because the estimated context "
                "size exceeds the maximum allowed token count."
            )

        # Ask the LLM what to do next
        message = call_agent_llm(state.messages, llm_call)

        # print(response.model_dump_json())

    if message.tool_calls:
        return "Agent stopped because the maximum iteration limit was reached."

    # print(json.dumps(state.messages))

    return message.content


def call_agent_llm(
    messages: list[dict],
    llm_call,
):
    response = llm_call(messages)
    return response.choices[0].message


def process_tool_call_batch(
    message,
    state: AgentState,
) -> list[dict]:
    """
    process_tool_call_batch()
    │
    ├── Tool-call safety
    │   ├── retrieval budget
    │   ├── validation
    │   └── failed-call tracking
    │
    ├── Tool execution
    │
    └── Tool result
        ↓
    update_agent_state()
        ↓
    semantic working state
    """

    # Agent-level budget allocation
    allowed_call_ids = allocate_retrieval_budget(
        message.tool_calls,
        state.retrieved_count,
    )

    # Validate each requested tool call
    seen_tool_calls_in_iteration = set()

    approved_calls = []
    results = []

    # Then process each call
    for tool_call in message.tool_calls:
        if tool_call.id not in allowed_call_ids:
            results.append(
                build_tool_result_message(
                    tool_call,
                    retrieval_limit_exceeded_error,
                )
            )
            continue

        policy, signature = validate_tool_call(
            tool_call,
            state.seen_failed_tool_calls,
            seen_tool_calls_in_iteration,
        )

        if policy == "duplicate":
            results.append(
                build_tool_result_message(
                    tool_call,
                    duplicate_tool_call_error,
                )
            )
        elif policy == "repeated":
            results.append(
                build_tool_result_message(
                    tool_call,
                    repeated_tool_call_error,
                )
            )
        elif policy == "allowed":
            approved_calls.append((tool_call, signature))

    # Execute approved calls in parallel
    executed_calls = execute_approved_calls(approved_calls)

    for tool_call, tool_call_signature, result in executed_calls:
        """
        record_tool_failure() answers:
        > Did this execution produce a failure that affects future tool-call safety?

        update_agent_state() answers:
        > Did this successful tool result change what the agent knows?
        """

        record_tool_failure(state, tool_call_signature, result)

        update_agent_state(state, tool_call, result)

        results.append(
            build_tool_result_message(
                tool_call,
                result,
            )
        )

    return results


def allocate_retrieval_budget(
    tool_calls,
    retrieved_count: int,
) -> set:
    remaining = config["max_retrieved_results"] - retrieved_count
    allowed = set()

    for tool_call in tool_calls:
        if tool_call.function.name != "search_customers":
            allowed.add(tool_call.id)
            continue

        if remaining < config["page_size"]:
            continue

        allowed.add(tool_call.id)
        remaining -= config["page_size"]

    return allowed


def execute_approved_calls(approved_calls) -> list[tuple]:

    if not approved_calls:
        return []

    with ThreadPoolExecutor(max_workers=len(approved_calls)) as executor:

        futures = [
            executor.submit(execute_tool_call, tool_call)
            for tool_call, _ in approved_calls
        ]

        return [
            (tool_call, signature, future.result())
            for (tool_call, signature), future in zip(approved_calls, futures)
        ]


def build_tool_result_message(tool_call, result) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result),
    }


def record_tool_failure(
    state: AgentState,
    tool_call_signature,
    result: dict,
) -> None:
    if result["success"] is False and result["error"]["type"] == "invalid_arguments":
        state.seen_failed_tool_calls.add(tool_call_signature)


def update_agent_state(
    state: AgentState,
    tool_call,
    result: dict,
) -> None:
    if result["success"] is False:
        return

    if tool_call.function.name == "search_customers":
        if result["success"] is True:
            state.retrieved_count += len(result["data"]["customers"])

    if tool_call.function.name == "get_customer":
        if result["success"] is True:
            state.select_customer(result["data"])


def estimate_message_tokens(messages) -> int:
    return len(json.dumps(messages)) // 4


def validate_tool_call(
    tool_call,
    seen_failed_tool_calls: set,
    seen_tool_calls_in_iteration: set,
) -> tuple:
    signature = make_tool_call_signature(tool_call)

    policy = check_tool_call_policy(
        signature,
        seen_tool_calls_in_iteration,
        seen_failed_tool_calls,
    )

    if policy == "allowed":
        # Record the call BEFORE executing it
        seen_tool_calls_in_iteration.add(signature)

    return policy, signature


def make_tool_call_signature(tool_call) -> tuple:
    print(type(tool_call.function.arguments))
    print(tool_call.function.arguments)
    # Deserialization = serialized representation → usable Python object
    arguments = json.loads(tool_call.function.arguments)
    # Sort JSON keys so equivalent arguments produce the same signature.
    # Serialization = Python object → format suitable for storage/transmission
    return (tool_call.function.name, json.dumps(arguments, sort_keys=True))


def check_tool_call_policy(
    signature,
    seen_tool_calls_in_iteration,
    seen_failed_tool_calls,
) -> str:
    if signature in seen_tool_calls_in_iteration:
        return "duplicate"

    if signature in seen_failed_tool_calls:
        return "repeated"

    return "allowed"


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


def execute_tool_call(tool_call) -> dict:

    print("EXECUTING:", tool_call.function.name)

    start = time.perf_counter()

    print(f"[START {tool_call.function.name}] " f"{start:.4f}")

    tool_name = tool_call.function.name

    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "invalid_arguments",
                "message": "Tool arguments contain invalid JSON",
            },
        }

    if tool_name not in tool_registry:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "unknown_tool",
                "message": f"Unknown tool: {tool_name}",
            },
        }

    validation_result = validate_arguments(tool_name, arguments)

    if not validation_result["success"]:
        return validation_result

    function = tool_registry[tool_name]["function"]

    try:
        for attempt in range(config["max_retries"] + 1):

            result = function(**arguments)

            if result["success"]:
                return result

            if result["error"]["type"] != "database_error":
                return result

            if not tool_registry[tool_name]["retryable"]:
                return result

            if attempt < config["max_retries"]:
                time.sleep(config["retry_delay"])

        return result
    except Exception:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "tool_execution_error",
                "message": "Tool execution failed",
            },
        }
    finally:
        elapsed = time.perf_counter() - start

        print(f"[END {tool_call.function.name}] " f"{elapsed:.4f}s")


def validate_arguments(tool_name: str, arguments: dict) -> dict:
    # # Manual implementation
    # schema = tool_schemas[tool_name]["parameters"]

    # properties = schema.get("properties", {})
    # required = schema.get("required", [])
    # additional_properties = schema.get("additionalProperties", True)

    # # Check required arguments
    # missing = [name for name in required if name not in arguments]

    # if missing:
    #     return {
    #         "success": False,
    #         "data": None,
    #         "error": {
    #             "type": "invalid_arguments",
    #             "message": f"{missing[0]} is required",
    #         },
    #     }

    # # Validate supplied arguments
    # for name, value in arguments.items():

    #     # Unknown property
    #     if name not in properties:
    #         if not additional_properties:
    #             return {
    #                 "success": False,
    #                 "data": None,
    #                 "error": {
    #                     "type": "invalid_arguments",
    #                     "message": f"{name} is not allowed",
    #                 },
    #             }

    #         # Additional properties are allowed,
    #         # so there is no schema to validate against.
    #         continue

    #     schema_type = properties[name].get("type")

    #     if schema_type == "integer":
    #         if not isinstance(value, int) or isinstance(value, bool):
    #             return {
    #                 "success": False,
    #                 "data": None,
    #                 "error": {
    #                     "type": "invalid_arguments",
    #                     "message": f"{name} must be an integer",
    #                 },
    #             }

    #     elif schema_type == "string":
    #         if not isinstance(value, str):
    #             return {
    #                 "success": False,
    #                 "data": None,
    #                 "error": {
    #                     "type": "invalid_arguments",
    #                     "message": f"{name} must be a string",
    #                 },
    #             }

    # return {
    #     "success": True,
    #     "data": None,
    #     "error": None,
    # }

    schema = tool_registry[tool_name]["parameters"]

    try:
        validate(instance=arguments, schema=schema)

        return {
            "success": True,
            "data": None,
            "error": None,
        }

    except ValidationError as e:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "invalid_arguments",
                "message": e.message,
            },
        }
