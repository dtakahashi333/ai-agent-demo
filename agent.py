# agent.py
import os
import json
from time import time
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

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

# same invalid call was already attempted
repeated_tool_call_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "repeated_tool_call",
        "message": "The same tool call was already attempted and failed. Do not repeat it; change the approach.",
    },
}
# same call appeared twice in the same response
duplicate_tool_call_error = {
    "success": False,
    "data": None,
    "error": {
        "type": "duplicate_tool_call",
        "message": "This exact tool call was already requested in the current tool-call batch.",
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

    if llm_call is None:
        llm_call = call_llm

    agent_policy = "Agent retrieval policy:\n"
    agent_policy += (
        f"- Maximum total customers that may be retrieved: "
        f"{config['max_retrieved_results']}"
    )

    system_prompt = SYSTEM_PROMPT + "\n\n" + agent_policy

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = llm_call(messages)
    message = response.choices[0].message

    # print(response.model_dump_json())

    counter = 0
    seen_failed_tool_calls = set()
    retrieved_count = 0

    while message.tool_calls and counter < config["max_iterations"]:

        counter += 1

        # First: record what the assistant requested
        messages.append(message)

        # Then: execute each requested tool
        seen_tool_calls_in_iteration = set()
        results = []

        for tool_call in message.tool_calls:

            if (
                tool_call.function.name == "search_customers"
                and retrieved_count >= config["max_retrieved_results"]
            ):
                result = too_many_results_error
            else:
                result = process_tool_call(
                    tool_call,
                    seen_failed_tool_calls,
                    seen_tool_calls_in_iteration,
                )

            if tool_call.function.name == "search_customers":
                data = json.loads(result["content"])
                if data["success"]:
                    retrieved_count += len(data["data"]["customers"])

            results.append(result)

        messages += results

        # Ask the LLM what to do next
        response = llm_call(messages)
        message = response.choices[0].message

        # print(response.model_dump_json())

    if message.tool_calls:
        return "Agent stopped because the maximum iteration limit was reached."

    # print(json.dumps(messages))

    return message.content


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


def build_tool_result_message(tool_call, result) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result),
    }


def process_tool_call(
    tool_call,
    seen_failed_tool_calls: set,
    seen_tool_calls_in_iteration: set,
) -> dict:
    signature = make_tool_call_signature(tool_call)

    policy = check_tool_call_policy(
        signature,
        seen_tool_calls_in_iteration,
        seen_failed_tool_calls,
    )

    if policy == "duplicate":
        return build_tool_result_message(
            tool_call,
            duplicate_tool_call_error,
        )

    elif policy == "repeated":
        return build_tool_result_message(
            tool_call,
            repeated_tool_call_error,
        )

    else:
        # Record the call BEFORE executing it
        seen_tool_calls_in_iteration.add(signature)

        result = execute_tool_call(tool_call)

        if (
            result["success"] == False
            and result["error"]["type"] == "invalid_arguments"
        ):
            seen_failed_tool_calls.add(signature)

        return build_tool_result_message(
            tool_call,
            result,
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


def execute_tool_call(tool_call) -> dict:

    print("EXECUTING:", tool_call.function.name)

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

            print(result)

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
