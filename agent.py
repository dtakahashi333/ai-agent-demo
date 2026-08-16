# agent.py
import os
import json
from time import time
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from openai import OpenAI
from dotenv import load_dotenv

from tools.calculator import calculator
from tools.database import (
    get_customer,
    get_customer_orders,
    get_order,
    get_order_status,
    search_customers,
)
from tools.rag import search_documents
from tools.weather import get_weather

from prompts.agent_prompt import SYSTEM_PROMPT

load_dotenv()

# Your actual Python functions
from tools.calculator import calculator
from tools.rag import search_documents
from tools.weather import get_weather

# Tool definitions sent to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a mathematical operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform.",
                    },
                },
                "required": ["a", "b", "operation"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the company's knowledge base for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or search query to look up in the knowledge base.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city."}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Retrieve a customer's information using their customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The unique ID of the customer.",
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve an order using its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The unique ID of the order.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Retrieve the current status of an order using its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The unique ID of the order.",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_customers",
            "description": "Search for customers by name. Returns all customers whose names contain the search term.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The customer's name or part of their name to search for.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "Retrieve all orders belonging to a customer using the customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The unique ID of the customer.",
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
]


# Python functions that the application can actually execute
tool_functions = {
    "calculator": calculator,
    "search_documents": search_documents,
    "get_weather": get_weather,
    "get_customer": get_customer,
    "get_order": get_order,
    "get_order_status": get_order_status,
    "search_customers": search_customers,
    "get_customer_orders": get_customer_orders,
}

tool_config = {
    "calculator": {"retryable": True},
    "search_documents": {"retryable": True},
    "get_weather": {"retryable": True},
    "get_customer": {"retryable": True},
    "get_order": {"retryable": True},
    "get_order_status": {"retryable": True},
    "search_customers": {"retryable": True},
    "get_customer_orders": {"retryable": True},
}

tool_schemas = {
    "get_weather": {
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city."}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    "get_customer": {
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The unique ID of the customer.",
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    "search_customers": {
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name to search for.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

client = OpenAI(
    # API keys vary by region. To get an API key, visit: https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # The following base_url is for the Singapore region. If you use a model in the US East 1 (Virginia) region, change the base_url to https://dashscope-us.aliyuncs.com/compatible-mode/v1.
    # If you use a model in the China (Beijing) region, change the base_url to https://dashscope.aliyuncs.com/compatible-mode/v1.
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    base_url="https://ws-a95hgp91msvbk42j.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)


MAX_ITERATIONS = 10
RETRYABLE_ERRORS = {
    "timeout",
    "temporary_database_error",
}


def run_agent(query: str, max_iterations: int = 10) -> str:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    response = call_llm(messages)
    message = response.choices[0].message

    print(response.model_dump_json())

    counter = 0
    while message.tool_calls and counter < max_iterations:

        counter += 1

        # First: record what the assistant requested
        messages.append(message)

        # Then: execute each requested tool
        results = []
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call)

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        messages += results

        # Ask the LLM what to do next
        response = call_llm(messages)
        message = response.choices[0].message

        print(response.model_dump_json())

    if message.tool_calls:
        return "Agent stopped because the maximum iteration limit was reached."

    return message.content


def call_llm(messages: list[any]) -> dict:
    return client.chat.completions.create(
        # This example uses qwen-plus. You can replace it with another model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        model="qwen-plus",
        messages=messages,
        tools=tools,
        # extra_body={"enable_thinking": False},
    )


def execute_tool_call(
    tool_call, max_retries: int = 1, retry_delay: float = 1.0
) -> dict:

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

    print(tool_name)
    print(arguments)

    validation_result = validate_arguments(tool_name, arguments)

    if not validation_result["success"]:
        return validation_result

    if tool_name not in tool_functions:
        return {
            "success": False,
            "data": None,
            "error": {
                "type": "unknown_tool",
                "message": f"Unknown tool: {tool_name}",
            },
        }

    function = tool_functions[tool_name]

    try:
        for attempt in range(max_retries + 1):

            result = function(**arguments)

            print(result)

            if result["success"]:
                return result

            if result["error"]["type"] != "database_error":
                return result

            if not tool_config[tool_name]["retryable"]:
                return result

            if attempt < max_retries:
                time.sleep(retry_delay)

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

    schema = tool_schemas[tool_name]["parameters"]

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
