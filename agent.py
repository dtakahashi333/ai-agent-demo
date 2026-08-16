# agent.py
import os
import json
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


client = OpenAI(
    # API keys vary by region. To get an API key, visit: https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # The following base_url is for the Singapore region. If you use a model in the US East 1 (Virginia) region, change the base_url to https://dashscope-us.aliyuncs.com/compatible-mode/v1.
    # If you use a model in the China (Beijing) region, change the base_url to https://dashscope.aliyuncs.com/compatible-mode/v1.
    # base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    base_url="https://ws-a95hgp91msvbk42j.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
)


def run_agent(query: str) -> str:
    # response = client.chat.completions.create(
    #     # This example uses qwen-plus. You can replace it with another model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    #     model="qwen-plus",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": query},
    #     ],
    #     tools=tools,
    #     # extra_body={"enable_thinking": False},
    # )

    # print(response.model_dump_json())
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    response = call_llm(messages)
    message = response.choices[0].message

    print(response.model_dump_json())

    while message.tool_calls:

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

    return message.content


def call_llm(messages: list[any]) -> dict:
    return client.chat.completions.create(
        # This example uses qwen-plus. You can replace it with another model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
        model="qwen-plus",
        messages=messages,
        tools=tools,
        # extra_body={"enable_thinking": False},
    )


def execute_tool_call(tool_call) -> dict:
    tool_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    print(tool_name)
    print(arguments)

    function = tool_functions[tool_name]

    result = function(**arguments)

    print(result)

    return result
