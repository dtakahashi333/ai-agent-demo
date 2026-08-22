# tool_registry.py
# actual Python functions
from tools.calculator import calculator
from tools.database import (
    count_customers,
    get_customer,
    get_customer_orders,
    get_order,
    get_order_status,
    search_customers,
)
from tools.rag import search_documents
from tools.weather import get_weather

tool_registry = {
    "calculator": {
        "function": calculator,
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
        "retryable": True,
    },
    "search_documents": {
        "function": search_documents,
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
        "retryable": True,
    },
    "get_weather": {
        "function": get_weather,
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city."}
            },
            "required": ["city"],
            "additionalProperties": False,
        },
        "retryable": True,
    },
    "get_customer": {
        "function": get_customer,
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
        "retryable": True,
    },
    "get_order": {
        "function": get_order,
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
        "retryable": True,
    },
    "get_order_status": {
        "function": get_order_status,
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
        "retryable": True,
    },
    "search_customers": {
        "function": search_customers,
        "description": "Returns up to {page_size} matching customers. If more results are available, has_more is true and next_cursor contains the cursor to use as pagination_cursor in the next call.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The customer's name or part of their name to search for.",
                },
                "pagination_cursor": {
                    "type": "integer",
                    "description": "The ID of the last customer from the previous page. Use this value to retrieve the next page. Omit it when requesting the first page.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "retryable": True,
    },
    "get_customer_orders": {
        "function": get_customer_orders,
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
        "retryable": True,
    },
    "count_customers": {
        "function": count_customers,
        "description": "Count the number of customers matching the given name.",
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
        "retryable": True,
    },
}


def build_llm_tools(tool_registry, config):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"].format(
                    page_size=config["page_size"]
                ),
                "parameters": tool["parameters"],
            },
        }
        for name, tool in tool_registry.items()
    ]
