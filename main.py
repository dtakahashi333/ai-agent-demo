# main.py
import json
from types import SimpleNamespace

from agent import mock_call_llm, run_agent, execute_tool_call
from tools.weather import get_weather
from tools.database import get_customer, get_order

# from tools.calculator import calculator

# result = calculator(20, 5, "multiply")

# print(result)

# What is the weather in Dallas, and what is the information for customer 1?
user_query = input("You: ")

print(user_query)

run_agent(user_query, llm_call=mock_call_llm)

# print(get_order(102))
