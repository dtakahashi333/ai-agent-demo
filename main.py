# main.py
from agent import (
    call_llm,
    run_agent,
)
from tools.weather import get_weather
from tools.database import get_customer, get_order

# from tools.calculator import calculator

# result = calculator(20, 5, "multiply")

# print(result)

# What is the weather in Dallas, and what is the information for customer 1?
user_query = input("You: ")

print(user_query)

answer = run_agent(user_query, llm_call=call_llm)

print(answer)
