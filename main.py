# main.py
import json
from types import SimpleNamespace

from agent import run_agent, execute_tool_call, run_second_agent
from tools.weather import get_weather
from tools.database import get_customer

# from tools.calculator import calculator

# result = calculator(20, 5, "multiply")

# print(result)

user_query = input("You: ")

print(user_query)

run_agent(user_query)

# weather = get_weather("Dallas")

# print(weather)

# tool_calls = [
#     SimpleNamespace(
#         id="call_d59653a0ff764ad3a39573",
#         function=SimpleNamespace(
#             arguments=json.dumps({"city": "Dallas"}),
#             name="get_weather",
#         ),
#         type="function",
#         index=0,
#     )
# ]

# result = execute_tool_call(tool_calls[0])

# print(result)

# print(get_customer(1))

# query ="What is the information for customer 1?"

# message = {
#     "content": "",
#     "refusal": None,
#     "role": "assistant",
#     "annotations": None,
#     "audio": None,
#     "function_call": None,
#     "tool_calls": [
#         {
#             "id": "call_e699f6d1b30541099968cd",
#             "function": {"arguments": '{"customer_id": 1}', "name": "get_customer"},
#             "type": "function",
#             "index": 0,
#         }
#     ],
# }

# result = {
#     "success": True,
#     "data": {"id": 1, "name": "Alice", "email": "alice@example.com", "plan": "pro"},
#     "error": None,
# }

# tool_call = {
#     "id": message["tool_calls"][0]["id"],
#     "result": result,
# }

# print(json.dumps(tool_call))

# run_second_agent(query, message, tool_call)