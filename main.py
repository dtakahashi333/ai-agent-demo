# main.py
import os
from pprint import pprint

from dotenv import load_dotenv
from openai import OpenAI

from agent import (
    call_llm,
    run_agent,
    tools,
)
from config.agent_config import AgentConfig
from llm.planner_llm import PlannerLLM
from llm.react_llm import ReActLLM
from planner.planner import Planner
from state.agent_state import AgentState

load_dotenv()

# # What is the weather in Dallas, and what is the information for customer 1?
# user_query = input("You: ")

# print(user_query)

# answer = run_agent(user_query, planner_llm=call_llm)

# print(answer)

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

query = """
Find the customer with email alice@example.com, retrieve their recent orders, retrieve their current subscription plan, and create a summary combining the customer information, order history, and subscription details.
"""

agent_config = AgentConfig(
    [
        "Find customer by email",
        "Get customer orders",
        "Get customer subscription plan",
        "Create a customer summary",
    ]
)

model = os.getenv("LLM_MODEL")

planner_llm = PlannerLLM(
    client=client,
    model=model,
)

planner = Planner(
    llm_call=planner_llm,
)

plan = planner.plan(
    objective=query,
    capabilities=agent_config.capabilities,
)

pprint(plan.steps)
