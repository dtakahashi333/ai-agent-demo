# main.py
import os
from pprint import pprint

from dotenv import load_dotenv
from openai import OpenAI

from config.agent_config import AgentConfig
from executor.plan_executor import PlanExecutionResult, PlanExecutionStatus
from llm.planner_llm import PlannerLLM
from planner.plan import Plan
from planner.plan_step import PlanStep
from planner.planner import Planner
from planner.replanner import Replanner
from state.agent_state import AgentState
from state.customer import Customer

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

# planner = Planner(
#     llm_call=planner_llm,
# )

# plan = planner.plan(
#     objective=query,
#     capabilities=agent_config.capabilities,
# )

# pprint(plan.steps)

replanner = Replanner(
    llm_call=planner_llm,
)

state = AgentState()

# Populate this the same way your real ReActExecutor would.
state.retrieved_customer = Customer(
    id=42,
    name="Alice",
    email="alice@example.com",
    plan="pro",
)

previous_plan = Plan(
    steps=[
        PlanStep(
            id="find_customer",
            description="Find the customer by email alice@example.com",
            dependencies=[],
        ),
        PlanStep(
            id="get_orders",
            description="Get all orders for the customer",
            dependencies=["find_customer"],
        ),
        PlanStep(
            id="create_summary",
            description="Create a summary of the customer's orders",
            dependencies=["get_orders"],
        ),
    ]
)

execution_result = PlanExecutionResult(
    status=PlanExecutionStatus.NEEDS_REPLAN,
    completed_steps={"find_customer"},
    failed_steps={"get_orders": "Database connection temporarily unavailable"},
)

new_plan = replanner.replan(
    objective=(
        "Find the customer with email alice@example.com "
        "and create a summary of her orders."
    ),
    capabilities=[
        "Find customer by email",
        "Get customer orders",
        "Get customer subscription plan",
        "Create a customer summary",
    ],
    previous_plan=previous_plan,
    execution_result=execution_result,
    state=state,
)

for step in new_plan.steps:
    print(
        step.id,
        "->",
        step.description,
        "dependencies:",
        step.dependencies,
    )
