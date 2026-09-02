# executor/tests/allocate_retrieval_budget.py
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import Mock

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from state.agent_state import AgentState
from config.settings import config
from tool_registry import build_llm_tools, tool_registry

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)

mock_react_client = Mock()
mock_react_client.chat.completions.create.return_value = SimpleNamespace(
    choices=[
        SimpleNamespace(
            message=SimpleNamespace(
                content="Customer found",
                tool_calls=[],
            )
        )
    ]
)


class TestAllocateRetrievalBudget(TestCase):
    def setUp(self):
        super().setUp()
        config["page_size"] = 5
        config["max_retrieved_results"] = 10
        self.executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

    def test_retrieval_budget_uses_agent_state(self):
        state = AgentState(retrieved_count=5)

        tool_call = Mock()
        tool_call.id = "call_1"
        tool_call.function.name = "search_customers"

        allowed_call_ids = self.executor.allocate_retrieval_budget(
            [tool_call],
            state.retrieved_count,
        )

        self.assertEqual(
            allowed_call_ids,
            {"call_1"},
        )

    def test_retrieval_budget_rejects_when_state_budget_is_insufficient(self):
        state = AgentState(retrieved_count=6)

        tool_call = Mock()
        tool_call.id = "call_1"
        tool_call.function.name = "search_customers"

        allowed_call_ids = self.executor.allocate_retrieval_budget(
            [tool_call],
            state.retrieved_count,
        )

        self.assertEqual(
            allowed_call_ids,
            set(),
        )
