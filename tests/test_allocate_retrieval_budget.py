# tests/allocate_retrieval_budget.py

from unittest import TestCase
from unittest.mock import Mock

from agent import allocate_retrieval_budget
from state.agent_state import AgentState
from config.settings import config


class TestAllocateRetrievalBudget(TestCase):
    def setUp(self):
        config["page_size"] = 5
        config["max_retrieved_results"] = 10
        return super().setUp()

    def test_retrieval_budget_uses_agent_state(self):
        state = AgentState(retrieved_count=5)

        tool_call = Mock()
        tool_call.id = "call_1"
        tool_call.function.name = "search_customers"

        allowed_call_ids = allocate_retrieval_budget(
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

        allowed_call_ids = allocate_retrieval_budget(
            [tool_call],
            state.retrieved_count,
        )

        self.assertEqual(
            allowed_call_ids,
            set(),
        )
