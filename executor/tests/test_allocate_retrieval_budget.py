# executor/tests/allocate_retrieval_budget.py
from unittest import TestCase
from unittest.mock import Mock

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from state.agent_state import AgentState
from config.settings import config
from tool_registry import build_llm_tools, tool_registry

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)


class TestAllocateRetrievalBudget(TestCase):
    def setUp(self):
        super().setUp()
        self.config = config.copy()
        self.config["page_size"] = 5
        self.config["max_retrieved_results"] = 10
        self.executor = ReActExecutor(
            llm_call=Mock(spec=ReActLLM),
            config=self.config,
        )
        self.tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "search_customers",
                "arguments": '{"name": "Alice"}',
            },
        )

    def test_retrieval_budget_uses_agent_state(self):
        state = AgentState(retrieved_count=5)

        allowed_call_ids = self.executor.allocate_retrieval_budget(
            [self.tool_call],
            state.retrieved_count,
        )

        self.assertEqual(
            allowed_call_ids,
            {"call_1"},
        )

    def test_retrieval_budget_rejects_when_state_budget_is_insufficient(self):
        state = AgentState(retrieved_count=6)

        allowed_call_ids = self.executor.allocate_retrieval_budget(
            [self.tool_call],
            state.retrieved_count,
        )

        self.assertEqual(
            allowed_call_ids,
            set(),
        )
