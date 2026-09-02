# executor/tests/test_record_tool_failure.py
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from state.agent_state import AgentState
from tool_registry import build_llm_tools, tool_registry
from config.settings import config

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


class TestRecordToolFailure(TestCase):
    def setUp(self):
        super().setUp()
        self.executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

    def test_record_tool_failure_records_invalid_arguments(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":"abc"}'

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "invalid_arguments",
                "message": "customer_id must be an integer",
            },
        }

        self.executor.record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertIn(
            signature,
            state.seen_failed_tool_calls,
        )

    def test_record_tool_failure_does_not_record_not_found(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":999}'

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "not_found",
                "message": "Customer was not found",
            },
        }

        self.executor.record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertNotIn(
            signature,
            state.seen_failed_tool_calls,
        )

    def test_record_tool_failure_does_not_record_success(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":42}'

        result = {
            "success": True,
            "data": {
                "id": 42,
                "name": "Alice Smith",
            },
            "error": None,
        }

        self.executor.record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertNotIn(
            signature,
            state.seen_failed_tool_calls,
        )
