# executor/tests/test_execute_tool_call.py
from unittest import TestCase
from unittest.mock import Mock, patch

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from config.settings import config
from tool_registry import tool_registry


class TestExecuteToolCall(TestCase):
    def setUp(self):
        super().setUp()
        self.react_executor = ReActExecutor(
            llm_call=Mock(spec=ReActLLM),
            config=config,
        )

    def test_executes_calculator_tool(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "calculator",
                "arguments": '{"a": 3, "b": 5, "operation": "multiply"}',
            },
        )

        # Mock the actual registered tool here rather than the LLM.
        result = self.react_executor.execute_tool_call(tool_call=tool_call)

        self.assertEqual(result, 15)

    def test_executes_customer_tool(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "get_customer",
                "arguments": '{"customer_id": 1}',
            },
        )

        # Mock the actual registered tool here rather than the LLM.
        result = self.react_executor.execute_tool_call(tool_call=tool_call)

        self.assertEqual(
            result["data"],
            {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "plan": "pro",
            },
        )

    def test_returns_tool_execution_error_when_tool_raises(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "calculator",
                "arguments": '{"a": 10, "b": 0, "operation": "divide"}',
            },
        )

        result = self.react_executor.execute_tool_call(tool_call=tool_call)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["type"], "tool_execution_error")

    def test_retries_retryable_database_error(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "get_customer",
                "arguments": '{"customer_id": 1}',
            },
        )

        mock_tool = Mock(
            side_effect=[
                {
                    "success": False,
                    "data": None,
                    "error": {
                        "type": "database_error",
                        "message": "Temporary database failure",
                    },
                },
                {
                    "success": True,
                    "data": {
                        "id": 1,
                        "name": "Alice",
                        "email": "alice@example.com",
                        "plan": "pro",
                    },
                    "error": None,
                },
            ]
        )

        original_function = tool_registry["get_customer"]["function"]
        tool_registry["get_customer"]["function"] = mock_tool

        try:
            with patch("time.sleep"):
                result = self.react_executor.execute_tool_call(tool_call)
        finally:
            tool_registry["get_customer"]["function"] = original_function

        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["name"], "Alice")
        self.assertEqual(mock_tool.call_count, 2)
