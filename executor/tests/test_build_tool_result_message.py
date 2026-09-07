# executor/tests/test_build_tool_result_message.py
from unittest import TestCase
from unittest.mock import Mock

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from config.settings import config


class TestBuildToolResultMessage(TestCase):
    def setUp(self):
        super().setUp()
        self.react_executor = ReActExecutor(
            llm_call=Mock(spec=ReActLLM),
            config=config,
        )

    def test_builds_tool_message_for_scalar_result(self):
        tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "calculator",
                "arguments": '{"a": 3, "b": 5, "operation": "multiply"}',
            },
        )

        message = self.react_executor.build_tool_result_message(
            tool_call=tool_call,
            result=15,
        )

        self.assertEqual(
            message,
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "15",
            },
        )
