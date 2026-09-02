# executor/tests/test_validate_arguments.py
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
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


class TestValidateArguments(TestCase):
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

    def test_validate_arguments1(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {
                "customer_id": 1,
            },
        )
        self.assertTrue(result["success"])

    def test_validate_arguments2(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {
                "customer_id": "abc",
            },
        )
        self.assertFalse(result["success"])

    def test_validate_arguments3(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {},
        )
        self.assertFalse(result["success"])

    def test_validate_arguments4(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {
                "customer_id": 1,
                "foo": "bar",
            },
        )
        self.assertFalse(result["success"])

    def test_validate_arguments5(self):
        result = self.executor.validate_arguments(
            "get_weather",
            {
                "city": "Dallas",
            },
        )
        self.assertTrue(result["success"])
