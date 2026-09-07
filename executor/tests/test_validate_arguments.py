# executor/tests/test_validate_arguments.py
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


class TestValidateArguments(TestCase):
    def setUp(self):
        super().setUp()
        self.executor = ReActExecutor(
            llm_call=Mock(spec=ReActLLM),
            config=config,
        )

    def test_accepts_valid_customer_arguments(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {"customer_id": 1},
        )

        self.assertTrue(result["success"])

    def test_rejects_wrong_argument_type(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {"customer_id": "abc"},
        )

        self.assertFalse(result["success"])

    def test_rejects_missing_required_argument(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {},
        )

        self.assertFalse(result["success"])

    def test_rejects_unexpected_argument(self):
        result = self.executor.validate_arguments(
            "get_customer",
            {
                "customer_id": 1,
                "foo": "bar",
            },
        )

        self.assertFalse(result["success"])

    def test_accepts_valid_weather_arguments(self):
        result = self.executor.validate_arguments(
            "get_weather",
            {"city": "Dallas"},
        )

        self.assertTrue(result["success"])
