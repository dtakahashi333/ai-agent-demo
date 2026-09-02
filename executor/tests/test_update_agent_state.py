# executor/tests/test_update_agent_state.py
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from state.agent_state import AgentState
from state.customer import Customer
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


class TestUpdateAgentState(TestCase):
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

    def test_get_customer_failure_does_not_select_customer(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "get_customer"

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "not_found",
                "message": "Customer was not found",
            },
        }

        # Invoke update_agent_state(...)
        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertIsNone(state.retrieved_customer)

    def test_get_customer_success_selects_customer(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "get_customer"

        result = {
            "success": True,
            "data": {
                "id": 42,
                "name": "Alice Smith",
                "email": "alice@example.com",
                "plan": "premium",
            },
            "error": None,
        }

        # Invoke update_agent_state(...)
        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 42)
        self.assertEqual(
            state.retrieved_customer.name,
            "Alice Smith",
        )
        self.assertEqual(
            state.retrieved_customer.email,
            "alice@example.com",
        )
        self.assertEqual(
            state.retrieved_customer.plan,
            "premium",
        )

    def test_get_customer_success_replaces_existing_customer(self):
        state = AgentState(
            retrieved_customer=Customer(
                id=42,
                name="Alice Smith",
                email="alice@example.com",
                plan="premium",
            )
        )

        tool_call = Mock()
        tool_call.function.name = "get_customer"

        result = {
            "success": True,
            "data": {
                "id": 84,
                "name": "Bob Jones",
                "email": "bob@example.com",
                "plan": "basic",
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 84)
        self.assertEqual(
            state.retrieved_customer.name,
            "Bob Jones",
        )
        self.assertEqual(
            state.retrieved_customer.email,
            "bob@example.com",
        )
        self.assertEqual(
            state.retrieved_customer.plan,
            "basic",
        )

    def test_get_customer_failure_preserves_existing_customer(self):
        customer = Customer(
            id=42,
            name="Alice Smith",
            email="alice@example.com",
            plan="premium",
        )

        state = AgentState(retrieved_customer=customer)

        tool_call = Mock()
        tool_call.function.name = "get_customer"

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "not_found",
                "message": "Customer was not found",
            },
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 42)
        self.assertEqual(
            state.retrieved_customer.email,
            "alice@example.com",
        )
        self.assertEqual(
            state.retrieved_customer.plan,
            "premium",
        )

    def test_search_customers_adds_retrieved_count(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "search_customers"

        result = {
            "success": True,
            "data": {
                "customers": [
                    {
                        "id": 1,
                        "name": "Alice Smith",
                        "email": "alice@example.com",
                        "plan": "premium",
                    },
                    {
                        "id": 2,
                        "name": "Alice Jones",
                        "email": "alice.jones@example.com",
                        "plan": "basic",
                    },
                    {
                        "id": 3,
                        "name": "Alice Brown",
                        "email": "alice.brown@example.com",
                        "plan": "premium",
                    },
                ],
                "has_more": True,
                "next_cursor": 3,
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 3)

    def test_search_customers_accumulates_retrieved_count(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "search_customers"

        first_result = {
            "success": True,
            "data": {
                "customers": [
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                ],
                "has_more": True,
                "next_cursor": 3,
            },
            "error": None,
        }

        second_result = {
            "success": True,
            "data": {
                "customers": [
                    {"id": 4},
                    {"id": 5},
                ],
                "has_more": False,
                "next_cursor": None,
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            first_result,
        )

        self.assertEqual(state.retrieved_count, 3)

        self.executor.update_agent_state(
            state,
            tool_call,
            second_result,
        )

        self.assertEqual(state.retrieved_count, 5)

    def test_search_customers_failure_does_not_increment_retrieved_count(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "search_customers"

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve customers",
            },
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 0)

    def test_search_customers_failure_preserves_retrieved_count(self):
        state = AgentState(retrieved_count=5)

        tool_call = Mock()
        tool_call.function.name = "search_customers"

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "database_error",
                "message": "Unable to retrieve customers",
            },
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 5)

    def test_get_customer_does_not_change_retrieved_count(self):
        state = AgentState(retrieved_count=5)

        tool_call = Mock()
        tool_call.function.name = "get_customer"

        result = {
            "success": True,
            "data": {
                "id": 42,
                "name": "Alice Smith",
                "email": "alice@example.com",
                "plan": "premium",
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 5)

    def test_search_customers_does_not_select_customer(self):
        state = AgentState()

        tool_call = Mock()
        tool_call.function.name = "search_customers"

        result = {
            "success": True,
            "data": {
                "customers": [
                    {
                        "id": 42,
                        "name": "Alice Smith",
                        "email": "alice@example.com",
                        "plan": "premium",
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 1)
        self.assertIsNone(state.retrieved_customer)

    def test_get_weather_does_not_modify_agent_state(self):
        state = AgentState(
            retrieved_count=5,
        )

        tool_call = Mock()
        tool_call.function.name = "get_weather"

        result = {
            "success": True,
            "data": {
                "city": "Dallas",
                "temperature": 96.3,
            },
            "error": None,
        }

        self.executor.update_agent_state(
            state,
            tool_call,
            result,
        )

        self.assertEqual(state.retrieved_count, 5)
        self.assertIsNone(state.retrieved_customer)
