# tests/test_update_agent_state.py

from unittest import TestCase
from unittest.mock import Mock

from agent import update_agent_state
from models.agent_state import AgentState
from models.customer import Customer


class TestUpdateAgentState(TestCase):
    def test_get_customer_failure_does_not_select_customer(self):
        state = AgentState()

        toolcall = Mock()
        toolcall.function.name = "get_customer"

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "not_found",
                "message": "Customer was not found",
            },
        }

        # Invoke update_agent_state(...)
        update_agent_state(state, toolcall, result)

        self.assertIsNone(state.selected_customer)

    def test_get_customer_success_selects_customer(self):
        state = AgentState()

        toolcall = Mock()
        toolcall.function.name = "get_customer"

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
        update_agent_state(state, toolcall, result)

        self.assertIsNotNone(state.selected_customer)
        self.assertEqual(state.selected_customer.id, 42)
        self.assertEqual(
            state.selected_customer.name,
            "Alice Smith",
        )

    def test_get_customer_failure_preserves_existing_customer(self):
        customer = Customer(
            id=42,
            name="Alice Smith",
            email="alice@example.com",
            plan="premium",
        )

        state = AgentState(selected_customer=customer)

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

        update_agent_state(state, tool_call, result)

        self.assertIsNotNone(state.selected_customer)
        self.assertEqual(state.selected_customer.id, 42)
