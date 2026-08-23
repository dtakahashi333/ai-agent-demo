# tests/test_agent_state.py

from unittest import TestCase

from models.agent_state import AgentState
from models.customer import Customer


class TestAgentState(TestCase):
    def setUp(self):
        return super().setUp()

    def test_agent_state_rejects_negative_iteration(self):
        with self.assertRaises(ValueError):
            AgentState(iteration=-1)

    def test_agent_state_rejects_negative_retrieved_count(self):
        with self.assertRaises(ValueError):
            AgentState(retrieved_count=-1)

    def test_agent_state_defaults(self):
        state = AgentState()

        self.assertEqual(state.iteration, 0)
        self.assertEqual(state.retrieved_count, 0)
        self.assertIsNone(state.selected_customer)
        self.assertEqual(state.seen_failed_tool_calls, set())

    def test_agent_state_selected_customer(self):
        customer = Customer(
            id=42,
            name="Alice Smith",
            email="alice@example.com",
            plan="premium",
        )

        state = AgentState(selected_customer=customer)

        self.assertIsNotNone(state.selected_customer)
        self.assertEqual(state.selected_customer.id, 42)
        self.assertEqual(
            state.selected_customer.name,
            "Alice Smith",
        )
