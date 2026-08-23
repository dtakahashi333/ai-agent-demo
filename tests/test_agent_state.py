# tests/test_agent_state.py

from unittest import TestCase

from state.agent_state import AgentState
from state.customer import Customer


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

    def test_select_customer(self):
        state = AgentState()

        data = {
            "id": 42,
            "name": "Alice Smith",
            "email": "alice@example.com",
            "plan": "premium",
        }

        state.select_customer(data)

        self.assertIsNotNone(state.selected_customer)
        self.assertEqual(state.selected_customer.id, 42)

    def test_add_retrieved_results(self):
        state = AgentState()

        state.add_retrieved_results(5)

        self.assertEqual(state.retrieved_count, 5)

        state.add_retrieved_results(2)

        self.assertEqual(state.retrieved_count, 7)

    def test_add_retrieved_results_rejects_negative_count(self):
        state = AgentState()

        with self.assertRaises(ValueError):
            state.add_retrieved_results(-1)

    def test_record_failed_tool_call(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":999}'

        state.record_failed_tool_call(signature)

        self.assertIn(
            signature,
            state.seen_failed_tool_calls,
        )

    def test_record_failed_tool_call_is_idempotent(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":999}'

        state.record_failed_tool_call(signature)
        state.record_failed_tool_call(signature)

        self.assertEqual(
            len(state.seen_failed_tool_calls),
            1,
        )

    def test_increment_iteration(self):
        state = AgentState()

        self.assertEqual(state.iteration, 0)

        state.increment_iteration()

        self.assertEqual(state.iteration, 1)

        state.increment_iteration()

        self.assertEqual(state.iteration, 2)
