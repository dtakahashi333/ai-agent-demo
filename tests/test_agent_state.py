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
        self.assertIsNone(state.retrieved_customer)
        self.assertEqual(state.seen_failed_tool_calls, set())

    def test_agent_state_selected_customer(self):
        customer = Customer(
            id=42,
            name="Alice Smith",
            email="alice@example.com",
            plan="premium",
        )

        state = AgentState(retrieved_customer=customer)

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

    def test_select_customer(self):
        state = AgentState()

        data = {
            "id": 42,
            "name": "Alice Smith",
            "email": "alice@example.com",
            "plan": "premium",
        }

        state.select_customer(data)

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

    def test_select_customer_invalid_data_preserves_existing_customer(self):
        """
        old customer
        │
        ├── valid Customer → replace
        │
        └── invalid Customer → exception
                               ↓
                         old customer remains
        """
        existing_customer = Customer(
            id=42,
            name="Alice Smith",
            email="alice@example.com",
            plan="premium",
        )

        state = AgentState(retrieved_customer=existing_customer)

        invalid_data = {
            "id": 0,
            "name": "Invalid Customer",
            "email": "invalid@example.com",
            "plan": "basic",
        }

        with self.assertRaises(ValueError):
            state.select_customer(invalid_data)

        self.assertIs(state.retrieved_customer, existing_customer)

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

    def test_add_messages(self):
        state = AgentState()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        state.add_messages(messages)

        self.assertEqual(state.messages, messages)

    def test_add_messages_appends_to_existing_messages(self):
        state = AgentState(messages=[{"role": "system", "content": "System"}])

        new_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        state.add_messages(new_messages)

        self.assertEqual(
            state.messages,
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        )

    def test_initialize_messages(self):
        state = AgentState()

        state.initialize_messages(
            "You are a helpful assistant.",
            "Find Alice.",
        )

        self.assertEqual(
            state.messages,
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "Find Alice.",
                },
            ],
        )

    def test_initialize_messages_replaces_existing_messages(self):
        state = AgentState(
            messages=[
                {
                    "role": "user",
                    "content": "Old message",
                }
            ]
        )

        state.initialize_messages(
            "System",
            "New request",
        )

        self.assertEqual(
            state.messages,
            [
                {
                    "role": "system",
                    "content": "System",
                },
                {
                    "role": "user",
                    "content": "New request",
                },
            ],
        )
