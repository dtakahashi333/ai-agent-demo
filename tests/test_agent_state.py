# tests/test_agent_state.py

from unittest import TestCase

from agent_state import AgentState


class TestAgentState(TestCase):
    def setUp(self):
        return super().setUp()

    def test_agent_state_rejects_negative_iteration(self):
        self.assertRaises
        with self.assertRaises(ValueError):
            AgentState(iteration=-1)

    def test_agent_state_rejects_negative_retrieved_count(self):
        with self.assertRaises(ValueError):
            AgentState(retrieved_count=-1)

    def test_agent_state_defaults(self):
        state = AgentState()

        self.assertTrue(state.iteration == 0)
        self.assertTrue(state.retrieved_count == 0)
        self.assertTrue(state.selected_customer is None)
        self.assertTrue(state.seen_failed_tool_calls == set())
