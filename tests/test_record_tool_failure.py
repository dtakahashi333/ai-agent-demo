# tests/test_record_tool_failure.py

from unittest import TestCase

from agent import record_tool_failure
from state.agent_state import AgentState


class TestRecordToolFailure(TestCase):
    def test_record_tool_failure_records_invalid_arguments(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":"abc"}'

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "invalid_arguments",
                "message": "customer_id must be an integer",
            },
        }

        record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertIn(
            signature,
            state.seen_failed_tool_calls,
        )

    def test_record_tool_failure_does_not_record_not_found(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":999}'

        result = {
            "success": False,
            "data": None,
            "error": {
                "type": "not_found",
                "message": "Customer was not found",
            },
        }

        record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertNotIn(
            signature,
            state.seen_failed_tool_calls,
        )

    def test_record_tool_failure_does_not_record_success(self):
        state = AgentState()

        signature = 'get_customer:{"customer_id":42}'

        result = {
            "success": True,
            "data": {
                "id": 42,
                "name": "Alice Smith",
            },
            "error": None,
        }

        record_tool_failure(
            state,
            signature,
            result,
        )

        self.assertNotIn(
            signature,
            state.seen_failed_tool_calls,
        )
